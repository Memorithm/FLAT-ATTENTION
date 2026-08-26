//! Deterministic WGSL emission from the FLAT Kernel IR (roadmap M21).
//!
//! The emitter walks the validated [`KernelModule`] phase program and produces
//! canonical WGSL text for the dense Q4 forward realizations:
//!
//! - scalar portable (M2/M4 lineage),
//! - vec4 portable (M6 lineage),
//! - double-buffered vec4 (M7 lineage),
//! - subgroup-assisted reduction (M5 lineage).
//!
//! Guarantees:
//!
//! - **Determinism:** identical modules produce byte-identical sources; no
//!   HashMap iteration, timestamps, random ids or adapter state participate.
//! - **Bounded output:** every emitted quantity derives from closed IR
//!   enumerations; no user-controlled repetition exists, and a hard source
//!   budget is enforced before returning.
//! - **Contract preservation:** bindings, uniform layout, packed `[O | LSE]`
//!   storage and dispatch geometry match the qualified handwritten shaders
//!   exactly, so generated pipelines are directly comparable against them.
//!
//! Generated sources must still pass Naga parsing/validation and device-level
//! qualification before any routing use; emission alone is not qualification.
//! This module makes no performance claim.

use crate::kernel_ir::{
    KernelConfig, KernelFamily, KernelIrVersion, KernelModule, KvStaging, ScoreReduction,
    VectorWidth,
};
use std::fmt;

/// Version of the WGSL generator. Any change to emitted text structure must
/// bump this version so derived source identities invalidate cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodegenVersion {
    /// Incompatible emitted-source change; invalidates derived identities.
    pub major: u32,
    /// Backward-compatible refinement within one major generator schema.
    pub minor: u32,
}

impl CodegenVersion {
    /// Generator schema produced by this crate.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
}

impl fmt::Display for CodegenVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Deterministic identity of one generated source.
///
/// Combines the IR identity with the generator identity. Runtime adapter
/// state is deliberately excluded; capability relevance lives inside the IR
/// itself (for example subgroup requirements), not in the source key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSourceKey {
    /// Schema version of the IR that produced the module.
    pub ir_version: KernelIrVersion,
    /// Schema version of this generator.
    pub codegen_version: CodegenVersion,
    /// Structural fingerprint of the source module.
    pub ir_fingerprint: u64,
}

impl KernelSourceKey {
    /// Canonical deterministic record used for cache keys and evidence.
    #[must_use]
    pub fn canonical_record(&self) -> String {
        format!(
            "ir=v{};cg=v{};irfp={:016x}",
            self.ir_version, self.codegen_version, self.ir_fingerprint
        )
    }

    /// Stable FNV-1a-64 fingerprint of [`Self::canonical_record`]; a cache
    /// key accelerator, never authentication.
    #[must_use]
    pub fn stable_fingerprint(&self) -> u64 {
        crate::fingerprint::fnv1a64(self.canonical_record().as_bytes())
    }
}

/// A generated, bounded WGSL compute shader plus its deterministic identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedShader {
    /// Deterministic identity of this source.
    pub key: KernelSourceKey,
    /// Canonical WGSL text.
    pub source: String,
    /// Stable FNV-1a-64 fingerprint of the exact source bytes.
    pub source_fingerprint: u64,
}

/// Hard cap on generated source size.
///
/// The largest reachable template (double-buffered vec4) stays far below this
/// bound because every emitted quantity derives from closed enumerations; the
/// cap exists so an invariant regression fails explicitly instead of turning
/// into unbounded host allocation.
pub const MAX_GENERATED_SOURCE_BYTES: usize = 128 * 1024;

/// Entry-point name shared by every generated dense forward shader; matches
/// the handwritten contract.
pub const GENERATED_ENTRY_POINT: &str = "flat_attention_forward";

/// Errors raised while generating WGSL from a validated module.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelCodegenError {
    /// Emitted source exceeded the hard generation budget.
    SourceBudgetExceeded {
        /// Configured limit in bytes.
        limit_bytes: usize,
    },
}

impl fmt::Display for KernelCodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceBudgetExceeded { limit_bytes } => write!(
                f,
                "generated WGSL exceeded the {limit_bytes}-byte generation budget"
            ),
        }
    }
}

impl std::error::Error for KernelCodegenError {}

/// Generate canonical WGSL for a validated module.
///
/// # Errors
///
/// Returns [`KernelCodegenError::SourceBudgetExceeded`] if the emitted source
/// would exceed [`MAX_GENERATED_SOURCE_BYTES`].
pub fn emit(module: &KernelModule) -> Result<GeneratedShader, KernelCodegenError> {
    debug_assert_eq!(module.family(), KernelFamily::DenseQ4Forward);

    let mut out = String::with_capacity(16 * 1024);
    emit_header(&mut out, module);
    emit_constants(&mut out, module.config());
    emit_bindings(&mut out, module.config().vector_width == VectorWidth::Vec4);
    emit_workgroup_arrays(&mut out);
    out.push('\n');
    emit_entry(&mut out, module);

    if out.len() > MAX_GENERATED_SOURCE_BYTES {
        return Err(KernelCodegenError::SourceBudgetExceeded {
            limit_bytes: MAX_GENERATED_SOURCE_BYTES,
        });
    }

    let key = KernelSourceKey {
        ir_version: module.ir_version(),
        codegen_version: CodegenVersion::CURRENT,
        ir_fingerprint: module.structural_fingerprint(),
    };
    let source_fingerprint = crate::fingerprint::fnv1a64(out.as_bytes());
    Ok(GeneratedShader {
        key,
        source: out,
        source_fingerprint,
    })
}

fn emit_header(out: &mut String, module: &KernelModule) {
    let config = module.config();
    out.push_str("// Generated by the FLAT-ATTENTION Kernel IR WGSL emitter.\n");
    out.push_str(&format!(
        "// ir=v{};cg=v{};irfp={:016x}\n",
        module.ir_version(),
        CodegenVersion::CURRENT,
        module.structural_fingerprint()
    ));
    out.push_str("//\n");
    out.push_str(&format!(
        "// Dense Q4 fused forward; vector_width={:?}; staging={:?}; reduction={:?}.\n",
        config.vector_width, config.kv_staging, config.score_reduction
    ));
    out.push_str(
        "// Do not edit by hand: regenerate from the Kernel IR instead. This\n// source requires Naga/device qualification before routing use.\n",
    );
}

fn emit_constants(out: &mut String, config: &KernelConfig) {
    let kv_tile = match config.kv_staging {
        KvStaging::SingleBuffered => 8_u32,
        KvStaging::DoubleBuffered => 4_u32,
    };
    out.push_str("\nconst WORKGROUP_SIZE: u32 = 64u;\n");
    out.push_str("const QUERY_ROWS: u32 = 4u;\n");
    out.push_str(&format!("const KV_TILE: u32 = {kv_tile}u;\n"));
    out.push_str("const MAX_HEAD_DIM: u32 = 128u;\n");
    if config.kv_staging == KvStaging::DoubleBuffered {
        out.push_str("const KV_BANK_STRIDE: u32 = KV_TILE * MAX_HEAD_DIM;\n");
    }
    out.push_str("const NEG_MAX_F32: f32 = -3.402823466e38;\n");
}

fn emit_bindings(out: &mut String, vec4: bool) {
    out.push_str(
        "\nstruct Params {\n    seq_len: u32,\n    head_dim: u32,\n    batch_heads: u32,\n    causal: u32,\n    scale_bits: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n};\n",
    );
    let element = if vec4 { "vec4<f32>" } else { "f32" };
    out.push_str(&format!(
        "\n@group(0) @binding(0) var<storage, read> q: array<{element}>;\n"
    ));
    out.push_str(&format!(
        "@group(0) @binding(1) var<storage, read> k: array<{element}>;\n"
    ));
    out.push_str(&format!(
        "@group(0) @binding(2) var<storage, read> v: array<{element}>;\n"
    ));
    out.push_str("@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;\n");
    out.push_str("@group(0) @binding(4) var<uniform> params: Params;\n");
}

fn emit_workgroup_arrays(out: &mut String) {
    out.push_str("\nvar<workgroup> q_shared: array<f32, 512>;\n");
    out.push_str("var<workgroup> k_shared: array<f32, 1024>;\n");
    out.push_str("var<workgroup> v_shared: array<f32, 1024>;\n");
    out.push_str("var<workgroup> reduce_shared: array<f32, 256>;\n");
    out.push_str("var<workgroup> running_max_shared: array<f32, 4>;\n");
    out.push_str("var<workgroup> running_sum_shared: array<f32, 4>;\n");
    out.push_str("var<workgroup> alpha_shared: array<f32, 4>;\n");
    out.push_str("var<workgroup> p_shared: array<f32, 4>;\n");
}

fn emit_entry(out: &mut String, module: &KernelModule) {
    let config = module.config();
    let subgroup = config.score_reduction == ScoreReduction::SubgroupAssisted;
    let vec4 = config.vector_width == VectorWidth::Vec4;
    let double = config.kv_staging == KvStaging::DoubleBuffered;

    out.push_str("\n@compute @workgroup_size(64, 1, 1)\n");
    out.push_str(&format!("fn {GENERATED_ENTRY_POINT}(\n"));
    out.push_str("    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n");
    if subgroup {
        out.push_str("    @builtin(local_invocation_index) lane: u32,\n");
        out.push_str("    @builtin(num_subgroups) num_subgroups: u32,\n");
        out.push_str("    @builtin(subgroup_id) subgroup_id: u32,\n");
        out.push_str("    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,\n");
    } else {
        out.push_str("    @builtin(local_invocation_id) local_id: vec3<u32>,\n");
    }
    out.push_str(") {\n");

    emit_prologue(out, vec4, subgroup);
    emit_accumulators(out);

    if vec4 {
        emit_stage_query_vec4(out);
    } else {
        emit_stage_query_scalar(out);
    }
    emit_softmax_init(out);
    out.push_str("    workgroupBarrier();\n\n");

    out.push_str("    let query_limit = min(params.seq_len, query_start + QUERY_ROWS);\n");
    out.push_str(
        "    let kv_limit = select(params.seq_len, query_limit, params.causal != 0u);\n\n",
    );

    match (vec4, double) {
        (true, false) => emit_tile_loop_vec4_single(out),
        (true, true) => emit_tile_loop_vec4_double(out),
        (false, _) => emit_tile_loop_scalar(out, subgroup),
    }

    emit_epilogue(out, vec4);
    out.push_str("}\n");
}

fn emit_prologue(out: &mut String, vec4: bool, subgroup: bool) {
    out.push_str("    let query_start = workgroup_id.x * QUERY_ROWS;\n");
    out.push_str("    let bh = workgroup_id.y;\n");
    if !subgroup {
        // Subgroup variants receive `lane` directly as the flat invocation
        // index builtin; other variants project it from the 3D id.
        out.push_str("    let lane = local_id.x;\n");
    }
    if vec4 {
        out.push_str("    if (query_start >= params.seq_len || bh >= params.batch_heads ||\n");
        out.push_str("        (params.head_dim != 64u && params.head_dim != 128u)) {\n");
    } else {
        out.push_str("    if (query_start >= params.seq_len || bh >= params.batch_heads ||\n");
        out.push_str("        params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM) {\n");
    }
    out.push_str("        return;\n    }\n");
    out.push_str("    let scale = bitcast<f32>(params.scale_bits);\n");
    out.push_str("    let head_stride = params.seq_len * params.head_dim;\n");
    out.push_str("    let head_base = bh * head_stride;\n");
    out.push_str("    let output_elems = params.batch_heads * head_stride;\n");
    if vec4 {
        out.push_str("    let head_dim4 = params.head_dim / 4u;\n");
        out.push_str("    let head_stride4 = params.seq_len * head_dim4;\n");
        out.push_str("    let head_base4 = bh * head_stride4;\n");
    }
    out.push_str("    let d0 = lane;\n");
    out.push_str("    let d1 = lane + WORKGROUP_SIZE;\n");
}

fn emit_accumulators(out: &mut String) {
    for qr in 0..4_u32 {
        for half in 0..2_u32 {
            out.push_str(&format!("    var acc{qr}{half} = 0.0;\n"));
        }
    }
}

fn emit_stage_query_scalar(out: &mut String) {
    // Declares the single function-scope `qr` reused by the tile loops and
    // the epilogue, mirroring the qualified scalar kernel.
    out.push_str("\n    var qr = 0u;\n    loop {\n");
    out.push_str("        if (qr >= QUERY_ROWS) {\n            break;\n        }\n");
    out.push_str("        let query_pos = query_start + qr;\n");
    out.push_str("        let q_shared_base = qr * MAX_HEAD_DIM;\n");
    out.push_str("        if (query_pos < params.seq_len) {\n");
    out.push_str("            let query_base = head_base + query_pos * params.head_dim;\n");
    out.push_str("            if (d0 < params.head_dim) {\n");
    out.push_str("                q_shared[q_shared_base + d0] = q[query_base + d0];\n");
    out.push_str("            }\n");
    out.push_str("            if (d1 < params.head_dim) {\n");
    out.push_str("                q_shared[q_shared_base + d1] = q[query_base + d1];\n");
    out.push_str("            }\n");
    out.push_str("        }\n");
    out.push_str("        qr += 1u;\n    }\n");
}

fn emit_stage_query_vec4(out: &mut String) {
    out.push_str("\n    var q_linear4 = lane;\n");
    out.push_str("    let q_tile_vec4 = QUERY_ROWS * head_dim4;\n    loop {\n");
    out.push_str("        if (q_linear4 >= q_tile_vec4) {\n            break;\n        }\n");
    out.push_str("        let qr = q_linear4 / head_dim4;\n");
    out.push_str("        let vec_col = q_linear4 - qr * head_dim4;\n");
    out.push_str("        let query_pos = query_start + qr;\n");
    out.push_str("        if (query_pos < params.seq_len) {\n");
    out.push_str("            let packed_q = q[head_base4 + query_pos * head_dim4 + vec_col];\n");
    out.push_str("            let shared_base = qr * MAX_HEAD_DIM + vec_col * 4u;\n");
    emit_unpack(out, "packed_q", "q_shared", 12);
    out.push_str("        }\n");
    out.push_str("        q_linear4 += WORKGROUP_SIZE;\n    }\n");
}

fn emit_softmax_init(out: &mut String) {
    out.push_str("\n    if (lane == 0u) {\n");
    out.push_str("        var init_qr = 0u;\n        loop {\n");
    out.push_str(
        "            if (init_qr >= QUERY_ROWS) {\n                break;\n            }\n",
    );
    out.push_str("            running_max_shared[init_qr] = NEG_MAX_F32;\n");
    out.push_str("            running_sum_shared[init_qr] = 0.0;\n");
    out.push_str("            alpha_shared[init_qr] = 1.0;\n");
    out.push_str("            p_shared[init_qr] = 0.0;\n");
    out.push_str("            init_qr += 1u;\n        }\n    }\n");
}

/// Emit `dst[shared_base (+ i)] = packed.x/y/z/w` at the given indent.
fn emit_unpack(out: &mut String, packed: &str, dst: &str, indent: usize) {
    let pad = " ".repeat(indent);
    out.push_str(&format!("{pad}{dst}[shared_base] = {packed}.x;\n"));
    out.push_str(&format!("{pad}{dst}[shared_base + 1u] = {packed}.y;\n"));
    out.push_str(&format!("{pad}{dst}[shared_base + 2u] = {packed}.z;\n"));
    out.push_str(&format!("{pad}{dst}[shared_base + 3u] = {packed}.w;\n"));
}

fn emit_tile_loop_scalar(out: &mut String, subgroup: bool) {
    let vec4 = false;
    out.push_str("    var tile_start = 0u;\n    loop {\n");
    out.push_str("        if (tile_start >= kv_limit) {\n            break;\n        }\n");
    out.push_str("        let tile_rows = min(KV_TILE, kv_limit - tile_start);\n");
    out.push_str("        let tile_elements = tile_rows * params.head_dim;\n\n");
    out.push_str("        var linear = lane;\n        loop {\n");
    out.push_str(
        "            if (linear >= tile_elements) {\n                break;\n            }\n",
    );
    out.push_str("            let tile_row = linear / params.head_dim;\n");
    out.push_str("            let dim = linear - tile_row * params.head_dim;\n");
    out.push_str("            let global_index = head_base + (tile_start + tile_row) * params.head_dim + dim;\n");
    out.push_str("            let shared_index = tile_row * MAX_HEAD_DIM + dim;\n");
    out.push_str("            k_shared[shared_index] = k[global_index];\n");
    out.push_str("            v_shared[shared_index] = v[global_index];\n");
    out.push_str("            linear += WORKGROUP_SIZE;\n        }\n");
    out.push_str("        workgroupBarrier();\n\n");

    emit_tile_row_loop(out, subgroup, "tile_rows", false, vec4);
    out.push_str("\n        workgroupBarrier();\n");
    out.push_str("        tile_start += KV_TILE;\n    }\n");
}

fn emit_tile_loop_vec4_single(out: &mut String) {
    let vec4 = true;
    out.push_str("    var tile_start = 0u;\n    loop {\n");
    out.push_str("        if (tile_start >= kv_limit) {\n            break;\n        }\n");
    out.push_str("        let tile_rows = min(KV_TILE, kv_limit - tile_start);\n");
    out.push_str("        let tile_vec4 = tile_rows * head_dim4;\n\n");
    out.push_str("        var linear4 = lane;\n        loop {\n");
    out.push_str(
        "            if (linear4 >= tile_vec4) {\n                break;\n            }\n",
    );
    out.push_str("            let tile_row = linear4 / head_dim4;\n");
    out.push_str("            let vec_col = linear4 - tile_row * head_dim4;\n");
    out.push_str(
        "            let global4 = head_base4 + (tile_start + tile_row) * head_dim4 + vec_col;\n",
    );
    out.push_str("            let shared_base = tile_row * MAX_HEAD_DIM + vec_col * 4u;\n");
    out.push_str("            let packed_k = k[global4];\n");
    out.push_str("            let packed_v = v[global4];\n");
    emit_unpack(out, "packed_k", "k_shared", 12);
    emit_unpack(out, "packed_v", "v_shared", 12);
    out.push_str("            linear4 += WORKGROUP_SIZE;\n        }\n");
    out.push_str("        workgroupBarrier();\n\n");

    emit_tile_row_loop(out, false, "tile_rows", false, vec4);
    out.push_str("\n        workgroupBarrier();\n");
    out.push_str("        tile_start += KV_TILE;\n    }\n");
}

fn emit_tile_loop_vec4_double(out: &mut String) {
    let vec4 = true;
    // Seed bank 0 before entering the software-pipelined loop.
    out.push_str("    let first_rows = min(KV_TILE, kv_limit);\n");
    out.push_str("    let first_vec4 = first_rows * head_dim4;\n");
    out.push_str("    var first_linear4 = lane;\n    loop {\n");
    out.push_str("        if (first_linear4 >= first_vec4) {\n            break;\n        }\n");
    out.push_str("        let tile_row = first_linear4 / head_dim4;\n");
    out.push_str("        let vec_col = first_linear4 - tile_row * head_dim4;\n");
    out.push_str("        let global4 = head_base4 + tile_row * head_dim4 + vec_col;\n");
    out.push_str("        let shared_base = tile_row * MAX_HEAD_DIM + vec_col * 4u;\n");
    out.push_str("        let packed_k = k[global4];\n");
    out.push_str("        let packed_v = v[global4];\n");
    emit_unpack(out, "packed_k", "k_shared", 8);
    emit_unpack(out, "packed_v", "v_shared", 8);
    out.push_str("        first_linear4 += WORKGROUP_SIZE;\n    }\n");
    out.push_str("    workgroupBarrier();\n\n");

    out.push_str("    var current_bank = 0u;\n");
    out.push_str("    var tile_start = 0u;\n    loop {\n");
    out.push_str("        if (tile_start >= kv_limit) {\n            break;\n        }\n");
    out.push_str("        let current_rows = min(KV_TILE, kv_limit - tile_start);\n");
    out.push_str("        let next_start = tile_start + KV_TILE;\n");
    out.push_str("        let next_bank = 1u - current_bank;\n\n");
    // Prefetch the next tile into the inactive bank. Deliberately no barrier
    // follows: current-bank reads are disjoint, and the first reduction
    // barrier below completes these writes before the next bank is read.
    out.push_str("        if (next_start < kv_limit) {\n");
    out.push_str("            let next_rows = min(KV_TILE, kv_limit - next_start);\n");
    out.push_str("            let next_vec4 = next_rows * head_dim4;\n");
    out.push_str("            var next_linear4 = lane;\n            loop {\n");
    out.push_str("                if (next_linear4 >= next_vec4) {\n                    break;\n                }\n");
    out.push_str("                let tile_row = next_linear4 / head_dim4;\n");
    out.push_str("                let vec_col = next_linear4 - tile_row * head_dim4;\n");
    out.push_str("                let global4 = head_base4 + (next_start + tile_row) * head_dim4 + vec_col;\n");
    out.push_str("                let shared_base =\n                    next_bank * KV_BANK_STRIDE + tile_row * MAX_HEAD_DIM + vec_col * 4u;\n");
    out.push_str("                let packed_k = k[global4];\n");
    out.push_str("                let packed_v = v[global4];\n");
    emit_unpack(out, "packed_k", "k_shared", 16);
    emit_unpack(out, "packed_v", "v_shared", 16);
    out.push_str("                next_linear4 += WORKGROUP_SIZE;\n            }\n");
    out.push_str("        }\n\n");
    out.push_str("        let current_base = current_bank * KV_BANK_STRIDE;\n\n");

    emit_tile_row_loop(out, false, "current_rows", true, vec4);
    out.push_str("\n        current_bank = next_bank;\n");
    out.push_str("        tile_start += KV_TILE;\n    }\n");
}

/// Per-tile-row query loop: partial products, score reduction, online-softmax
/// update, V accumulation. Closes the tile-row loop itself.
fn emit_tile_row_loop(
    out: &mut String,
    subgroup: bool,
    rows_let: &str,
    double_buffered: bool,
    vec4: bool,
) {
    out.push_str("        var tile_row = 0u;\n        loop {\n");
    out.push_str(&format!(
        "            if (tile_row >= {rows_let}) {{\n                break;\n            }}\n"
    ));
    out.push_str("            let key_pos = tile_start + tile_row;\n");
    if double_buffered {
        out.push_str("            let shared_row = current_base + tile_row * MAX_HEAD_DIM;\n");
    } else {
        out.push_str("            let shared_row = tile_row * MAX_HEAD_DIM;\n");
    }
    if vec4 {
        // Vec4 realizations scope `qr` per tile-row iteration because the
        // query-staging stage never introduced one at function scope.
        out.push_str("\n            var qr = 0u;\n            loop {\n");
    } else {
        // Scalar realizations reuse the function-scope `qr`.
        out.push_str("\n            qr = 0u;\n            loop {\n");
    }
    out.push_str(
        "                if (qr >= QUERY_ROWS) {\n                    break;\n                }\n",
    );
    out.push_str("                let query_pos = query_start + qr;\n");
    out.push_str("                let valid_query = query_pos < params.seq_len;\n");
    out.push_str("                let participates = valid_query && (params.causal == 0u || key_pos <= query_pos);\n");
    out.push_str("                let q_shared_base = qr * MAX_HEAD_DIM;\n");
    out.push_str("                let reduce_base = qr * WORKGROUP_SIZE;\n\n");

    out.push_str("                var partial = 0.0;\n");
    for d in ["d0", "d1"] {
        out.push_str(&format!(
            "                if (participates && {d} < params.head_dim) {{\n"
        ));
        out.push_str(&format!(
            "                    partial += q_shared[q_shared_base + {d}] * k_shared[shared_row + {d}];\n"
        ));
        out.push_str("                }\n");
    }

    if subgroup {
        emit_subgroup_reduction(out);
    } else {
        emit_tree_reduction(out);
    }
    emit_softmax_update(out);
    emit_value_accumulation(out);

    out.push_str("                workgroupBarrier();\n");
    out.push_str("                qr += 1u;\n            }\n");
    out.push_str("            tile_row += 1u;\n        }\n");
}

fn emit_tree_reduction(out: &mut String) {
    out.push_str("                reduce_shared[reduce_base + lane] = partial;\n");
    out.push_str("                workgroupBarrier();\n\n");
    out.push_str("                var offset = 32u;\n                loop {\n");
    out.push_str("                    if (offset == 0u) {\n                        break;\n                    }\n");
    out.push_str("                    if (lane < offset) {\n");
    out.push_str("                        reduce_shared[reduce_base + lane] += reduce_shared[reduce_base + lane + offset];\n");
    out.push_str("                    }\n");
    out.push_str("                    workgroupBarrier();\n");
    out.push_str("                    offset = offset / 2u;\n                }\n\n");
}

fn emit_subgroup_reduction(out: &mut String) {
    out.push_str(
        "\n                // Native subgroup reduction. Every invocation participates,\n",
    );
    out.push_str(
        "                // so this collective stays uniform even for causal/tail rows.\n",
    );
    out.push_str("                let subgroup_sum = subgroupAdd(partial);\n");
    out.push_str("                if (subgroup_invocation_id == 0u) {\n");
    out.push_str("                    reduce_shared[reduce_base + subgroup_id] = subgroup_sum;\n");
    out.push_str("                }\n");
    out.push_str("                // Zero the unused slots padding the second-stage tree.\n");
    out.push_str("                if (lane >= num_subgroups) {\n");
    out.push_str("                    reduce_shared[reduce_base + lane] = 0.0;\n");
    out.push_str("                }\n");
    out.push_str("                workgroupBarrier();\n\n");
    out.push_str("                var reduction_width = 1u;\n                loop {\n");
    out.push_str("                    if (reduction_width >= num_subgroups) {\n                        break;\n                    }\n");
    out.push_str("                    reduction_width *= 2u;\n                }\n");
    out.push_str("                var offset = reduction_width / 2u;\n                loop {\n");
    out.push_str("                    if (offset == 0u) {\n                        break;\n                    }\n");
    out.push_str("                    if (lane < offset) {\n");
    out.push_str("                        reduce_shared[reduce_base + lane] += reduce_shared[reduce_base + lane + offset];\n");
    out.push_str("                    }\n");
    out.push_str("                    workgroupBarrier();\n");
    out.push_str("                    offset /= 2u;\n                }\n\n");
}

fn emit_softmax_update(out: &mut String) {
    out.push_str("                if (lane == 0u) {\n");
    out.push_str("                    if (participates) {\n");
    out.push_str("                        let score = reduce_shared[reduce_base] * scale;\n");
    out.push_str("                        let previous_max = running_max_shared[qr];\n");
    out.push_str("                        let new_max = max(previous_max, score);\n");
    out.push_str("                        let alpha = select(\n");
    out.push_str("                            exp(previous_max - new_max),\n");
    out.push_str("                            0.0,\n");
    out.push_str("                            running_sum_shared[qr] == 0.0,\n");
    out.push_str("                        );\n");
    out.push_str("                        let p = exp(score - new_max);\n");
    out.push_str("                        running_max_shared[qr] = new_max;\n");
    out.push_str(
        "                        running_sum_shared[qr] = running_sum_shared[qr] * alpha + p;\n",
    );
    out.push_str("                        alpha_shared[qr] = alpha;\n");
    out.push_str("                        p_shared[qr] = p;\n");
    out.push_str("                    } else {\n");
    out.push_str("                        alpha_shared[qr] = 1.0;\n");
    out.push_str("                        p_shared[qr] = 0.0;\n");
    out.push_str("                    }\n");
    out.push_str("                }\n");
    out.push_str("                workgroupBarrier();\n\n");
    out.push_str("                let alpha = alpha_shared[qr];\n");
    out.push_str("                let p = p_shared[qr];\n");
}

fn emit_value_accumulation(out: &mut String) {
    out.push_str("                if (qr == 0u) {\n");
    emit_acc_branch(out, 0);
    out.push_str("                } else if (qr == 1u) {\n");
    emit_acc_branch(out, 1);
    out.push_str("                } else if (qr == 2u) {\n");
    emit_acc_branch(out, 2);
    out.push_str("                } else {\n");
    emit_acc_branch(out, 3);
    out.push_str("                }\n");
}

fn emit_acc_branch(out: &mut String, qr: u32) {
    for half in 0..2_u32 {
        let d = if half == 0 { "d0" } else { "d1" };
        out.push_str(&format!(
            "                    if ({d} < params.head_dim) {{\n                        acc{qr}{half} = acc{qr}{half} * alpha + p * v_shared[shared_row + {d}];\n                    }}\n"
        ));
    }
}

fn emit_epilogue(out: &mut String, vec4: bool) {
    if vec4 {
        out.push_str("\n    var qr = 0u;\n    loop {\n");
    } else {
        out.push_str("\n    qr = 0u;\n    loop {\n");
    }
    out.push_str("        if (qr >= QUERY_ROWS) {\n            break;\n        }\n");
    out.push_str("        let query_pos = query_start + qr;\n");
    out.push_str("        if (query_pos < params.seq_len) {\n");
    out.push_str("            let query_base = head_base + query_pos * params.head_dim;\n");
    out.push_str("            let inv_sum = 1.0 / running_sum_shared[qr];\n");
    for qr in 0..4_u32 {
        match qr {
            0 => out.push_str("            if (qr == 0u) {\n"),
            3 => out.push_str("            } else {\n"),
            n => out.push_str(&format!("            }} else if (qr == {n}u) {{\n")),
        }
        for half in 0..2_u32 {
            let d = if half == 0 { "d0" } else { "d1" };
            out.push_str(&format!(
                "                if ({d} < params.head_dim) {{\n                    out_and_lse[query_base + {d}] = acc{qr}{half} * inv_sum;\n                }}\n"
            ));
        }
    }
    out.push_str("            }\n");
    out.push_str("            if (lane == 0u) {\n");
    out.push_str(
        "                let lse_index = output_elems + bh * params.seq_len + query_pos;\n",
    );
    out.push_str("                out_and_lse[lse_index] = running_max_shared[qr] + log(running_sum_shared[qr]);\n");
    out.push_str("            }\n");
    out.push_str("        }\n");
    out.push_str("        qr += 1u;\n    }\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::{AttentionProblem, KernelConfig, KernelFamily, KvStaging, QueryRows};
    use crate::{AttentionShape, FlatAttentionConfig};

    fn module(config: KernelConfig, causal: bool) -> KernelModule {
        let problem = AttentionProblem::from_shape(
            &AttentionShape {
                batch: 2,
                heads: 4,
                seq_len: 129,
                head_dim: 64,
            },
            FlatAttentionConfig {
                causal,
                softmax_scale: None,
            },
        )
        .unwrap();
        KernelModule::build(KernelFamily::DenseQ4Forward, problem, config).unwrap()
    }

    fn all_configs() -> Vec<KernelConfig> {
        vec![
            KernelConfig::PORTABLE_SCALAR,
            KernelConfig::PORTABLE_VEC4,
            KernelConfig::DOUBLE_BUFFERED_VEC4,
            KernelConfig::SUBGROUP_ASSISTED,
        ]
    }

    #[test]
    fn emission_is_byte_deterministic() {
        for config in all_configs() {
            let m = module(config, true);
            let a = emit(&m).unwrap();
            let b = emit(&m).unwrap();
            assert_eq!(a.source, b.source);
            assert_eq!(a.source_fingerprint, b.source_fingerprint);
            assert_eq!(a.key, b.key);
        }
    }

    #[test]
    fn distinct_configurations_produce_distinct_identities() {
        let mut fingerprints = Vec::new();
        for config in all_configs() {
            let g = emit(&module(config, true)).unwrap();
            fingerprints.push((g.key.stable_fingerprint(), g.source_fingerprint));
        }
        fingerprints.sort_unstable();
        let len = fingerprints.len();
        fingerprints.dedup();
        assert_eq!(fingerprints.len(), len, "identities must be unique");
    }

    #[test]
    fn causal_flip_changes_source_identity_but_not_key_irfpless_part() {
        // The IR fingerprint includes causal mode, so both key and source
        // identity must change.
        let a = emit(&module(KernelConfig::PORTABLE_SCALAR, true)).unwrap();
        let b = emit(&module(KernelConfig::PORTABLE_SCALAR, false)).unwrap();
        assert_ne!(a.key.ir_fingerprint, b.key.ir_fingerprint);
        assert_ne!(a.source_fingerprint, b.source_fingerprint);
    }

    #[test]
    fn generated_sources_stay_well_under_the_budget() {
        for config in all_configs() {
            let g = emit(&module(config, true)).unwrap();
            assert!(
                g.source.len() < 24 * 1024,
                "unexpected growth: {}",
                g.source.len()
            );
            assert!(g.source.len() < MAX_GENERATED_SOURCE_BYTES);
        }
    }

    #[test]
    fn sources_declare_contract_constants_and_bindings() {
        for config in all_configs() {
            let g = emit(&module(config, true)).unwrap();
            assert!(g.source.contains("@compute @workgroup_size(64, 1, 1)"));
            assert!(g.source.contains(GENERATED_ENTRY_POINT));
            for binding in [
                "@group(0) @binding(0)",
                "@group(0) @binding(1)",
                "@group(0) @binding(2)",
                "@group(0) @binding(3)",
                "@group(0) @binding(4)",
            ] {
                assert!(g.source.contains(binding), "missing {binding}");
            }
            assert!(g.source.contains("out_and_lse"));
            let expected_kv_tile = if config.kv_staging == KvStaging::DoubleBuffered {
                "const KV_TILE: u32 = 4u;"
            } else {
                "const KV_TILE: u32 = 8u;"
            };
            assert!(g.source.contains(expected_kv_tile));
        }
    }

    #[test]
    fn balanced_braces_in_every_generated_source() {
        for config in all_configs() {
            let g = emit(&module(config, true)).unwrap();
            assert_eq!(
                g.source.matches('{').count(),
                g.source.matches('}').count(),
                "unbalanced braces"
            );
        }
    }

    #[test]
    fn source_key_is_version_sensitive() {
        let m = module(KernelConfig::PORTABLE_SCALAR, true);
        let key_a = KernelSourceKey {
            ir_version: m.ir_version(),
            codegen_version: CodegenVersion::CURRENT,
            ir_fingerprint: m.structural_fingerprint(),
        };
        let key_b = KernelSourceKey {
            codegen_version: CodegenVersion { major: 9, minor: 0 },
            ..key_a.clone()
        };
        assert_ne!(key_a.stable_fingerprint(), key_b.stable_fingerprint());
    }

    #[test]
    fn query_rows_enum_remains_q4_only_for_this_family() {
        // Guards the emitter assumption that QUERY_ROWS is always Four.
        assert_eq!(QueryRows::Four.get(), 4);
        assert_eq!(
            module(KernelConfig::PORTABLE_SCALAR, true)
                .config()
                .query_rows,
            QueryRows::Four
        );
    }

    #[test]
    fn subgroup_template_used_exactly_when_configured() {
        let tree = emit(&module(KernelConfig::PORTABLE_SCALAR, true)).unwrap();
        assert!(!tree.source.contains("subgroupAdd"));
        assert!(tree.source.contains("@builtin(local_invocation_id)"));

        let sub = emit(&module(KernelConfig::SUBGROUP_ASSISTED, true)).unwrap();
        assert!(sub.source.contains("subgroupAdd"));
        assert!(sub.source.contains("@builtin(num_subgroups)"));
        assert!(!sub.source.contains("let lane = local_id.x;"));
    }

    #[test]
    fn double_buffered_template_contains_two_bank_references() {
        let g = emit(&module(KernelConfig::DOUBLE_BUFFERED_VEC4, true)).unwrap();
        assert!(g.source.contains("KV_BANK_STRIDE"));
        assert!(g.source.contains("current_bank"));
        assert!(g.source.contains("next_bank"));
    }
}
