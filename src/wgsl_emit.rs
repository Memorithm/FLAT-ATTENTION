//! Deterministic WGSL emission from FLAT Kernel IR (roadmap M21, first
//! subset).
//!
//! Scope of the first subset:
//!
//! - the Q4 fused-forward family with FP32 storage/accumulation;
//! - portable workgroup-tree reduction ([`ReductionStrategy::TreeInWorkgroup`]);
//! - the canonical bind-group/Params layout of the qualified handwritten
//!   kernel, so host packing code is shared unchanged.
//!
//! The subgroup reduction strategy and packed-f16 policy are representable in
//! IR but not yet emitted; [`emit_wgsl`] refuses them explicitly instead of
//! generating an unqualified shader. Existing handwritten shaders remain the
//! qualification reference for those families.
//!
//! Determinism contract: identical IR → byte-identical WGSL and identical
//! source fingerprint and cache key. Emission is a pure function; no map
//! iteration, no randomness, no time.

use crate::kernel_ir::{FlatKernelIr, IrOp, KernelIrError, PrecisionPolicy, ReductionStrategy};
use std::fmt;

/// Version of the WGSL backend code generator.
///
/// Bumped whenever emitted text could change for identical IR (formatting,
/// structural changes). Part of the kernel cache key so stale compiled
/// pipelines can never be reused across codegen generations.
pub const BACKEND_CODEGEN_VERSION: u32 = 1;

/// Cache identity of one generated kernel (roadmap M21 deliverable).
///
/// `KernelCacheKey = hash(flat_ir, capability-relevant specialization,
/// precision policy, backend_codegen_version)`. The capability-relevant
/// specialization is already baked into a normalized IR (tiles, workgroup
/// geometry, reduction strategy), so hashing the normalized IR covers it;
/// the fields are restated here to make the key self-describing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelCacheKey {
    ir_fingerprint: u64,
    precision_tag: &'static str,
    backend_codegen_version: u32,
    bits: u64,
}

impl KernelCacheKey {
    /// Derive the cache key from one validated IR.
    #[must_use]
    pub fn from_ir(ir: &FlatKernelIr) -> Self {
        let ir_fingerprint = ir.fingerprint();
        let precision_tag = match ir.plan().precision() {
            PrecisionPolicy::F32StorageF32Accumulate => "f32f32",
            PrecisionPolicy::PackedF16StorageF32Accumulate => "f16acc-f32",
        };
        let mut hash = Fnv::new();
        hash.update(b"flat-attention/kernel-cache-key/v1\0");
        hash.update(&ir_fingerprint.to_le_bytes());
        hash.update(precision_tag.as_bytes());
        hash.update(b"\0");
        hash.update(&BACKEND_CODEGEN_VERSION.to_le_bytes());
        Self {
            ir_fingerprint,
            precision_tag,
            backend_codegen_version: BACKEND_CODEGEN_VERSION,
            bits: hash.finish(),
        }
    }

    /// Fingerprint of the originating IR.
    #[must_use]
    pub const fn ir_fingerprint(&self) -> u64 {
        self.ir_fingerprint
    }

    /// Canonical precision tag.
    #[must_use]
    pub const fn precision_tag(&self) -> &'static str {
        self.precision_tag
    }

    /// Code-generator version.
    #[must_use]
    pub const fn backend_codegen_version(&self) -> u32 {
        self.backend_codegen_version
    }

    /// Raw cache-key bits (diagnostics; equality uses the whole struct).
    #[must_use]
    pub const fn bits(&self) -> u64 {
        self.bits
    }
}

impl fmt::Display for KernelCacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kck:{:016x}/ir:{:016x}/prec:{}:gen{}",
            self.bits, self.ir_fingerprint, self.precision_tag, self.backend_codegen_version
        )
    }
}

/// One deterministically generated kernel artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKernel {
    source: String,
    source_fingerprint: u64,
    cache_key: KernelCacheKey,
}

impl GeneratedKernel {
    /// Generated WGSL text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// FNV-1a-64 fingerprint over the generated text.
    #[must_use]
    pub const fn source_fingerprint(&self) -> u64 {
        self.source_fingerprint
    }

    /// Cache identity derived from the IR and codegen version.
    #[must_use]
    pub const fn cache_key(&self) -> &KernelCacheKey {
        &self.cache_key
    }
}

/// Errors produced during WGSL emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmitError {
    /// The IR itself failed validation.
    InvalidIr(KernelIrError),
    /// The plan requests a strategy outside the current emitter subset.
    UnsupportedSubset {
        /// What is missing.
        detail: &'static str,
    },
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIr(error) => write!(f, "cannot emit from invalid IR: {error}"),
            Self::UnsupportedSubset { detail } => {
                write!(f, "kernel is outside the current emitter subset: {detail}")
            }
        }
    }
}

impl std::error::Error for EmitError {}

/// Emit deterministic WGSL for one validated Q4 kernel IR.
///
/// # Errors
///
/// Returns [`EmitError`] when the IR is invalid or requests a reduction/
/// precision combination outside the first emitter subset.
pub fn emit_wgsl(ir: &FlatKernelIr) -> Result<GeneratedKernel, EmitError> {
    if ir.plan().reduction() != ReductionStrategy::TreeInWorkgroup {
        return Err(EmitError::UnsupportedSubset {
            detail: "subgroup-assisted reduction emission lands with its own qualification pass",
        });
    }
    if ir.plan().precision() != PrecisionPolicy::F32StorageF32Accumulate {
        return Err(EmitError::UnsupportedSubset {
            detail: "packed-f16 storage emission is representable but not yet generated",
        });
    }
    // Structural sanity: only the canonical program shape is supported by
    // this subset.
    let ops = ir.plan().program().ops();
    let expected_shape_matches = ops.first() == Some(&IrOp::StageQueryTile)
        && ops.contains(&IrOp::BeginKvTileLoop)
        && ops.last() == Some(&IrOp::FinalNormalizeAndStore);
    if !expected_shape_matches {
        return Err(EmitError::InvalidIr(
            KernelIrError::MalformedProgramStructure,
        ));
    }

    let tiles = ir.plan().tiles();
    let workgroup = ir.plan().workgroup();

    let mut out = String::with_capacity(12 * 1024);
    push_header(
        &mut out,
        ir.identity().variant,
        tiles.query_rows,
        tiles.kv_tile,
        workgroup.invocations,
    );
    push_layout(
        &mut out,
        workgroup.invocations,
        tiles.query_rows,
        tiles.kv_tile,
    );

    // Query-row load prologue.
    out.push_str("    var qr = 0u;\n    loop {\n        if (qr >= QUERY_ROWS) {\n            break;\n        }\n");
    out.push_str(
        "        let query_pos = query_start + qr;\n        let q_shared_base = qr * MAX_HEAD_DIM;\n",
    );
    out.push_str("        if (query_pos < params.seq_len) {\n");
    out.push_str(
        "            let query_base = head_base + query_pos * params.head_dim;\n            if (d0 < params.head_dim) {\n                q_shared[q_shared_base + d0] = q[query_base + d0];\n            }\n",
    );
    out.push_str("            if (d1 < params.head_dim) {\n                q_shared[q_shared_base + d1] = q[query_base + d1];\n            }\n        }\n");
    out.push_str("        if (lane == 0u) {\n            running_max_shared[qr] = NEG_MAX_F32;\n            running_sum_shared[qr] = 0.0;\n            alpha_shared[qr] = 1.0;\n            p_shared[qr] = 0.0;\n        }\n");
    out.push_str("        qr += 1u;\n    }\n    workgroupBarrier();\n\n");

    out.push_str(
        "    let query_limit = min(params.seq_len, query_start + QUERY_ROWS);\n    let kv_limit = select(params.seq_len, query_limit, params.causal != 0u);\n\n",
    );

    // K/V tile loop.
    out.push_str("    var tile_start = 0u;\n    loop {\n        if (tile_start >= kv_limit) {\n            break;\n        }\n");
    out.push_str("        let tile_rows = min(KV_TILE, kv_limit - tile_start);\n        let tile_elements = tile_rows * params.head_dim;\n\n");
    out.push_str("        var linear = lane;\n        loop {\n            if (linear >= tile_elements) {\n                break;\n            }\n");
    out.push_str("            let tile_row = linear / params.head_dim;\n            let dim = linear - tile_row * params.head_dim;\n");
    out.push_str("            let global_index = head_base + (tile_start + tile_row) * params.head_dim + dim;\n");
    out.push_str("            let shared_index = tile_row * MAX_HEAD_DIM + dim;\n            k_shared[shared_index] = k[global_index];\n            v_shared[shared_index] = v[global_index];\n");
    out.push_str(
        "            linear += WORKGROUP_SIZE;\n        }\n        workgroupBarrier();\n\n",
    );

    out.push_str("        var tile_row = 0u;\n        loop {\n            if (tile_row >= tile_rows) {\n                break;\n            }\n");
    out.push_str("            let key_pos = tile_start + tile_row;\n            let shared_row = tile_row * MAX_HEAD_DIM;\n\n            qr = 0u;\n            loop {\n                if (qr >= QUERY_ROWS) {\n                    break;\n                }\n");
    out.push_str("                let query_pos = query_start + qr;\n                let valid_query = query_pos < params.seq_len;\n                let participates = valid_query && (params.causal == 0u || key_pos <= query_pos);\n");
    out.push_str("                let q_shared_base = qr * MAX_HEAD_DIM;\n                let reduce_base = qr * WORKGROUP_SIZE;\n\n");
    out.push_str("                var partial = 0.0;\n                if (participates && d0 < params.head_dim) {\n                    partial += q_shared[q_shared_base + d0] * k_shared[shared_row + d0];\n                }\n");
    out.push_str("                if (participates && d1 < params.head_dim) {\n                    partial += q_shared[q_shared_base + d1] * k_shared[shared_row + d1];\n                }\n                reduce_shared[reduce_base + lane] = partial;\n                workgroupBarrier();\n\n");

    emit_tree_reduction(&mut out, workgroup.invocations);

    out.push_str("                if (lane == 0u) {\n                    if (participates) {\n                        let score = reduce_shared[reduce_base] * scale;\n                        let previous_max = running_max_shared[qr];\n                        let new_max = max(previous_max, score);\n");
    out.push_str("                        let alpha = select(\n                            exp(previous_max - new_max),\n                            0.0,\n                            running_sum_shared[qr] == 0.0,\n                        );\n");
    out.push_str("                        let p = exp(score - new_max);\n                        running_max_shared[qr] = new_max;\n                        running_sum_shared[qr] = running_sum_shared[qr] * alpha + p;\n");
    out.push_str("                        alpha_shared[qr] = alpha;\n                        p_shared[qr] = p;\n                    } else {\n                        alpha_shared[qr] = 1.0;\n                        p_shared[qr] = 0.0;\n                    }\n                }\n");
    out.push_str("                workgroupBarrier();\n\n                let alpha = alpha_shared[qr];\n                let p = p_shared[qr];\n");

    emit_accumulator_update(&mut out, tiles.query_rows);

    out.push_str("                workgroupBarrier();\n                qr += 1u;\n            }\n            tile_row += 1u;\n        }\n\n        workgroupBarrier();\n        tile_start += KV_TILE;\n    }\n\n");

    // Epilogue: normalize and store O + LSE.
    out.push_str("    qr = 0u;\n    loop {\n        if (qr >= QUERY_ROWS) {\n            break;\n        }\n        let query_pos = query_start + qr;\n        if (query_pos < params.seq_len) {\n");
    out.push_str("            let query_base = head_base + query_pos * params.head_dim;\n            let inv_sum = 1.0 / running_sum_shared[qr];\n");
    emit_output_store(&mut out, tiles.query_rows);
    out.push_str("            if (lane == 0u) {\n                let lse_index = output_elems + bh * params.seq_len + query_pos;\n                out_and_lse[lse_index] = running_max_shared[qr] + log(running_sum_shared[qr]);\n            }\n");
    out.push_str("        }\n        qr += 1u;\n    }\n}\n");

    let source_fingerprint = {
        let mut hash = Fnv::new();
        hash.update(out.as_bytes());
        hash.finish()
    };
    Ok(GeneratedKernel {
        cache_key: KernelCacheKey::from_ir(ir),
        source_fingerprint,
        source: out,
    })
}

fn push_header(out: &mut String, variant: &str, query_rows: u32, kv_tile: u32, workgroup: u32) {
    out.push_str("// FLAT-ATTENTION generated fused forward kernel — Q4 tiled generation.\n");
    out.push_str("//\n");
    out.push_str(&format!(
        "// Variant: `{variant}`. Generated deterministically from FLAT Kernel IR\n"
    ));
    out.push_str(&format!(
        "// (query_rows={query_rows}, kv_tile={kv_tile}, workgroup={workgroup}). Do not edit:\n"
    ));
    out.push_str("// regenerate from the IR instead.\n//\n");
    out.push_str("// Dispatch contract:\n//   x = ceil(seq_len / QUERY_ROWS)\n//   y = batch_heads\n//   z = 1\n\n");
}

#[allow(clippy::too_many_lines)]
fn push_layout(out: &mut String, workgroup: u32, query_rows: u32, kv_tile: u32) {
    out.push_str(&format!("const WORKGROUP_SIZE: u32 = {workgroup}u;\n"));
    out.push_str(&format!("const QUERY_ROWS: u32 = {query_rows}u;\n"));
    out.push_str(&format!("const KV_TILE: u32 = {kv_tile}u;\n"));
    out.push_str("const MAX_HEAD_DIM: u32 = 128u;\n");
    out.push_str("const NEG_MAX_F32: f32 = -3.402823466e38;\n\n");
    out.push_str("struct Params {\n    seq_len: u32,\n    head_dim: u32,\n    batch_heads: u32,\n    causal: u32,\n    scale_bits: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n};\n\n");
    out.push_str("@group(0) @binding(0) var<storage, read> q: array<f32>;\n");
    out.push_str("@group(0) @binding(1) var<storage, read> k: array<f32>;\n");
    out.push_str("@group(0) @binding(2) var<storage, read> v: array<f32>;\n");
    out.push_str("@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;\n");
    out.push_str("@group(0) @binding(4) var<uniform> params: Params;\n\n");
    let q_elems = query_rows * 128;
    let kv_elems = kv_tile * 128;
    out.push_str(&format!(
        "var<workgroup> q_shared: array<f32, {q_elems}>;\n"
    ));
    out.push_str(&format!(
        "var<workgroup> k_shared: array<f32, {kv_elems}>;\nvar<workgroup> v_shared: array<f32, {kv_elems}>;\n"
    ));
    let reduce_elems = query_rows * workgroup;
    out.push_str(&format!(
        "var<workgroup> reduce_shared: array<f32, {reduce_elems}>;\n"
    ));
    out.push_str(&format!(
        "var<workgroup> running_max_shared: array<f32, {query_rows}>;\nvar<workgroup> running_sum_shared: array<f32, {query_rows}>;\nvar<workgroup> alpha_shared: array<f32, {query_rows}>;\nvar<workgroup> p_shared: array<f32, {query_rows}>;\n\n"
    ));
    out.push_str(&format!("@compute @workgroup_size({workgroup}, 1, 1)\n"));
    out.push_str("fn flat_attention_forward(\n    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n    @builtin(local_invocation_id) local_id: vec3<u32>,\n) {\n");
    out.push_str("    let query_start = workgroup_id.x * QUERY_ROWS;\n    let bh = workgroup_id.y;\n    let lane = local_id.x;\n\n");
    out.push_str("    if (query_start >= params.seq_len || bh >= params.batch_heads || params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM) {\n        return;\n    }\n\n");
    out.push_str("    let scale = bitcast<f32>(params.scale_bits);\n    let head_stride = params.seq_len * params.head_dim;\n    let head_base = bh * head_stride;\n    let output_elems = params.batch_heads * head_stride;\n    let d0 = lane;\n    let d1 = lane + WORKGROUP_SIZE;\n\n");
    emit_accumulator_declarations(out, query_rows);
}

fn emit_accumulator_declarations(out: &mut String, query_rows: u32) {
    out.push('\n');
    for qr in 0..query_rows {
        for d in 0..2u32 {
            let index = qr * 10 + d;
            out.push_str(&format!("    var acc{index:02} = 0.0;\n"));
        }
    }
    out.push('\n');
}

fn emit_tree_reduction(out: &mut String, workgroup: u32) {
    out.push_str(&format!(
        "                var offset = {}u;\n                loop {{\n                    if (offset == 0u) {{\n                        break;\n                    }}\n",
        workgroup / 2
    ));
    out.push_str("                    if (lane < offset) {\n                        reduce_shared[reduce_base + lane] += reduce_shared[reduce_base + lane + offset];\n                    }\n");
    out.push_str("                    workgroupBarrier();\n                    offset = offset / 2u;\n                }\n\n");
}

/// Emit the per-query-row accumulator rescale as named scalars, preserving
/// the qualified kernel's arithmetic structure for any QUERY_ROWS value.
fn emit_accumulator_update(out: &mut String, query_rows: u32) {
    for qr in 0..query_rows {
        let keyword = if qr == 0 { "if" } else { "} else if" };
        let a = qr * 10;
        let b = qr * 10 + 1;
        out.push_str(&format!(
            "                {keyword} (qr == {qr}u) {{\n                    if (d0 < params.head_dim) {{\n                        acc{a:02} = acc{a:02} * alpha + p * v_shared[shared_row + d0];\n                    }}\n"
        ));
        out.push_str(&format!(
            "                    if (d1 < params.head_dim) {{\n                        acc{b:02} = acc{b:02} * alpha + p * v_shared[shared_row + d1];\n                    }}\n"
        ));
    }
    out.push_str("                }\n");
}

fn emit_output_store(out: &mut String, query_rows: u32) {
    for qr in 0..query_rows {
        let keyword = if qr == 0 { "if" } else { "} else if" };
        let a = qr * 10;
        let b = qr * 10 + 1;
        out.push_str(&format!(
            "            {keyword} (qr == {qr}u) {{\n                if (d0 < params.head_dim) {{\n                    out_and_lse[query_base + d0] = acc{a:02} * inv_sum;\n                }}\n"
        ));
        out.push_str(&format!(
            "                if (d1 < params.head_dim) {{\n                    out_and_lse[query_base + d1] = acc{b:02} * inv_sum;\n                }}\n"
        ));
    }
    out.push_str("            }\n");
}

/// Local framed-free FNV-1a-64 matching the repository fingerprint
/// convention.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
