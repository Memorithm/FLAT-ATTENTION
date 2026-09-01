// FDAL3: research-only q_len=1 DA-LUC WGPU candidate.
//
// Supported subset is deliberately narrow and host-gated:
// - contiguous BatchHeadToken physical rows;
// - F32 K codebooks, shared or per-KV-head;
// - 8-bit LSB0 K indices, no K residual;
// - groupwise-affine U8 V with F32 scales/U8 zero-points, no V residual;
// - key/value head dimensions <= 128;
// - no dense K/V materialization.
//
// This shader converts packed scalar values in registers. It therefore makes
// no "zero dequantization" claim and carries no performance claim.

const WORKGROUP_SIZE: u32 = 64u;
const MAX_HEAD_DIM: u32 = 128u;
const MAX_LUT_ENTRIES: u32 = 2048u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    kv_len: u32,
    kv_capacity: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    batch: u32,
    subspace_dim: u32,
    codebook_entries: u32,
    codebook_scope_per_head: u32,
    value_group_size: u32,
    scale_bits: u32,
    causal: u32,
    query_position: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> codebook: array<f32>;
@group(0) @binding(2) var<storage, read> key_indices: array<u32>;
@group(0) @binding(3) var<storage, read> values: array<u32>;
@group(0) @binding(4) var<storage, read> value_scales: array<f32>;
@group(0) @binding(5) var<storage, read> value_zero_points: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 128>;
var<workgroup> lut_shared: array<f32, 2048>;
var<workgroup> reduce_shared: array<f32, 64>;
var<workgroup> running_max_shared: f32;
var<workgroup> running_sum_shared: f32;
var<workgroup> alpha_shared: f32;
var<workgroup> p_shared: f32;

fn read_key_index(byte_index: u32) -> u32 {
    let word = key_indices[byte_index / 4u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 255u;
}

fn read_value(byte_index: u32) -> u32 {
    let word = values[byte_index / 4u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 255u;
}

fn read_zero_point(byte_index: u32) -> u32 {
    let word = value_zero_points[byte_index / 4u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 255u;
}

@compute @workgroup_size(64, 1, 1)
fn flat_da_luc_decode(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q_batch_head = workgroup_id.x;
    let lane = local_id.x;
    let q_batch_heads = params.batch * params.q_heads;
    let subspaces = params.key_head_dim / params.subspace_dim;
    let lut_entries = subspaces * params.codebook_entries;

    if (q_batch_head >= q_batch_heads
        || params.kv_len == 0u
        || params.kv_len > params.kv_capacity
        || params.key_head_dim == 0u
        || params.key_head_dim > MAX_HEAD_DIM
        || params.value_head_dim == 0u
        || params.value_head_dim > MAX_HEAD_DIM
        || params.subspace_dim == 0u
        || params.key_head_dim % params.subspace_dim != 0u
        || params.codebook_entries == 0u
        || lut_entries > MAX_LUT_ENTRIES
        || params.q_heads == 0u
        || params.kv_heads == 0u
        || params.value_group_size == 0u
        || params.value_head_dim % params.value_group_size != 0u) {
        return;
    }

    let head_group_size = params.q_heads / params.kv_heads;
    if (head_group_size == 0u) {
        return;
    }

    let batch_index = q_batch_head / params.q_heads;
    let q_head = q_batch_head - batch_index * params.q_heads;
    let kv_head = q_head / head_group_size;
    if (kv_head >= params.kv_heads) {
        return;
    }

    let q_base = q_batch_head * params.key_head_dim;
    let out_base = q_batch_head * params.value_head_dim;
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    if (d0 < params.key_head_dim) {
        q_shared[d0] = q[q_base + d0];
    }
    if (d1 < params.key_head_dim) {
        q_shared[d1] = q[q_base + d1];
    }
    if (lane == 0u) {
        running_max_shared = NEG_MAX_F32;
        running_sum_shared = 0.0;
        alpha_shared = 1.0;
        p_shared = 0.0;
    }
    workgroupBarrier();

    // Query-local codebook LUT. This materializes only scores for the small
    // codebook, never a dense K cache.
    var linear = lane;
    loop {
        if (linear >= lut_entries) {
            break;
        }
        let subspace = linear / params.codebook_entries;
        let entry = linear - subspace * params.codebook_entries;
        let scope_head = select(0u, kv_head, params.codebook_scope_per_head != 0u);
        let codebook_base = (((scope_head * subspaces + subspace)
            * params.codebook_entries + entry) * params.subspace_dim);
        let query_base = subspace * params.subspace_dim;
        var dot = 0.0;
        var inner = 0u;
        loop {
            if (inner >= params.subspace_dim) {
                break;
            }
            dot += q_shared[query_base + inner] * codebook[codebook_base + inner];
            inner += 1u;
        }
        lut_shared[linear] = dot;
        linear += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    let groups = params.value_head_dim / params.value_group_size;
    let scale = bitcast<f32>(params.scale_bits);
    var acc0 = 0.0;
    var acc1 = 0.0;
    var key_pos = 0u;
    loop {
        if (key_pos >= params.kv_len) {
            break;
        }
        if (params.causal != 0u && key_pos > params.query_position) {
            break;
        }

        // BatchHeadToken contiguous row order.
        let row = (batch_index * params.kv_heads + kv_head) * params.kv_capacity + key_pos;
        var partial = 0.0;
        var subspace = lane;
        loop {
            if (subspace >= subspaces) {
                break;
            }
            let packed_index = row * subspaces + subspace;
            let entry = read_key_index(packed_index);
            partial += lut_shared[subspace * params.codebook_entries + entry];
            subspace += WORKGROUP_SIZE;
        }
        reduce_shared[lane] = partial;
        workgroupBarrier();

        var reduction_offset = 32u;
        loop {
            if (reduction_offset == 0u) {
                break;
            }
            if (lane < reduction_offset) {
                reduce_shared[lane] += reduce_shared[lane + reduction_offset];
            }
            workgroupBarrier();
            reduction_offset /= 2u;
        }

        if (lane == 0u) {
            let score = reduce_shared[0] * scale;
            let old_max = running_max_shared;
            let new_max = max(old_max, score);
            let alpha = select(exp(old_max - new_max), 0.0, old_max == NEG_MAX_F32);
            let p = exp(score - new_max);
            running_max_shared = new_max;
            running_sum_shared = running_sum_shared * alpha + p;
            alpha_shared = alpha;
            p_shared = p;
        }
        workgroupBarrier();

        if (d0 < params.value_head_dim) {
            let group = d0 / params.value_group_size;
            let group_index = row * groups + group;
            let raw = read_value(row * params.value_head_dim + d0);
            let zp = read_zero_point(group_index);
            let value = value_scales[group_index] * (f32(raw) - f32(zp));
            acc0 = acc0 * alpha_shared + p_shared * value;
        }
        if (d1 < params.value_head_dim) {
            let group = d1 / params.value_group_size;
            let group_index = row * groups + group;
            let raw = read_value(row * params.value_head_dim + d1);
            let zp = read_zero_point(group_index);
            let value = value_scales[group_index] * (f32(raw) - f32(zp));
            acc1 = acc1 * alpha_shared + p_shared * value;
        }
        workgroupBarrier();
        key_pos += 1u;
    }

    let inv_sum = 1.0 / running_sum_shared;
    if (d0 < params.value_head_dim) {
        out_and_lse[out_base + d0] = acc0 * inv_sum;
    }
    if (d1 < params.value_head_dim) {
        out_and_lse[out_base + d1] = acc1 * inv_sum;
    }
    if (lane == 0u) {
        let output_elements = params.batch * params.q_heads * params.value_head_dim;
        out_and_lse[output_elements + q_batch_head] = running_max_shared + log(running_sum_shared);
    }
}
