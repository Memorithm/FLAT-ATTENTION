// M48: opt-in q_len=1 decode over projection-layout, pre-rotated K.
//
// One workgroup owns up to four query heads mapped to one physical GQA KV
// head. Each K/V tile is staged once and reused across those query heads.
// Buffers remain caller-owned and use the M11 projection layout:
//   Q/O [batch, 1, q_heads * head_dim]
//   K/V [batch, kv_len, kv_heads * head_dim]

const WORKGROUP_SIZE: u32 = 64u;
const Q_HEADS_PER_WORKGROUP: u32 = 4u;
const KV_TILE: u32 = 8u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    q_len: u32,
    kv_len: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    batch: u32,
    causal: u32,
    scale_bits: u32,
    theta_bits: u32,
    causal_query_offset: u32,
    q_rope_offset: u32,
    kv_rope_offset: u32,
    rotate_k: u32,
    bias_mode: u32,
    bias_q_offset: u32,
    bias_kv_offset: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 512>;
var<workgroup> k_shared: array<f32, 1024>;
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 256>;
var<workgroup> running_max_shared: array<f32, 4>;
var<workgroup> running_sum_shared: array<f32, 4>;
var<workgroup> alpha_shared: array<f32, 4>;
var<workgroup> probability_shared: array<f32, 4>;

fn rope_pair(e: f32, o: f32, pair: u32, position: u32, head_dim: u32, theta: f32) -> vec2<f32> {
    let frequency = pow(theta, -2.0 * f32(pair) / f32(head_dim));
    let angle = f32(position) * frequency;
    let cosine = cos(angle);
    let sine = sin(angle);
    return vec2<f32>(e * cosine - o * sine, e * sine + o * cosine);
}

@compute @workgroup_size(64, 1, 1)
fn flat_attention_decode_kv_reuse(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lane = local_id.x;
    if (params.q_len != 1u || params.kv_len == 0u ||
        params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM ||
        (params.head_dim & 1u) != 0u || params.q_heads == 0u ||
        params.kv_heads == 0u || params.rotate_k != 0u || params.bias_mode != 0u) {
        return;
    }

    let group_size = params.q_heads / params.kv_heads;
    if (group_size < 2u || group_size * params.kv_heads != params.q_heads) {
        return;
    }
    let tiles_per_kv_head =
        (group_size + Q_HEADS_PER_WORKGROUP - 1u) / Q_HEADS_PER_WORKGROUP;
    let batch_kv_tile = workgroup_id.y;
    let batch_kv = batch_kv_tile / tiles_per_kv_head;
    let head_tile = batch_kv_tile - batch_kv * tiles_per_kv_head;
    let batch_index = batch_kv / params.kv_heads;
    let kv_head = batch_kv - batch_index * params.kv_heads;
    if (batch_index >= params.batch || kv_head >= params.kv_heads) {
        return;
    }

    let q_group_start = kv_head * group_size;
    let q_head_start = q_group_start + head_tile * Q_HEADS_PER_WORKGROUP;
    let q_head_count = min(
        Q_HEADS_PER_WORKGROUP,
        q_group_start + group_size - q_head_start,
    );
    let q_width = params.q_heads * params.head_dim;
    let kv_width = params.kv_heads * params.head_dim;
    let theta = bitcast<f32>(params.theta_bits);
    let scale = bitcast<f32>(params.scale_bits);
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    var accumulators: array<f32, 8>;
    var local_head = 0u;
    loop {
        if (local_head >= q_head_count) {
            break;
        }
        let q_head = q_head_start + local_head;
        let q_base = batch_index * q_width + q_head * params.head_dim;
        let shared_base = local_head * MAX_HEAD_DIM;
        if (d0 < params.head_dim) {
            q_shared[shared_base + d0] = q[q_base + d0];
        }
        if (d1 < params.head_dim) {
            q_shared[shared_base + d1] = q[q_base + d1];
        }
        if (lane == 0u) {
            running_max_shared[local_head] = NEG_MAX_F32;
            running_sum_shared[local_head] = 0.0;
            alpha_shared[local_head] = 1.0;
            probability_shared[local_head] = 0.0;
        }
        local_head += 1u;
    }
    workgroupBarrier();

    let half_dim = params.head_dim / 2u;
    local_head = 0u;
    loop {
        if (local_head >= q_head_count) {
            break;
        }
        var pair = lane;
        loop {
            if (pair >= half_dim) {
                break;
            }
            let base = local_head * MAX_HEAD_DIM + 2u * pair;
            let rotated = rope_pair(
                q_shared[base],
                q_shared[base + 1u],
                pair,
                params.q_rope_offset,
                params.head_dim,
                theta,
            );
            q_shared[base] = rotated.x;
            q_shared[base + 1u] = rotated.y;
            pair += WORKGROUP_SIZE;
        }
        local_head += 1u;
    }
    workgroupBarrier();

    var tile_start = 0u;
    loop {
        if (tile_start >= params.kv_len) {
            break;
        }
        let tile_rows = min(KV_TILE, params.kv_len - tile_start);
        let tile_elements = tile_rows * params.head_dim;
        var linear = lane;
        loop {
            if (linear >= tile_elements) {
                break;
            }
            let tile_row = linear / params.head_dim;
            let dim = linear - tile_row * params.head_dim;
            let key_position = tile_start + tile_row;
            let kv_row = batch_index * params.kv_len + key_position;
            let global_index = kv_row * kv_width + kv_head * params.head_dim + dim;
            let shared_index = tile_row * MAX_HEAD_DIM + dim;
            k_shared[shared_index] = k[global_index];
            v_shared[shared_index] = v[global_index];
            linear += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var tile_row = 0u;
        loop {
            if (tile_row >= tile_rows) {
                break;
            }
            let shared_row = tile_row * MAX_HEAD_DIM;
            local_head = 0u;
            loop {
                if (local_head >= q_head_count) {
                    break;
                }
                let q_shared_base = local_head * MAX_HEAD_DIM;
                let reduce_base = local_head * WORKGROUP_SIZE;
                var partial = 0.0;
                if (d0 < params.head_dim) {
                    partial += q_shared[q_shared_base + d0] * k_shared[shared_row + d0];
                }
                if (d1 < params.head_dim) {
                    partial += q_shared[q_shared_base + d1] * k_shared[shared_row + d1];
                }
                reduce_shared[reduce_base + lane] = partial;
                workgroupBarrier();

                var reduction_offset = 32u;
                loop {
                    if (reduction_offset == 0u) {
                        break;
                    }
                    if (lane < reduction_offset) {
                        reduce_shared[reduce_base + lane] +=
                            reduce_shared[reduce_base + lane + reduction_offset];
                    }
                    workgroupBarrier();
                    reduction_offset /= 2u;
                }

                if (lane == 0u) {
                    let score = reduce_shared[reduce_base] * scale;
                    let old_max = running_max_shared[local_head];
                    let new_max = max(old_max, score);
                    let alpha = select(exp(old_max - new_max), 0.0, old_max == NEG_MAX_F32);
                    let probability = exp(score - new_max);
                    running_max_shared[local_head] = new_max;
                    running_sum_shared[local_head] =
                        running_sum_shared[local_head] * alpha + probability;
                    alpha_shared[local_head] = alpha;
                    probability_shared[local_head] = probability;
                }
                workgroupBarrier();

                let accumulator_base = local_head * 2u;
                if (d0 < params.head_dim) {
                    accumulators[accumulator_base] =
                        accumulators[accumulator_base] * alpha_shared[local_head] +
                        probability_shared[local_head] * v_shared[shared_row + d0];
                }
                if (d1 < params.head_dim) {
                    accumulators[accumulator_base + 1u] =
                        accumulators[accumulator_base + 1u] * alpha_shared[local_head] +
                        probability_shared[local_head] * v_shared[shared_row + d1];
                }
                workgroupBarrier();
                local_head += 1u;
            }
            tile_row += 1u;
        }
        workgroupBarrier();
        tile_start += tile_rows;
    }

    let output_elements = params.batch * q_width;
    local_head = 0u;
    loop {
        if (local_head >= q_head_count) {
            break;
        }
        let q_head = q_head_start + local_head;
        let q_batch_head = batch_index * params.q_heads + q_head;
        let output_base = batch_index * q_width + q_head * params.head_dim;
        let accumulator_base = local_head * 2u;
        let inverse_sum = 1.0 / running_sum_shared[local_head];
        if (d0 < params.head_dim) {
            out_and_lse[output_base + d0] =
                accumulators[accumulator_base] * inverse_sum;
        }
        if (d1 < params.head_dim) {
            out_and_lse[output_base + d1] =
                accumulators[accumulator_base + 1u] * inverse_sum;
        }
        if (lane == 0u) {
            out_and_lse[output_elements + q_batch_head] =
                running_max_shared[local_head] + log(running_sum_shared[local_head]);
        }
        local_head += 1u;
    }
}
