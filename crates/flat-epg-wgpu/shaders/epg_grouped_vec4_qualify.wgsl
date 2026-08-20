// Correctness-first EPG control kernel.
//
// This kernel deliberately prioritizes an auditable scalar execution order over
// throughput. Q/K/V are read as vec4<f32>; no score/probability matrix and no
// rotated Q/K tensor is materialized. One workgroup computes one query row.

const MAX_HEAD_DIM: u32 = 128u;

struct Params {
    seq_len: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    batch: u32,
    causal: u32,
    scale_bits: u32,
    theta_bits: u32,
    position_offset: u32,
    so4_dims: u32,
    geometry_mode: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn rotate_pair(pair: vec2<f32>, frequency: f32, position: u32) -> vec2<f32> {
    let angle = f32(position) * frequency;
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(pair.x * c - pair.y * s, pair.x * s + pair.y * c);
}

fn rotate_block(
    value: vec4<f32>,
    vec_col: u32,
    position: u32,
    head_dim: u32,
    theta: f32,
    so4_dims: u32,
    geometry_mode: u32,
) -> vec4<f32> {
    let pair0 = 2u * vec_col;
    let pair1 = pair0 + 1u;
    let frequency0 = pow(theta, -2.0 * f32(pair0) / f32(head_dim));
    var frequency1 = pow(theta, -2.0 * f32(pair1) / f32(head_dim));

    let first_dim = 4u * vec_col;
    let so4_start = head_dim - so4_dims;
    let in_so4_tail = so4_dims != 0u && first_dim >= so4_start;
    if (geometry_mode == 2u && in_so4_tail) {
        frequency1 = frequency0;
    }

    let xy = rotate_pair(value.xy, frequency0, position);
    let zw = rotate_pair(value.zw, frequency1, position);
    return vec4<f32>(xy.x, xy.y, zw.x, zw.y);
}

@compute @workgroup_size(1, 1, 1)
fn epg_grouped_vec4_qualify(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let query_pos = workgroup_id.x;
    let q_batch_head = workgroup_id.y;
    let q_batch_heads = params.batch * params.q_heads;

    if (
        query_pos >= params.seq_len ||
        q_batch_head >= q_batch_heads ||
        params.seq_len == 0u ||
        params.head_dim == 0u ||
        params.head_dim > MAX_HEAD_DIM ||
        (params.head_dim & 3u) != 0u ||
        params.q_heads == 0u ||
        params.kv_heads == 0u ||
        params.q_heads % params.kv_heads != 0u ||
        params.so4_dims > params.head_dim ||
        (params.so4_dims & 3u) != 0u ||
        params.geometry_mode > 2u
    ) {
        return;
    }

    let group_size = params.q_heads / params.kv_heads;
    let batch_index = q_batch_head / params.q_heads;
    let q_head = q_batch_head - batch_index * params.q_heads;
    let kv_head = q_head / group_size;
    let head_dim4 = params.head_dim / 4u;
    let q_head_base4 = (batch_index * params.q_heads + q_head) * params.seq_len * head_dim4;
    let kv_head_base4 = (batch_index * params.kv_heads + kv_head) * params.seq_len * head_dim4;
    let q_row_base4 = q_head_base4 + query_pos * head_dim4;
    let scale = bitcast<f32>(params.scale_bits);
    let theta = bitcast<f32>(params.theta_bits);
    let query_position = params.position_offset + query_pos;

    var running_max = -3.402823466e38;
    var running_sum = 0.0;
    var accumulator: array<f32, 128>;

    for (var key_pos = 0u; key_pos < params.seq_len; key_pos += 1u) {
        if (params.causal != 0u && key_pos > query_pos) {
            break;
        }

        let key_position = params.position_offset + key_pos;
        let kv_row_base4 = kv_head_base4 + key_pos * head_dim4;
        var rotated_dot = 0.0;

        for (var vec_col = 0u; vec_col < head_dim4; vec_col += 1u) {
            let qr = rotate_block(
                q[q_row_base4 + vec_col],
                vec_col,
                query_position,
                params.head_dim,
                theta,
                params.so4_dims,
                params.geometry_mode,
            );
            let kr = rotate_block(
                k[kv_row_base4 + vec_col],
                vec_col,
                key_position,
                params.head_dim,
                theta,
                params.so4_dims,
                params.geometry_mode,
            );
            rotated_dot += dot(qr, kr);
        }

        let score = rotated_dot * scale;
        let new_max = max(running_max, score);
        var alpha = 0.0;
        if (running_sum != 0.0) {
            alpha = exp(running_max - new_max);
        }
        let p = exp(score - new_max);

        for (var vec_col = 0u; vec_col < head_dim4; vec_col += 1u) {
            let packed_v = v[kv_row_base4 + vec_col];
            let dim = 4u * vec_col;
            accumulator[dim] = accumulator[dim] * alpha + p * packed_v.x;
            accumulator[dim + 1u] = accumulator[dim + 1u] * alpha + p * packed_v.y;
            accumulator[dim + 2u] = accumulator[dim + 2u] * alpha + p * packed_v.z;
            accumulator[dim + 3u] = accumulator[dim + 3u] * alpha + p * packed_v.w;
        }

        running_sum = running_sum * alpha + p;
        running_max = new_max;
    }

    let output_head_base = q_batch_head * params.seq_len * params.head_dim;
    let output_row_base = output_head_base + query_pos * params.head_dim;
    let inv_sum = 1.0 / running_sum;
    for (var dim = 0u; dim < params.head_dim; dim += 1u) {
        out_and_lse[output_row_base + dim] = accumulator[dim] * inv_sum;
    }

    let output_elements = params.batch * params.q_heads * params.seq_len * params.head_dim;
    let lse_index = output_elements + q_batch_head * params.seq_len + query_pos;
    out_and_lse[lse_index] = running_max + log(running_sum);
}
