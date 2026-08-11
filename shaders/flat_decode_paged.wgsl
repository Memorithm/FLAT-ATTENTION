// M16: specialized q_len=1 decode over paged resident K/V storage.
//
// Logical storage:
//   Q          [1, q_heads * head_dim]
//   K/V        [physical_pages, page_size, kv_heads * head_dim]
//   O          [1, q_heads * head_dim]
//   LSE        [q_heads]
//
// The compact logical-page -> physical-page table is packed into the uniform
// block so the portable path retains only four storage bindings (Q/K/V/O|LSE).
// Only logical tokens [0, kv_len) are visible. K is already RoPE-rotated;
// Q RoPE is fused at q_rope_position. This first paged consumer is batch=1.

const WORKGROUP_SIZE: u32 = 64u;
const KV_TILE: u32 = 8u;
const MAX_HEAD_DIM: u32 = 128u;
const MAX_LOGICAL_PAGES: u32 = 256u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    kv_len: u32,
    page_size: u32,
    physical_pages: u32,
    mapped_pages: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    scale_bits: u32,
    theta_bits: u32,
    q_rope_position: u32,
    _pad0: u32,
    _pad1: u32,
    page_table: array<vec4<u32>, 64>,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 128>;
var<workgroup> k_shared: array<f32, 1024>;
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 64>;
var<workgroup> running_max_shared: f32;
var<workgroup> running_sum_shared: f32;
var<workgroup> alpha_shared: f32;
var<workgroup> p_shared: f32;

fn rope_pair(e: f32, o: f32, pair: u32, position: u32, head_dim: u32, theta: f32) -> vec2<f32> {
    let freq = pow(theta, -2.0 * f32(pair) / f32(head_dim));
    let angle = f32(position) * freq;
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(e * c - o * s, e * s + o * c);
}

fn physical_page(logical_page: u32) -> u32 {
    let packed = params.page_table[logical_page / 4u];
    return packed[logical_page % 4u];
}

@compute @workgroup_size(64, 1, 1)
fn flat_attention_decode_paged(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q_head = workgroup_id.x;
    let lane = local_id.x;
    if (q_head >= params.q_heads || params.kv_len == 0u || params.page_size == 0u || params.mapped_pages == 0u || params.mapped_pages > MAX_LOGICAL_PAGES || params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM || (params.head_dim & 1u) != 0u || params.q_heads == 0u || params.kv_heads == 0u) {
        return;
    }

    let group_size = params.q_heads / params.kv_heads;
    if (group_size == 0u) {
        return;
    }
    let kv_head = q_head / group_size;
    if (kv_head >= params.kv_heads) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let theta = bitcast<f32>(params.theta_bits);
    let q_width = params.q_heads * params.head_dim;
    let kv_width = params.kv_heads * params.head_dim;
    let q_base = q_head * params.head_dim;
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    if (d0 < params.head_dim) {
        q_shared[d0] = q[q_base + d0];
    }
    if (d1 < params.head_dim) {
        q_shared[d1] = q[q_base + d1];
    }
    if (lane == 0u) {
        running_max_shared = NEG_MAX_F32;
        running_sum_shared = 0.0;
        alpha_shared = 1.0;
        p_shared = 0.0;
    }
    workgroupBarrier();

    let half_dim = params.head_dim / 2u;
    var pair = lane;
    loop {
        if (pair >= half_dim) {
            break;
        }
        let base = 2u * pair;
        let rotated = rope_pair(
            q_shared[base],
            q_shared[base + 1u],
            pair,
            params.q_rope_position,
            params.head_dim,
            theta,
        );
        q_shared[base] = rotated.x;
        q_shared[base + 1u] = rotated.y;
        pair += WORKGROUP_SIZE;
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
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
            let key_pos = tile_start + tile_row;
            let logical_page = key_pos / params.page_size;
            let offset_in_page = key_pos - logical_page * params.page_size;
            let page = physical_page(logical_page);
            let physical_row = page * params.page_size + offset_in_page;
            let global_index = physical_row * kv_width + kv_head * params.head_dim + dim;
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
            var partial = 0.0;
            if (d0 < params.head_dim) {
                partial += q_shared[d0] * k_shared[shared_row + d0];
            }
            if (d1 < params.head_dim) {
                partial += q_shared[d1] * k_shared[shared_row + d1];
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

            if (d0 < params.head_dim) {
                acc0 = acc0 * alpha_shared + p_shared * v_shared[shared_row + d0];
            }
            if (d1 < params.head_dim) {
                acc1 = acc1 * alpha_shared + p_shared * v_shared[shared_row + d1];
            }
            workgroupBarrier();
            tile_row += 1u;
        }
        tile_start += tile_rows;
    }

    let inv_sum = 1.0 / running_sum_shared;
    if (d0 < params.head_dim) {
        out_and_lse[q_base + d0] = acc0 * inv_sum;
    }
    if (d1 < params.head_dim) {
        out_and_lse[q_base + d1] = acc1 * inv_sum;
    }
    if (lane == 0u) {
        out_and_lse[q_width + q_head] = running_max_shared + log(running_sum_shared);
    }
}
