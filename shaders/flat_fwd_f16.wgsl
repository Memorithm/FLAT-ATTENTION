// FLAT-ATTENTION M8 packed-binary16 forward kernel.
//
// WGPU/Naga 0.20 does not parse native WGSL f16. M8 therefore stores two
// IEEE-754 binary16 values in each u32 and converts at the storage boundary
// with WGSL's pack2x16float/unpack2x16float builtins. All arithmetic remains
// f32: QK accumulation, score scaling, online-softmax state, exponentials and
// PV accumulation. O is packed back to binary16; LSE remains f32 bits.

const WORKGROUP_SIZE: u32 = 64u;
const QUERY_ROWS: u32 = 4u;
const KV_TILE: u32 = 8u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    seq_len: u32,
    head_dim: u32,
    batch_heads: u32,
    causal: u32,
    scale_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Each input word contains two consecutive binary16 scalars.
@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> k: array<u32>;
@group(0) @binding(2) var<storage, read> v: array<u32>;
// [O binary16 pairs | LSE f32 bit patterns].
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 512>;
var<workgroup> k_shared: array<f32, 1024>;
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 256>;
var<workgroup> output_shared: array<f32, 512>;
var<workgroup> running_max_shared: array<f32, 4>;
var<workgroup> running_sum_shared: array<f32, 4>;
var<workgroup> alpha_shared: array<f32, 4>;
var<workgroup> p_shared: array<f32, 4>;

@compute @workgroup_size(64, 1, 1)
fn flat_attention_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let query_start = workgroup_id.x * QUERY_ROWS;
    let bh = workgroup_id.y;
    let lane = local_id.x;

    if (query_start >= params.seq_len || bh >= params.batch_heads ||
        (params.head_dim != 64u && params.head_dim != 128u)) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let head_stride = params.seq_len * params.head_dim;
    let head_base = bh * head_stride;
    let head_dim2 = params.head_dim / 2u;
    let head_stride2 = params.seq_len * head_dim2;
    let head_base2 = bh * head_stride2;
    let output_words = (params.batch_heads * head_stride) / 2u;
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc20 = 0.0;
    var acc21 = 0.0;
    var acc30 = 0.0;
    var acc31 = 0.0;

    // Stage Q: one u32 storage load yields two f32 workgroup scalars.
    var q_pair = lane;
    let q_tile_pairs = QUERY_ROWS * head_dim2;
    loop {
        if (q_pair >= q_tile_pairs) {
            break;
        }
        let qr = q_pair / head_dim2;
        let pair_col = q_pair - qr * head_dim2;
        let query_pos = query_start + qr;
        if (query_pos < params.seq_len) {
            let packed = unpack2x16float(q[head_base2 + query_pos * head_dim2 + pair_col]);
            let shared_base = qr * MAX_HEAD_DIM + pair_col * 2u;
            q_shared[shared_base] = packed.x;
            q_shared[shared_base + 1u] = packed.y;
        }
        q_pair += WORKGROUP_SIZE;
    }

    if (lane == 0u) {
        var init_qr = 0u;
        loop {
            if (init_qr >= QUERY_ROWS) {
                break;
            }
            running_max_shared[init_qr] = NEG_MAX_F32;
            running_sum_shared[init_qr] = 0.0;
            alpha_shared[init_qr] = 1.0;
            p_shared[init_qr] = 0.0;
            init_qr += 1u;
        }
    }
    workgroupBarrier();

    let query_limit = min(params.seq_len, query_start + QUERY_ROWS);
    let kv_limit = select(params.seq_len, query_limit, params.causal != 0u);

    var tile_start = 0u;
    loop {
        if (tile_start >= kv_limit) {
            break;
        }
        let tile_rows = min(KV_TILE, kv_limit - tile_start);
        let tile_pairs = tile_rows * head_dim2;

        // Stage K/V: each u32 load yields two adjacent f32 values.
        var linear_pair = lane;
        loop {
            if (linear_pair >= tile_pairs) {
                break;
            }
            let tile_row = linear_pair / head_dim2;
            let pair_col = linear_pair - tile_row * head_dim2;
            let global_pair = head_base2 + (tile_start + tile_row) * head_dim2 + pair_col;
            let shared_base = tile_row * MAX_HEAD_DIM + pair_col * 2u;
            let packed_k = unpack2x16float(k[global_pair]);
            let packed_v = unpack2x16float(v[global_pair]);
            k_shared[shared_base] = packed_k.x;
            k_shared[shared_base + 1u] = packed_k.y;
            v_shared[shared_base] = packed_v.x;
            v_shared[shared_base + 1u] = packed_v.y;
            linear_pair += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var tile_row = 0u;
        loop {
            if (tile_row >= tile_rows) {
                break;
            }
            let key_pos = tile_start + tile_row;
            let shared_row = tile_row * MAX_HEAD_DIM;

            var qr = 0u;
            loop {
                if (qr >= QUERY_ROWS) {
                    break;
                }
                let query_pos = query_start + qr;
                let valid_query = query_pos < params.seq_len;
                let participates = valid_query && (params.causal == 0u || key_pos <= query_pos);
                let q_shared_base = qr * MAX_HEAD_DIM;
                let reduce_base = qr * WORKGROUP_SIZE;

                var partial = 0.0;
                if (participates && d0 < params.head_dim) {
                    partial += q_shared[q_shared_base + d0] * k_shared[shared_row + d0];
                }
                if (participates && d1 < params.head_dim) {
                    partial += q_shared[q_shared_base + d1] * k_shared[shared_row + d1];
                }
                reduce_shared[reduce_base + lane] = partial;
                workgroupBarrier();

                var offset = 32u;
                loop {
                    if (offset == 0u) {
                        break;
                    }
                    if (lane < offset) {
                        reduce_shared[reduce_base + lane] +=
                            reduce_shared[reduce_base + lane + offset];
                    }
                    workgroupBarrier();
                    offset = offset / 2u;
                }

                if (lane == 0u) {
                    if (participates) {
                        let score = reduce_shared[reduce_base] * scale;
                        let previous_max = running_max_shared[qr];
                        let new_max = max(previous_max, score);
                        let alpha = select(
                            exp(previous_max - new_max),
                            0.0,
                            running_sum_shared[qr] == 0.0,
                        );
                        let p = exp(score - new_max);
                        running_max_shared[qr] = new_max;
                        running_sum_shared[qr] = running_sum_shared[qr] * alpha + p;
                        alpha_shared[qr] = alpha;
                        p_shared[qr] = p;
                    } else {
                        alpha_shared[qr] = 1.0;
                        p_shared[qr] = 0.0;
                    }
                }
                workgroupBarrier();

                let alpha = alpha_shared[qr];
                let p = p_shared[qr];
                if (qr == 0u) {
                    if (d0 < params.head_dim) {
                        acc00 = acc00 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc01 = acc01 * alpha + p * v_shared[shared_row + d1];
                    }
                } else if (qr == 1u) {
                    if (d0 < params.head_dim) {
                        acc10 = acc10 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc11 = acc11 * alpha + p * v_shared[shared_row + d1];
                    }
                } else if (qr == 2u) {
                    if (d0 < params.head_dim) {
                        acc20 = acc20 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc21 = acc21 * alpha + p * v_shared[shared_row + d1];
                    }
                } else {
                    if (d0 < params.head_dim) {
                        acc30 = acc30 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc31 = acc31 * alpha + p * v_shared[shared_row + d1];
                    }
                }
                workgroupBarrier();
                qr += 1u;
            }
            tile_row += 1u;
        }

        workgroupBarrier();
        tile_start += KV_TILE;
    }

    // Normalize in f32, then pack two adjacent output scalars into binary16.
    var qr = 0u;
    loop {
        if (qr >= QUERY_ROWS) {
            break;
        }
        let query_pos = query_start + qr;
        if (query_pos < params.seq_len) {
            let inv_sum = 1.0 / running_sum_shared[qr];
            let shared_base = qr * MAX_HEAD_DIM;
            if (qr == 0u) {
                if (d0 < params.head_dim) {
                    output_shared[shared_base + d0] = acc00 * inv_sum;
                }
                if (d1 < params.head_dim) {
                    output_shared[shared_base + d1] = acc01 * inv_sum;
                }
            } else if (qr == 1u) {
                if (d0 < params.head_dim) {
                    output_shared[shared_base + d0] = acc10 * inv_sum;
                }
                if (d1 < params.head_dim) {
                    output_shared[shared_base + d1] = acc11 * inv_sum;
                }
            } else if (qr == 2u) {
                if (d0 < params.head_dim) {
                    output_shared[shared_base + d0] = acc20 * inv_sum;
                }
                if (d1 < params.head_dim) {
                    output_shared[shared_base + d1] = acc21 * inv_sum;
                }
            } else {
                if (d0 < params.head_dim) {
                    output_shared[shared_base + d0] = acc30 * inv_sum;
                }
                if (d1 < params.head_dim) {
                    output_shared[shared_base + d1] = acc31 * inv_sum;
                }
            }
            workgroupBarrier();

            let pair = lane;
            if (pair < head_dim2) {
                let dim = pair * 2u;
                let packed = pack2x16float(vec2<f32>(
                    output_shared[shared_base + dim],
                    output_shared[shared_base + dim + 1u],
                ));
                let output_word_base = (head_base + query_pos * params.head_dim) / 2u;
                out_and_lse[output_word_base + pair] = packed;
            }
            if (lane == 0u) {
                let lse_index = output_words + bh * params.seq_len + query_pos;
                let lse = running_max_shared[qr] + log(running_sum_shared[qr]);
                out_and_lse[lse_index] = bitcast<u32>(lse);
            }
            workgroupBarrier();
        }
        qr += 1u;
    }
}
