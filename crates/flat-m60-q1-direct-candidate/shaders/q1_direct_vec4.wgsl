// FLAT-ATTENTION M60 direct-load Q1 vec4 MHA candidate.
//
// M58 already removed Q4 query-row multiplexing, but it still staged each K/V
// tile through workgroup memory even though a Q1 workgroup consumes every K/V
// element exactly once. M60 changes only that mechanism: each active lane loads
// its own K and V vec4 directly from storage for the current key row. The M58
// 64-lane reduction and online-softmax synchronization remain unchanged so the
// benchmark isolates the value of removing non-reused K/V staging.

const WORKGROUP_SIZE: u32 = 64u;
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

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> reduce_shared: array<f32, 64>;
var<workgroup> running_max_shared: f32;
var<workgroup> running_sum_shared: f32;
var<workgroup> alpha_shared: f32;
var<workgroup> p_shared: f32;

@compute @workgroup_size(64, 1, 1)
fn flat_attention_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let query_pos = workgroup_id.x;
    let bh = workgroup_id.y;
    let lane = local_id.x;

    if (query_pos >= params.seq_len || bh >= params.batch_heads ||
        (params.head_dim != 64u && params.head_dim != 128u)) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let head_dim4 = params.head_dim / 4u;
    let head_stride = params.seq_len * params.head_dim;
    let head_base = bh * head_stride;
    let head_stride4 = params.seq_len * head_dim4;
    let head_base4 = bh * head_stride4;
    let output_elems = params.batch_heads * head_stride;
    let query_base4 = head_base4 + query_pos * head_dim4;

    var q_reg = vec4<f32>(0.0);
    if (lane < head_dim4) {
        q_reg = q[query_base4 + lane];
    }

    var acc = vec4<f32>(0.0);

    if (lane == 0u) {
        running_max_shared = NEG_MAX_F32;
        running_sum_shared = 0.0;
    }
    workgroupBarrier();

    let kv_limit = select(params.seq_len, query_pos + 1u, params.causal != 0u);

    var key_pos = 0u;
    loop {
        if (key_pos >= kv_limit) {
            break;
        }

        var partial = 0.0;
        var v_slice = vec4<f32>(0.0);
        if (lane < head_dim4) {
            let global4 = head_base4 + key_pos * head_dim4 + lane;
            let k_slice = k[global4];
            v_slice = v[global4];
            partial = dot(q_reg, k_slice);
        }

        reduce_shared[lane] = partial;
        workgroupBarrier();

        var offset = 32u;
        loop {
            if (offset == 0u) {
                break;
            }
            if (lane < offset) {
                reduce_shared[lane] += reduce_shared[lane + offset];
            }
            workgroupBarrier();
            offset = offset / 2u;
        }

        if (lane == 0u) {
            let score = reduce_shared[0] * scale;
            let previous_max = running_max_shared;
            let new_max = max(previous_max, score);
            let alpha = select(
                exp(previous_max - new_max),
                0.0,
                running_sum_shared == 0.0,
            );
            let p = exp(score - new_max);
            running_max_shared = new_max;
            running_sum_shared = running_sum_shared * alpha + p;
            alpha_shared = alpha;
            p_shared = p;
        }
        workgroupBarrier();

        if (lane < head_dim4) {
            acc = acc * alpha_shared + p_shared * v_slice;
        }
        workgroupBarrier();
        key_pos += 1u;
    }

    let query_base = head_base + query_pos * params.head_dim;
    if (lane < head_dim4) {
        out_and_lse[query_base + lane * 4u] = acc.x / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 1u] = acc.y / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 2u] = acc.z / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 3u] = acc.w / running_sum_shared;
    }
    if (lane == 0u) {
        let lse_index = output_elems + bh * params.seq_len + query_pos;
        out_and_lse[lse_index] = running_max_shared + log(running_sum_shared);
    }
}
