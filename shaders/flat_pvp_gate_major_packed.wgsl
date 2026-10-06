// Separate physical layout: one vec4 carries 128 addresses of one gate.
struct Params {
    addresses: u32,
    gates: u32,
    vectors_per_gate: u32,
    stride_vectors: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pvp_packed_prefix(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.gates * params.vectors_per_gate) { return; }
    var value = state[id.x];
    // Only logical stages are executed, preserving zero padding when K < 128.
    if (params.addresses > 1u) {
        value = value ^ ((value & vec4<u32>(0x55555555u)) << vec4<u32>(1u));
    }
    if (params.addresses > 2u) {
        value = value ^ ((value & vec4<u32>(0x33333333u)) << vec4<u32>(2u));
    }
    if (params.addresses > 4u) {
        value = value ^ ((value & vec4<u32>(0x0f0f0f0fu)) << vec4<u32>(4u));
    }
    if (params.addresses > 8u) {
        value = value ^ ((value & vec4<u32>(0x00ff00ffu)) << vec4<u32>(8u));
    }
    if (params.addresses > 16u) {
        value = value ^ ((value & vec4<u32>(0x0000ffffu)) << vec4<u32>(16u));
    }
    if (params.addresses > 32u) {
        value.y = value.y ^ value.x;
        value.w = value.w ^ value.z;
    }
    if (params.addresses > 64u) {
        value.z = value.z ^ value.x;
        value.w = value.w ^ value.y;
    }
    state[id.x] = value;
}

@compute @workgroup_size(64)
fn pvp_packed_suffix(@builtin(global_invocation_id) id: vec3<u32>) {
    let pairs_per_gate = params.vectors_per_gate / 2u;
    if (id.x >= params.gates * pairs_per_gate) { return; }
    let gate = id.x / pairs_per_gate;
    let pair = id.x % pairs_per_gate;
    let block = pair / params.stride_vectors;
    let offset = pair % params.stride_vectors;
    let source = gate * params.vectors_per_gate + block * (2u * params.stride_vectors) + offset;
    let target_index = source + params.stride_vectors;
    state[target_index] = state[target_index] ^ state[source];
}
