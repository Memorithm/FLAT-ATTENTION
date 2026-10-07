// Same packed-address representation; exactly one low logical stage per dispatch.
struct Params {
    addresses: u32,
    gates: u32,
    vectors_per_gate: u32,
    stage: u32,
};
@group(0) @binding(0) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pvp_packed_one_stage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.gates * params.vectors_per_gate) { return; }
    var value = state[id.x];
    switch params.stage {
        case 0u: { value ^= (value & vec4<u32>(0x55555555u)) << vec4<u32>(1u); }
        case 1u: { value ^= (value & vec4<u32>(0x33333333u)) << vec4<u32>(2u); }
        case 2u: { value ^= (value & vec4<u32>(0x0f0f0f0fu)) << vec4<u32>(4u); }
        case 3u: { value ^= (value & vec4<u32>(0x00ff00ffu)) << vec4<u32>(8u); }
        case 4u: { value ^= (value & vec4<u32>(0x0000ffffu)) << vec4<u32>(16u); }
        case 5u: { value.y ^= value.x; value.w ^= value.z; }
        case 6u: { value.z ^= value.x; value.w ^= value.y; }
        default: { return; }
    }
    state[id.x] = value;
}
