struct Params {
    addresses: u32,
    vectors_per_address: u32,
    block_count: u32,
    _padding: u32,
};

@group(0) @binding(0) var<storage, read_write> state_vectors: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pvp_subset_zeta_fused2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let total = params.block_count * params.vectors_per_address;
    if (linear >= total) {
        return;
    }

    let vector_index = linear % params.vectors_per_address;
    let block_index = linear / params.vectors_per_address;
    let address0 = block_index * 4u;
    if (address0 + 3u >= params.addresses) {
        return;
    }

    let base0 = address0 * params.vectors_per_address + vector_index;
    let base1 = base0 + params.vectors_per_address;
    let base2 = base1 + params.vectors_per_address;
    let base3 = base2 + params.vectors_per_address;

    let a0 = state_vectors[base0];
    var a1 = state_vectors[base1];
    var a2 = state_vectors[base2];
    var a3 = state_vectors[base3];

    // stride 1
    a1 = a1 ^ a0;
    a3 = a3 ^ a2;

    // stride 2, consuming the already-updated stride-1 values.
    a2 = a2 ^ a0;
    a3 = a3 ^ a1;

    state_vectors[base0] = a0;
    state_vectors[base1] = a1;
    state_vectors[base2] = a2;
    state_vectors[base3] = a3;
}
