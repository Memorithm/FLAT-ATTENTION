struct Params {
    addresses: u32,
    vectors_per_address: u32,
    stride: u32,
    pair_count: u32,
};

@group(0) @binding(0) var<storage, read_write> state_vectors: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pvp_subset_zeta_stage_vec4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let total = params.pair_count * params.vectors_per_address;
    if (linear >= total) {
        return;
    }

    let vector_index = linear % params.vectors_per_address;
    let pair_index = linear / params.vectors_per_address;
    let block = pair_index / params.stride;
    let offset = pair_index % params.stride;
    let source_address = block * (2u * params.stride) + offset;
    let target_address = source_address + params.stride;

    if (target_address >= params.addresses) {
        return;
    }

    let source_index = source_address * params.vectors_per_address + vector_index;
    let target_index = target_address * params.vectors_per_address + vector_index;
    state_vectors[target_index] = state_vectors[target_index] ^ state_vectors[source_index];
}
