struct Params {
    addresses: u32,
    vectors_per_address: u32,
    tile_count: u32,
    _padding: u32,
};

var<workgroup> tile_values: array<vec4<u32>, 8>;

@group(0) @binding(0) var<storage, read_write> state_vectors: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(8)
fn pvp_subset_zeta_tile8(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let linear_group = workgroup_id.x;
    let vector_index = linear_group % params.vectors_per_address;
    let tile_index = linear_group / params.vectors_per_address;
    if (tile_index >= params.tile_count) {
        return;
    }

    let lane = local_id.x;
    let address = tile_index * 8u + lane;
    if (address >= params.addresses) {
        return;
    }

    let state_index = address * params.vectors_per_address + vector_index;
    var value = state_vectors[state_index];
    tile_values[lane] = value;
    workgroupBarrier();

    // stride 1
    if ((lane & 1u) != 0u) {
        value = value ^ tile_values[lane - 1u];
    }
    workgroupBarrier();
    tile_values[lane] = value;
    workgroupBarrier();

    // stride 2
    if ((lane & 2u) != 0u) {
        value = value ^ tile_values[lane - 2u];
    }
    workgroupBarrier();
    tile_values[lane] = value;
    workgroupBarrier();

    // stride 4
    if ((lane & 4u) != 0u) {
        value = value ^ tile_values[lane - 4u];
    }
    workgroupBarrier();

    state_vectors[state_index] = value;
}
