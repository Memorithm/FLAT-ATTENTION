use flat_attention::{FLAT_FWD_SINGLE_WGSL, FLAT_FWD_SUBGROUP_WGSL, FLAT_FWD_WGSL};
use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

fn validate_shader(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{name} WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} WGSL validation failed: {err:?}"));
}

fn validate_subgroup_shader(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{name} WGSL parse failed: {err:?}"));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::SUBGROUP);
    validator
        .subgroup_stages(ShaderStages::COMPUTE)
        .subgroup_operations(SubgroupOperationSet::ARITHMETIC)
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} WGSL validation failed: {err:?}"));
}

#[test]
fn q4_fused_forward_shader_parses_and_validates() {
    validate_shader("Q4", FLAT_FWD_WGSL);
}

#[test]
fn subgroup_q4_shader_parses_and_validates() {
    validate_subgroup_shader("Q4 subgroup", FLAT_FWD_SUBGROUP_WGSL);
}

#[test]
fn qualified_single_row_shader_parses_and_validates() {
    validate_shader("single-row", FLAT_FWD_SINGLE_WGSL);
}
