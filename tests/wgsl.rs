use flat_attention::{
    FLAT_FWD_F16_WGSL, FLAT_FWD_GROUPED_ROPE_WGSL, FLAT_FWD_GROUPED_WGSL,
    FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL, FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL,
    FLAT_FWD_PROJECTION_ROPE_WGSL, FLAT_FWD_SINGLE_WGSL, FLAT_FWD_SUBGROUP_WGSL, FLAT_FWD_WGSL,
};
use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

const FLAT_FWD_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_vec4.wgsl");
const FLAT_FWD_DOUBLE_BUFFER_WGSL: &str = include_str!("../shaders/flat_fwd_double_buffer.wgsl");

fn validate_shader_with_capabilities(name: &str, source: &str, capabilities: Capabilities) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{name} WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} WGSL validation failed: {err:?}"));
}

fn validate_shader(name: &str, source: &str) {
    validate_shader_with_capabilities(name, source, Capabilities::empty());
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
fn grouped_q4_shader_parses_and_validates() {
    validate_shader("Q4 grouped", FLAT_FWD_GROUPED_WGSL);
}

#[test]
fn fused_rotary_grouped_q4_shader_parses_and_validates() {
    validate_shader("Q4 grouped RoPE", FLAT_FWD_GROUPED_ROPE_WGSL);
}

#[test]
fn projection_rotary_grouped_q4_shader_parses_and_validates() {
    validate_shader("Q4 projection RoPE", FLAT_FWD_PROJECTION_ROPE_WGSL);
}

#[test]
fn asymmetric_projection_rotary_grouped_q4_shader_parses_and_validates() {
    validate_shader(
        "Q4 asymmetric projection RoPE",
        FLAT_FWD_PROJECTION_ROPE_ASYMMETRIC_WGSL,
    );
}

#[test]
fn variable_projection_rotary_grouped_q4_shader_parses_and_validates() {
    validate_shader(
        "Q4 variable-length projection RoPE",
        FLAT_FWD_PROJECTION_ROPE_VARIABLE_WGSL,
    );
}

#[test]
fn vec4_q4_shader_parses_and_validates() {
    validate_shader("Q4 vec4", FLAT_FWD_VEC4_WGSL);
}

#[test]
fn double_buffer_q4_shader_parses_and_validates() {
    validate_shader("Q4 double-buffer", FLAT_FWD_DOUBLE_BUFFER_WGSL);
}

#[test]
fn f16_q4_shader_parses_and_validates() {
    validate_shader_with_capabilities(
        "Q4 f16",
        FLAT_FWD_F16_WGSL,
        Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    );
}

#[test]
fn subgroup_q4_shader_parses_and_validates() {
    validate_subgroup_shader("Q4 subgroup", FLAT_FWD_SUBGROUP_WGSL);
}

#[test]
fn qualified_single_row_shader_parses_and_validates() {
    validate_shader("single-row", FLAT_FWD_SINGLE_WGSL);
}
