use flat_attention::{FLAT_FWD_SINGLE_WGSL, FLAT_FWD_WGSL};
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn validate_shader(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{name} WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} WGSL validation failed: {err:?}"));
}

#[test]
fn q4_fused_forward_shader_parses_and_validates() {
    validate_shader("Q4", FLAT_FWD_WGSL);
}

#[test]
fn qualified_single_row_shader_parses_and_validates() {
    validate_shader("single-row", FLAT_FWD_SINGLE_WGSL);
}
