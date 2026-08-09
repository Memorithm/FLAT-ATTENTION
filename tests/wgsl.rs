use flat_attention::FLAT_FWD_WGSL;
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[test]
fn fused_forward_shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(FLAT_FWD_WGSL)
        .unwrap_or_else(|err| panic!("WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|err| panic!("WGSL validation failed: {err:?}"));
}
