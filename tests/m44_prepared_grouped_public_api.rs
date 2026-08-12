#![cfg(feature = "wgpu")]

use flat_attention::api::wgpu::PreparedGroupedForward;

#[test]
fn prepared_grouped_forward_type_is_publicly_nameable() {
    fn accepts_prepared(_: Option<PreparedGroupedForward>) {}
    accepts_prepared(None);
}
