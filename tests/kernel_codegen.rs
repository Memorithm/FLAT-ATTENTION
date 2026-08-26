//! Host-side Naga validation of Kernel-IR-generated WGSL (roadmap M21 gate).
//!
//! Generated sources must parse and validate exactly like the handwritten
//! shaders before any device-level qualification or routing use.

use flat_attention::kernel_ir::{AttentionProblem, KernelConfig, KernelFamily, KernelModule};
use flat_attention::kernel_wgsl::emit;
use flat_attention::{AttentionShape, FlatAttentionConfig};
use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};

fn module(config: KernelConfig) -> KernelModule {
    let problem = AttentionProblem::from_shape(
        &AttentionShape {
            batch: 2,
            heads: 4,
            seq_len: 33,
            head_dim: 64,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
    )
    .unwrap();
    KernelModule::build(KernelFamily::DenseQ4Forward, problem, config).unwrap()
}

fn validate(name: &str, source: &str, capabilities: Capabilities) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{name} generated WGSL parse failed: {err:?}"));
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .unwrap_or_else(|err| panic!("{name} generated WGSL validation failed: {err:?}"));
}

#[test]
fn generated_scalar_q4_parses_and_validates() {
    let generated = emit(&module(KernelConfig::PORTABLE_SCALAR)).unwrap();
    validate("Q4 scalar", &generated.source, Capabilities::empty());
}

#[test]
fn generated_vec4_q4_parses_and_validates() {
    let generated = emit(&module(KernelConfig::PORTABLE_VEC4)).unwrap();
    validate("Q4 vec4", &generated.source, Capabilities::empty());
}

#[test]
fn generated_double_buffered_q4_parses_and_validates() {
    let generated = emit(&module(KernelConfig::DOUBLE_BUFFERED_VEC4)).unwrap();
    validate("Q4 double-buffer", &generated.source, Capabilities::empty());
}

#[test]
fn generated_subgroup_q4_parses_and_validates_with_subgroup_capability() {
    let generated = emit(&module(KernelConfig::SUBGROUP_ASSISTED)).unwrap();
    let parsed = naga::front::wgsl::parse_str(&generated.source)
        .unwrap_or_else(|err| panic!("Q4 subgroup generated WGSL parse failed: {err:?}"));
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::SUBGROUP);
    validator
        .subgroup_stages(ShaderStages::COMPUTE)
        .subgroup_operations(SubgroupOperationSet::ARITHMETIC);
    validator
        .validate(&parsed)
        .unwrap_or_else(|err| panic!("Q4 subgroup generated WGSL validation failed: {err:?}"));
}

#[test]
fn generated_sources_validate_for_causal_and_non_causal_problems() {
    for causal in [true, false] {
        let problem = AttentionProblem::from_shape(
            &AttentionShape {
                batch: 1,
                heads: 2,
                seq_len: 65,
                head_dim: 128,
            },
            FlatAttentionConfig {
                causal,
                softmax_scale: None,
            },
        )
        .unwrap();
        let m = KernelModule::build(
            KernelFamily::DenseQ4Forward,
            problem,
            KernelConfig::PORTABLE_VEC4,
        )
        .unwrap();
        let generated = emit(&m).unwrap();
        validate(
            if causal {
                "vec4 causal"
            } else {
                "vec4 non-causal"
            },
            &generated.source,
            Capabilities::empty(),
        );
    }
}
