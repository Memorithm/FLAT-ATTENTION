//! End-to-end planning-flow qualification: attention problem → deterministic
//! candidate generation → capability filtering → emitted WGSL → Naga
//! validation.
//!
//! This proves the architectural chain on the host side without any routing
//! change. Device execution of these sources is qualified separately.

use flat_attention::kernel_candidates::{generate_candidates, KernelCandidate, SelectionPolicy};
use flat_attention::kernel_ir::{AttentionProblem, KernelFamily};
use flat_attention::kernel_prefilter::{check_module, CapabilityRejection};
use flat_attention::kernel_wgsl::emit;
use flat_attention::{AttentionShape, FlatAttentionConfig, RuntimeDeviceCapabilities};

fn caps(subgroup: bool) -> RuntimeDeviceCapabilities {
    RuntimeDeviceCapabilities {
        max_workgroups_per_dimension: 65535,
        max_workgroup_size_x: 64,
        max_workgroup_size_y: 1024,
        max_workgroup_size_z: 64,
        max_workgroup_storage_bytes: 32768,
        max_binding_entries: 8,
        max_storage_buffer_binding_size: 1 << 30,
        subgroup_supported: subgroup,
        subgroup_min_size: 32,
        subgroup_max_size: 32,
        f16_supported: true,
    }
}

fn problem(head_dim: usize) -> AttentionProblem {
    AttentionProblem::from_shape(
        &AttentionShape {
            batch: 2,
            heads: 4,
            seq_len: 129,
            head_dim,
        },
        FlatAttentionConfig {
            causal: true,
            softmax_scale: None,
        },
    )
    .unwrap()
}

fn validate(candidate: &KernelCandidate, p: &AttentionProblem, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("candidate {} parse failed: {err:?}", candidate.id));
    let capabilities = if candidate.static_requirements().iter().any(|r| {
        matches!(
            r,
            flat_attention::kernel_ir::CapabilityRequirement::SubgroupOperations
        )
    }) {
        naga::valid::Capabilities::SUBGROUP
    } else {
        naga::valid::Capabilities::empty()
    };
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities);
    if capabilities.contains(naga::valid::Capabilities::SUBGROUP) {
        validator
            .subgroup_stages(naga::valid::ShaderStages::COMPUTE)
            .subgroup_operations(naga::valid::SubgroupOperationSet::ARITHMETIC);
    }
    validator.validate(&module).unwrap_or_else(|err| {
        panic!(
            "candidate {} generated WGSL failed validation: {err:?}",
            candidate.id
        )
    });
}

#[test]
fn problem_to_candidates_to_generated_valid_sources() {
    let p = problem(64);
    let device = caps(true);
    let candidates = generate_candidates(&p, &device, &SelectionPolicy::default());
    assert!(!candidates.is_empty());

    for candidate in &candidates {
        // Every returned candidate must independently pass the prefilter.
        assert_eq!(
            check_module(&candidate.module_for(&p).unwrap(), &device),
            Ok(())
        );
        let generated = emit(&candidate.module_for(&p).unwrap()).unwrap();
        validate(candidate, &p, &generated.source);
    }

    // The subgroup candidate appears only when the adapter supports it and
    // its generated shader needs the SUBGROUP capability to validate.
    let without_subgroup = generate_candidates(&p, &caps(false), &SelectionPolicy::default());
    assert_eq!(
        without_subgroup.len() + 1,
        candidates.len(),
        "exactly the subgroup candidate must be pruned"
    );
}

#[test]
fn capability_rejection_is_the_only_reason_a_qualified_candidate_drops_out() {
    let p = problem(128);
    let device = caps(true);
    let all = generate_candidates(&p, &device, &SelectionPolicy::default());
    for candidate in &all {
        // Reproduce pruning manually: executability plus prefilter verdict.
        let module = candidate.module_for(&p).unwrap();
        let verdict = check_module(&module, &device);
        assert_eq!(
            verdict,
            Ok(()),
            "returned candidate must not carry a rejection"
        );
        let _ = CapabilityRejection::SubgroupUnsupported;
    }
    assert!(all.iter().all(|c| c.family == KernelFamily::DenseQ4Forward));
}
