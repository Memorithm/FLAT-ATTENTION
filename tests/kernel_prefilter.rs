//! Host-only M24 prefilter qualification against synthetic capability
//! profiles.
//!
//! Synthetic profiles are planning/protocol tests; they are never performance
//! evidence. The same semantic problem must change candidate eligibility
//! deterministically as capabilities change.

use flat_attention::kernel_ir::{AttentionProblem, KernelConfig, KernelFamily, KernelModule};
use flat_attention::kernel_prefilter::{check_module, CapabilityRejection};
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
        f16_supported: false,
    }
}

fn module(config: KernelConfig) -> KernelModule {
    let problem = AttentionProblem::from_shape(
        &AttentionShape {
            batch: 2,
            heads: 4,
            seq_len: 129,
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

#[test]
fn capability_removal_changes_eligibility_deterministically() {
    let with_subgroup = caps(true);
    let without_subgroup = caps(false);

    assert_eq!(
        check_module(&module(KernelConfig::SUBGROUP_ASSISTED), &with_subgroup),
        Ok(())
    );
    assert_eq!(
        check_module(&module(KernelConfig::SUBGROUP_ASSISTED), &without_subgroup),
        Err(CapabilityRejection::SubgroupUnsupported)
    );

    // The portable scalar path is unaffected by subgroup availability.
    assert_eq!(
        check_module(&module(KernelConfig::PORTABLE_SCALAR), &with_subgroup),
        Ok(())
    );
    assert_eq!(
        check_module(&module(KernelConfig::PORTABLE_SCALAR), &without_subgroup),
        Ok(())
    );
}

#[test]
fn identical_inputs_produce_identical_verdicts() {
    let c = caps(true);
    let a = check_module(&module(KernelConfig::DOUBLE_BUFFERED_VEC4), &c);
    let b = check_module(&module(KernelConfig::DOUBLE_BUFFERED_VEC4), &c);
    assert_eq!(a, b);
}

#[test]
fn minimal_portable_floor_still_admits_the_qualified_kernel() {
    // A downlevel device with exactly the historical portable floor.
    let floor = RuntimeDeviceCapabilities {
        max_workgroups_per_dimension: 65535,
        max_workgroup_size_x: 64,
        max_workgroup_size_y: 64,
        max_workgroup_size_z: 1,
        max_workgroup_storage_bytes: 16384,
        max_binding_entries: 5,
        max_storage_buffer_binding_size: 128 * 1024 * 1024,
        subgroup_supported: false,
        subgroup_min_size: 4,
        subgroup_max_size: 128,
        f16_supported: false,
    };
    assert_eq!(
        check_module(&module(KernelConfig::PORTABLE_SCALAR), &floor),
        Ok(())
    );
    assert_eq!(
        check_module(&module(KernelConfig::PORTABLE_VEC4), &floor),
        Ok(())
    );
}
