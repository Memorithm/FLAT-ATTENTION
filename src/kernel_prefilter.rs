//! Static capability prefilter for kernel realizations (roadmap M24).
//!
//! The M24 capability model provides deterministic device limits; this module
//! completes its second half by checking candidate requirements **before**
//! pipeline creation so unsupported configurations are rejected with typed
//! reasons instead of opaque backend failures.
//!
//! Two checks are provided:
//!
//! - [`check_static`]: configuration-imposed facts only (subgroup support,
//!   workgroup invocations/storage, bind-group entries). Usable at context
//!   construction time when no problem is known yet.
//! - [`check_dispatch`]: problem-derived facts (dispatch extents and packed
//!   output binding size) against the same limits.
//!
//! [`check_module`] runs both. Every rejection is an explicit
//! [`CapabilityRejection`]; nothing here silently substitutes another
//! implementation. Marketing names are never consulted.

use crate::device_model::RuntimeDeviceCapabilities;
use crate::kernel_ir::{CapabilityRequirement, KernelModule, KernelResources};
use std::fmt;

/// Why a candidate realization was statically rejected on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CapabilityRejection {
    /// Required workgroup-addressable storage exceeds the device limit.
    WorkgroupStorageExceeded {
        /// Bytes required by the configuration.
        required_bytes: u64,
        /// Device maximum in bytes.
        maximum_bytes: u32,
    },
    /// Required invocations per workgroup exceed the X-axis limit.
    WorkgroupInvocationsExceeded {
        /// Invocations required by the configuration.
        required: u32,
        /// Device maximum along X.
        maximum: u32,
    },
    /// Required bind-group entries exceed the device limit.
    BindingEntriesExceeded {
        /// Entries required by the entry point.
        required: u32,
        /// Device maximum bind groups.
        maximum: u32,
    },
    /// The configuration requires subgroup operations the adapter lacks.
    SubgroupUnsupported,
    /// A dispatch axis extent exceeds the per-dimension device limit.
    DispatchExtentExceeded {
        /// Zero-based dispatch axis (`0` = x, `1` = y, `2` = z).
        axis: usize,
        /// Required workgroups along the axis.
        required: u64,
        /// Device maximum per dimension.
        maximum: u32,
    },
    /// Packed output binding exceeds the storage-buffer binding limit.
    OutputBindingExceeded {
        /// Required bytes for the packed `[O | LSE]` buffer.
        required_bytes: u64,
        /// Device maximum storage-buffer binding in bytes.
        maximum_bytes: u64,
    },
}

impl fmt::Display for CapabilityRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkgroupStorageExceeded {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "workgroup storage {required_bytes} bytes exceeds device maximum {maximum_bytes}"
            ),
            Self::WorkgroupInvocationsExceeded { required, maximum } => write!(
                f,
                "workgroup invocations {required} exceed device maximum {maximum}"
            ),
            Self::BindingEntriesExceeded { required, maximum } => write!(
                f,
                "{required} binding entries exceed device maximum {maximum}"
            ),
            Self::SubgroupUnsupported => {
                write!(f, "configuration requires unsupported subgroup operations")
            }
            Self::DispatchExtentExceeded {
                axis,
                required,
                maximum,
            } => write!(
                f,
                "dispatch axis {axis} requires {required} workgroups, device maximum is {maximum}"
            ),
            Self::OutputBindingExceeded {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "packed output binding of {required_bytes} bytes exceeds storage-binding maximum {maximum_bytes}"
            ),
        }
    }
}

impl std::error::Error for CapabilityRejection {}

/// Check configuration-static capability requirements against a device.
///
/// # Errors
///
/// Returns the first violated requirement in deterministic order.
pub fn check_static(
    requirements: &[CapabilityRequirement],
    capabilities: &RuntimeDeviceCapabilities,
) -> Result<(), CapabilityRejection> {
    for requirement in requirements {
        match *requirement {
            CapabilityRequirement::SubgroupOperations => {
                if !capabilities.subgroup_supported {
                    return Err(CapabilityRejection::SubgroupUnsupported);
                }
            }
            CapabilityRequirement::MinWorkgroupInvocations(required) => {
                if required > capabilities.max_workgroup_size_x {
                    return Err(CapabilityRejection::WorkgroupInvocationsExceeded {
                        required,
                        maximum: capabilities.max_workgroup_size_x,
                    });
                }
            }
            CapabilityRequirement::MinWorkgroupStorageBytes(required_bytes) => {
                let bytes = u64::from(required_bytes);
                let maximum = u64::from(capabilities.max_workgroup_storage_bytes);
                if bytes > maximum {
                    return Err(CapabilityRejection::WorkgroupStorageExceeded {
                        required_bytes: bytes,
                        maximum_bytes: capabilities.max_workgroup_storage_bytes,
                    });
                }
            }
            CapabilityRequirement::MinBindingEntries(required) => {
                if required > capabilities.max_binding_entries {
                    return Err(CapabilityRejection::BindingEntriesExceeded {
                        required,
                        maximum: capabilities.max_binding_entries,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Check problem-derived dispatch/output facts against a device.
///
/// # Errors
///
/// Returns the first violated fact in deterministic order (extents x/y/z,
/// then output binding size).
pub fn check_dispatch(
    resources: &KernelResources,
    capabilities: &RuntimeDeviceCapabilities,
) -> Result<(), CapabilityRejection> {
    for (axis, &extent) in resources.dispatch_extents.iter().enumerate() {
        let required = u64::from(extent);
        if required > u64::from(capabilities.max_workgroups_per_dimension) {
            return Err(CapabilityRejection::DispatchExtentExceeded {
                axis,
                required,
                maximum: capabilities.max_workgroups_per_dimension,
            });
        }
    }
    // Packed [O | LSE] output is bound as one read/write storage buffer.
    let required_bytes = resources.output_elements.checked_mul(4).ok_or(
        CapabilityRejection::OutputBindingExceeded {
            required_bytes: u64::MAX,
            maximum_bytes: u64::from(capabilities.max_storage_buffer_binding_size),
        },
    )?;
    let maximum_bytes = u64::from(capabilities.max_storage_buffer_binding_size);
    if required_bytes > maximum_bytes {
        return Err(CapabilityRejection::OutputBindingExceeded {
            required_bytes,
            maximum_bytes,
        });
    }
    Ok(())
}

/// Run both static and problem-derived checks for one module.
///
/// # Errors
///
/// Returns [`CapabilityRejection`] for the first violated fact.
pub fn check_module(
    module: &KernelModule,
    capabilities: &RuntimeDeviceCapabilities,
) -> Result<(), CapabilityRejection> {
    check_static(&module.config().static_requirements(), capabilities)?;
    check_dispatch(&module.resources(), capabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::{
        AttentionProblem, KernelConfig, KernelFamily, KvStaging, ScoreReduction, VectorWidth,
    };
    use crate::{AttentionShape, FlatAttentionConfig};

    fn caps() -> RuntimeDeviceCapabilities {
        RuntimeDeviceCapabilities {
            max_workgroups_per_dimension: 65535,
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_workgroup_storage_bytes: 32768,
            max_binding_entries: 8,
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            subgroup_supported: true,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f16_supported: true,
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
    fn qualified_variants_pass_on_reasonable_synthetic_device() {
        for config in [
            KernelConfig::PORTABLE_SCALAR,
            KernelConfig::PORTABLE_VEC4,
            KernelConfig::DOUBLE_BUFFERED_VEC4,
        ] {
            assert_eq!(
                check_module(&module(config), &caps()),
                Ok(()),
                "variant must pass: {config:?}"
            );
        }
    }

    #[test]
    fn subgroup_variant_requires_subgroup_capability() {
        let m = module(KernelConfig::SUBGROUP_ASSISTED);
        let mut without_subgroup = caps();
        without_subgroup.subgroup_supported = false;
        assert_eq!(
            check_module(&m, &without_subgroup),
            Err(CapabilityRejection::SubgroupUnsupported)
        );
        assert_eq!(check_module(&m, &caps()), Ok(()));
    }

    #[test]
    fn storage_boundary_is_exact_limit_minus_one() {
        let m = module(KernelConfig::PORTABLE_SCALAR);
        let required = m.resources().workgroup_storage_bytes;
        let mut c = caps();
        c.max_workgroup_storage_bytes = required as u32;
        assert_eq!(check_module(&m, &c), Ok(()));
        c.max_workgroup_storage_bytes = required as u32 - 1;
        assert_eq!(
            check_module(&m, &c),
            Err(CapabilityRejection::WorkgroupStorageExceeded {
                required_bytes: required,
                maximum_bytes: required as u32 - 1,
            })
        );
    }

    #[test]
    fn invocation_boundary_is_exact_limit_minus_one() {
        let mut c = caps();
        let reqs = KernelConfig::PORTABLE_SCALAR.static_requirements();
        c.max_workgroup_size_x = 63;
        assert_eq!(
            check_static(&reqs, &c),
            Err(CapabilityRejection::WorkgroupInvocationsExceeded {
                required: 64,
                maximum: 63,
            })
        );
        c.max_workgroup_size_x = 64;
        assert_eq!(check_static(&reqs, &c), Ok(()));
    }

    #[test]
    fn binding_boundary_is_exact() {
        let reqs = KernelConfig::PORTABLE_SCALAR.static_requirements();
        let mut c = caps();
        c.max_binding_entries = 5;
        assert_eq!(check_static(&reqs, &c), Ok(()));
        c.max_binding_entries = 4;
        assert_eq!(
            check_static(&reqs, &c),
            Err(CapabilityRejection::BindingEntriesExceeded {
                required: 5,
                maximum: 4,
            })
        );
    }

    #[test]
    fn dispatch_extents_are_checked_per_axis() {
        let m = module(KernelConfig::PORTABLE_SCALAR);
        let mut c = caps();
        c.max_workgroups_per_dimension = 32;
        // ceil(129/4)=33 tiles on axis x trips first.
        assert_eq!(
            check_dispatch(&m.resources(), &c),
            Err(CapabilityRejection::DispatchExtentExceeded {
                axis: 0,
                required: 33,
                maximum: 32,
            })
        );
        c.max_workgroups_per_dimension = 33;
        // y folds 2 batches * 4 heads = 8, still under 33; passes now.
        assert_eq!(check_dispatch(&m.resources(), &c), Ok(()));
    }

    #[test]
    fn output_binding_boundary_is_exact() {
        let m = module(KernelConfig::PORTABLE_SCALAR);
        let required_bytes = m.resources().output_elements * 4;
        let mut c = caps();
        c.max_storage_buffer_binding_size = u32::try_from(required_bytes).unwrap();
        assert_eq!(check_dispatch(&m.resources(), &c), Ok(()));
        c.max_storage_buffer_binding_size = u32::try_from(required_bytes - 1).unwrap();
        assert_eq!(
            check_dispatch(&m.resources(), &c),
            Err(CapabilityRejection::OutputBindingExceeded {
                required_bytes,
                maximum_bytes: required_bytes - 1,
            })
        );
    }

    #[test]
    fn rejection_order_is_deterministic() {
        // A device failing everything reports subgroup first (requirement
        // order), demonstrating deterministic first-failure semantics.
        let m = module(KernelConfig::SUBGROUP_ASSISTED);
        let mut c = caps();
        c.subgroup_supported = false;
        c.max_workgroup_storage_bytes = 0;
        c.max_binding_entries = 0;
        assert_eq!(
            check_module(&m, &c),
            Err(CapabilityRejection::SubgroupUnsupported)
        );

        // Without subgroup involvement, invocation limits still hold on this
        // synthetic device, so the first failure is the zeroed storage limit
        // reported ahead of the binding limit.
        let tree = module(KernelConfig::PORTABLE_SCALAR);
        assert_eq!(
            check_module(&tree, &c),
            Err(CapabilityRejection::WorkgroupStorageExceeded {
                required_bytes: 11328,
                maximum_bytes: 0,
            })
        );
    }

    #[test]
    fn vec4_double_buffer_requirements_exceed_scalar_footprint_check() {
        let scalar = module(KernelConfig::PORTABLE_SCALAR)
            .config()
            .static_requirements();
        let double = module(KernelConfig::DOUBLE_BUFFERED_VEC4)
            .config()
            .static_requirements();
        let storage = |reqs: &[CapabilityRequirement]| -> u64 {
            reqs.iter()
                .find_map(|r| match r {
                    CapabilityRequirement::MinWorkgroupStorageBytes(b) => Some(u64::from(*b)),
                    _ => None,
                })
                .unwrap_or(0)
        };
        assert!(storage(&double) > storage(&scalar));
    }

    #[test]
    fn staging_enums_remain_closed_over_real_machinery() {
        // Guards accidental widening of tuning dimensions without codegen.
        assert_ne!(
            KvStaging::SingleBuffered.buffers(),
            KvStaging::DoubleBuffered.buffers()
        );
        assert_ne!(
            VectorWidth::Scalar.components(),
            VectorWidth::Vec4.components()
        );
        assert_ne!(
            ScoreReduction::WorkgroupTree.canonical(),
            ScoreReduction::SubgroupAssisted.canonical()
        );
    }
}
