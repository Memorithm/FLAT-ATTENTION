//! Deterministic bridge from a selected mathematical semantic to compatible
//! runtime execution families.
//!
//! This crate deliberately sits between semantic selection and the existing
//! kernel/device/autotuning machinery. It answers only one question:
//!
//! > Which already-implemented [`RuntimeKernelId`] values are declared to
//! > realize this exact [`SemanticId`] for this execution role?
//!
//! It does not choose a kernel, inspect device capabilities, time candidates,
//! or change the selected semantic when no compatible execution is available.
//! Those are separate decisions and evidence layers.

#![forbid(unsafe_code)]

use std::fmt::{Display, Formatter};

use flat_attention::RuntimeKernelId;
use flat_semantic::v1::{SemanticFamily, SemanticId};
use flat_semantic_control::SemanticSelectionDecision;

/// Version of the semantic-to-execution compatibility contract.
pub const SEMANTIC_EXECUTION_VERSION: u16 = 1;

/// Logical operation performed by one runtime implementation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionRole {
    /// Full/prefill-style forward evaluation.
    Forward,
    /// Incremental state/cache decode evaluation.
    Decode,
    /// Gradient/backward evaluation of the selected semantic.
    Backward,
}

/// One explicit compatibility assertion between semantic and runtime code.
///
/// This record is static capability metadata, not correctness or performance
/// evidence. A binding means only that the runtime family is intended to
/// implement the named semantic/role and may proceed to its normal correctness,
/// device-capability and autotuning gates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBinding {
    semantic: SemanticId,
    role: ExecutionRole,
    kernel: RuntimeKernelId,
}

impl ExecutionBinding {
    /// Construct one semantic/runtime compatibility assertion.
    #[must_use]
    pub const fn new(
        semantic: SemanticId,
        role: ExecutionRole,
        kernel: RuntimeKernelId,
    ) -> Self {
        Self {
            semantic,
            role,
            kernel,
        }
    }

    /// Exact mathematical semantic implemented by this binding.
    #[must_use]
    pub const fn semantic(&self) -> &SemanticId {
        &self.semantic
    }

    /// Logical execution role of the runtime family.
    #[must_use]
    pub const fn role(&self) -> ExecutionRole {
        self.role
    }

    /// Existing FLAT runtime family; no replacement kernel taxonomy is added.
    #[must_use]
    pub const fn kernel(&self) -> RuntimeKernelId {
        self.kernel
    }
}

/// Deterministic catalog of semantic/runtime compatibility assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticExecutionCatalog {
    bindings: Vec<ExecutionBinding>,
}

impl SemanticExecutionCatalog {
    /// Construct a deterministic catalog.
    ///
    /// Input order is not observable: bindings are sorted by stable semantic,
    /// role and existing runtime-kernel ranks. Exact duplicate assertions fail
    /// closed instead of being silently deduplicated.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionCatalogError::UnsupportedFamily`] for a future
    /// semantic family without a v1 stable rank, or
    /// [`ExecutionCatalogError::DuplicateBinding`] for an exact duplicate.
    pub fn new(
        bindings: impl IntoIterator<Item = ExecutionBinding>,
    ) -> Result<Self, ExecutionCatalogError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        for binding in &bindings {
            if semantic_family_rank(binding.semantic.family()).is_none() {
                return Err(ExecutionCatalogError::UnsupportedFamily {
                    semantic: binding.semantic.clone(),
                });
            }
        }
        bindings.sort_by(|left, right| binding_key(left).cmp(&binding_key(right)));
        if let Some(window) = bindings.windows(2).find(|window| window[0] == window[1]) {
            return Err(ExecutionCatalogError::DuplicateBinding {
                binding: window[0].clone(),
            });
        }
        Ok(Self { bindings })
    }

    /// Stable ordered bindings.
    #[must_use]
    pub fn bindings(&self) -> &[ExecutionBinding] {
        &self.bindings
    }

    /// Return runtime families compatible with the already-selected semantic
    /// and requested operation role.
    ///
    /// The function cannot perform semantic fallback: a no-selection decision
    /// or a selected semantic with no compatible runtime binding returns an
    /// empty list.
    #[must_use]
    pub fn compatible_kernels(
        &self,
        selection: &SemanticSelectionDecision,
        role: ExecutionRole,
    ) -> Vec<RuntimeKernelId> {
        let Some(selected) = selection.semantic() else {
            return Vec::new();
        };
        self.bindings
            .iter()
            .filter(|binding| binding.semantic() == selected && binding.role() == role)
            .map(ExecutionBinding::kernel)
            .collect()
    }
}

/// Build the compatibility catalog for the existing StandardSoftmax runtime
/// surface on `main`.
///
/// This is not a routing policy. It only states which existing runtime families
/// belong to StandardSoftmax v1 and which logical role each serves. Candidate
/// generation, device prefiltering, correctness gates and autotuning remain in
/// their existing layers.
#[must_use]
pub fn standard_softmax_runtime_catalog() -> SemanticExecutionCatalog {
    let semantic = SemanticId::new(SemanticFamily::StandardSoftmax, "standard-softmax", 1)
        .expect("the built-in StandardSoftmax v1 identity is valid");
    SemanticExecutionCatalog::new([
        ExecutionBinding::new(semantic.clone(), ExecutionRole::Forward, RuntimeKernelId::Q4Portable),
        ExecutionBinding::new(
            semantic.clone(),
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Vec4Portable,
        ),
        ExecutionBinding::new(
            semantic.clone(),
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Vec4DoubleBuffered,
        ),
        ExecutionBinding::new(semantic.clone(), ExecutionRole::Forward, RuntimeKernelId::Q4Subgroup),
        ExecutionBinding::new(
            semantic.clone(),
            ExecutionRole::Forward,
            RuntimeKernelId::GroupedForwardPortable,
        ),
        ExecutionBinding::new(
            semantic.clone(),
            ExecutionRole::Decode,
            RuntimeKernelId::ResidentDecodePortable,
        ),
        ExecutionBinding::new(
            semantic.clone(),
            ExecutionRole::Decode,
            RuntimeKernelId::PagedDecodePortable,
        ),
        ExecutionBinding::new(
            semantic,
            ExecutionRole::Backward,
            RuntimeKernelId::GroupedBackwardRecomputePortable,
        ),
    ])
    .expect("the built-in StandardSoftmax execution catalog has unique v1 bindings")
}

/// Typed catalog construction failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionCatalogError {
    /// A future semantic family has no stable execution-contract-v1 rank yet.
    UnsupportedFamily {
        /// Semantic identity whose family cannot be serialized deterministically.
        semantic: SemanticId,
    },
    /// The same semantic/role/runtime compatibility assertion was declared
    /// more than once.
    DuplicateBinding {
        /// Duplicated assertion.
        binding: ExecutionBinding,
    },
}

impl Display for ExecutionCatalogError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFamily { semantic } => write!(
                formatter,
                "semantic family for {} revision {} is not supported by execution contract v{}",
                semantic.name(),
                semantic.revision(),
                SEMANTIC_EXECUTION_VERSION
            ),
            Self::DuplicateBinding { binding } => write!(
                formatter,
                "duplicate semantic execution binding: {} revision {} / {:?} / {:?}",
                binding.semantic().name(),
                binding.semantic().revision(),
                binding.role(),
                binding.kernel()
            ),
        }
    }
}

impl std::error::Error for ExecutionCatalogError {}

fn binding_key(binding: &ExecutionBinding) -> (u8, &str, u32, u8, u8) {
    (
        semantic_family_rank(binding.semantic.family())
            .expect("catalog construction validates semantic-family ranks"),
        binding.semantic.name(),
        binding.semantic.revision(),
        execution_role_rank(binding.role),
        runtime_kernel_rank(binding.kernel),
    )
}

const fn semantic_family_rank(family: SemanticFamily) -> Option<u8> {
    match family {
        SemanticFamily::StandardSoftmax => Some(0),
        SemanticFamily::DifferentialSigned => Some(1),
        SemanticFamily::ToeplitzStructured => Some(2),
        SemanticFamily::ProlateConcentration => Some(3),
        SemanticFamily::GroundStateGreen => Some(4),
        SemanticFamily::SpectralFlow => Some(5),
        SemanticFamily::RecurrentMemory => Some(6),
        SemanticFamily::Hybrid => Some(7),
        SemanticFamily::Experimental => Some(8),
        _ => None,
    }
}

const fn execution_role_rank(role: ExecutionRole) -> u8 {
    match role {
        ExecutionRole::Forward => 0,
        ExecutionRole::Decode => 1,
        ExecutionRole::Backward => 2,
    }
}

const fn runtime_kernel_rank(kernel: RuntimeKernelId) -> u8 {
    match kernel {
        RuntimeKernelId::Q4Portable => 0,
        RuntimeKernelId::Q4Vec4Portable => 1,
        RuntimeKernelId::Q4Vec4DoubleBuffered => 2,
        RuntimeKernelId::Q4Subgroup => 3,
        RuntimeKernelId::GroupedForwardPortable => 4,
        RuntimeKernelId::ResidentDecodePortable => 5,
        RuntimeKernelId::PagedDecodePortable => 6,
        RuntimeKernelId::GroupedBackwardRecomputePortable => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_semantic_control::SemanticSelectionDecision;

    fn semantic(family: SemanticFamily, name: &str, revision: u32) -> SemanticId {
        SemanticId::new(family, name, revision).unwrap()
    }

    #[test]
    fn built_in_standard_softmax_catalog_exposes_existing_roles_only() {
        let catalog = standard_softmax_runtime_catalog();
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: standard,
            preference_rank: 0,
        };

        assert_eq!(
            catalog.compatible_kernels(&selection, ExecutionRole::Decode),
            vec![
                RuntimeKernelId::ResidentDecodePortable,
                RuntimeKernelId::PagedDecodePortable,
            ]
        );
        assert_eq!(
            catalog.compatible_kernels(&selection, ExecutionRole::Backward),
            vec![RuntimeKernelId::GroupedBackwardRecomputePortable]
        );
        assert_eq!(
            catalog.compatible_kernels(&selection, ExecutionRole::Forward),
            vec![
                RuntimeKernelId::Q4Portable,
                RuntimeKernelId::Q4Vec4Portable,
                RuntimeKernelId::Q4Vec4DoubleBuffered,
                RuntimeKernelId::Q4Subgroup,
                RuntimeKernelId::GroupedForwardPortable,
            ]
        );
    }

    #[test]
    fn unavailable_execution_never_changes_selected_semantic() {
        let recurrent = semantic(SemanticFamily::RecurrentMemory, "delta-memory", 1);
        let selection = SemanticSelectionDecision::Selected {
            semantic: recurrent.clone(),
            preference_rank: 0,
        };
        let catalog = standard_softmax_runtime_catalog();

        assert!(catalog
            .compatible_kernels(&selection, ExecutionRole::Forward)
            .is_empty());
        assert_eq!(selection.semantic(), Some(&recurrent));
    }

    #[test]
    fn no_semantic_selection_produces_no_execution_candidates() {
        let catalog = standard_softmax_runtime_catalog();
        let selection = SemanticSelectionDecision::NoRegisteredPreference;
        assert!(catalog
            .compatible_kernels(&selection, ExecutionRole::Forward)
            .is_empty());
    }

    #[test]
    fn catalog_order_is_input_order_independent_and_role_explicit() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let left = SemanticExecutionCatalog::new([
            ExecutionBinding::new(
                standard.clone(),
                ExecutionRole::Decode,
                RuntimeKernelId::PagedDecodePortable,
            ),
            ExecutionBinding::new(
                standard.clone(),
                ExecutionRole::Forward,
                RuntimeKernelId::Q4Subgroup,
            ),
        ])
        .unwrap();
        let right = SemanticExecutionCatalog::new([
            ExecutionBinding::new(
                standard.clone(),
                ExecutionRole::Forward,
                RuntimeKernelId::Q4Subgroup,
            ),
            ExecutionBinding::new(
                standard,
                ExecutionRole::Decode,
                RuntimeKernelId::PagedDecodePortable,
            ),
        ])
        .unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn duplicate_binding_fails_closed() {
        let standard = semantic(SemanticFamily::StandardSoftmax, "standard-softmax", 1);
        let binding = ExecutionBinding::new(
            standard,
            ExecutionRole::Forward,
            RuntimeKernelId::Q4Portable,
        );
        let error = SemanticExecutionCatalog::new([binding.clone(), binding.clone()]).unwrap_err();
        assert_eq!(error, ExecutionCatalogError::DuplicateBinding { binding });
    }
}
