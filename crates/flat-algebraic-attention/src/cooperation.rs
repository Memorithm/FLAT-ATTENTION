use core::fmt;

use crate::f2::F2AffinePredicate;
use crate::max_plus::MaxPlusValue;
use crate::zhegalkin::{ZhegalkinError, ZhegalkinPolynomial};

const M13B_MASK_SCHEMA_VERSION: u32 = 1;
const M13B_SIGNATURE_SCHEMA_VERSION: u32 = 1;
const M13B_BOOLEAN_KV_SCHEMA_VERSION: u32 = 1;
const PACKED_WORD_BITS: usize = u64::BITS as usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlgebraDomain {
    Boolean,
    F2,
    Zhegalkin,
    MaxPlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionQuality {
    Exact,
    Lossless,
    Restricted,
    Approximate,
    PolicyDefined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeRestriction {
    PositiveBooleanSemiring,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooperationTarget {
    Single(AlgebraDomain),
    Split {
        primary: AlgebraDomain,
        residual: AlgebraDomain,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlgebraicEvidence {
    source: AlgebraDomain,
    target: CooperationTarget,
    quality: ConversionQuality,
    restriction: Option<BridgeRestriction>,
}

impl AlgebraicEvidence {
    #[must_use]
    pub const fn source(&self) -> AlgebraDomain {
        self.source
    }

    #[must_use]
    pub const fn target(&self) -> CooperationTarget {
        self.target
    }

    #[must_use]
    pub const fn quality(&self) -> ConversionQuality {
        self.quality
    }

    #[must_use]
    pub const fn restriction(&self) -> Option<BridgeRestriction> {
        self.restriction
    }
}

/// Dependency-free adapter carrying canonical M13B Boolean survivor evidence.
///
/// The values are copied from the already-validated M13B mask/signature/Boolean
/// KV contracts by the caller. This adapter validates the serialized geometry
/// again and retains it verbatim so the multi-algebra route cannot collapse the
/// Boolean side into a capability flag. It deliberately does not interpret or
/// regenerate numerical K/V data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct M13bBooleanRoutingEvidence {
    mask_schema_version: u32,
    blocks: usize,
    mask_words: Vec<u64>,
    signature_schema_version: u32,
    signature_bits: usize,
    query_signature_words: Vec<u64>,
    key_signature_words: Vec<u64>,
    boolean_kv_schema_version: Option<u32>,
    boolean_kv_generation: Option<u64>,
}

impl M13bBooleanRoutingEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mask_schema_version: u32,
        blocks: usize,
        mask_words: Vec<u64>,
        signature_schema_version: u32,
        signature_bits: usize,
        query_signature_words: Vec<u64>,
        key_signature_words: Vec<u64>,
        boolean_kv_schema_version: Option<u32>,
        boolean_kv_generation: Option<u64>,
    ) -> Result<Self, CooperationError> {
        if mask_schema_version != M13B_MASK_SCHEMA_VERSION {
            return Err(CooperationError::UnsupportedM13bSchema {
                contract: "boolean-attention-mask",
                expected: M13B_MASK_SCHEMA_VERSION,
                actual: mask_schema_version,
            });
        }
        validate_packed("Boolean attention mask", blocks, &mask_words)?;

        if signature_schema_version != M13B_SIGNATURE_SCHEMA_VERSION {
            return Err(CooperationError::UnsupportedM13bSchema {
                contract: "boolean-attention-signature",
                expected: M13B_SIGNATURE_SCHEMA_VERSION,
                actual: signature_schema_version,
            });
        }
        validate_packed("query Boolean attention signature", signature_bits, &query_signature_words)?;
        validate_packed("key Boolean attention signature", signature_bits, &key_signature_words)?;

        match (boolean_kv_schema_version, boolean_kv_generation) {
            (None, None) => {}
            (Some(version), Some(_)) if version == M13B_BOOLEAN_KV_SCHEMA_VERSION => {}
            (Some(version), Some(_)) => {
                return Err(CooperationError::UnsupportedM13bSchema {
                    contract: "boolean-kv",
                    expected: M13B_BOOLEAN_KV_SCHEMA_VERSION,
                    actual: version,
                });
            }
            _ => return Err(CooperationError::IncompleteBooleanKvProvenance),
        }

        Ok(Self {
            mask_schema_version,
            blocks,
            mask_words,
            signature_schema_version,
            signature_bits,
            query_signature_words,
            key_signature_words,
            boolean_kv_schema_version,
            boolean_kv_generation,
        })
    }

    #[must_use]
    pub const fn mask_schema_version(&self) -> u32 {
        self.mask_schema_version
    }

    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }

    #[must_use]
    pub fn mask_words(&self) -> &[u64] {
        &self.mask_words
    }

    #[must_use]
    pub const fn signature_schema_version(&self) -> u32 {
        self.signature_schema_version
    }

    #[must_use]
    pub const fn signature_bits(&self) -> usize {
        self.signature_bits
    }

    #[must_use]
    pub fn query_signature_words(&self) -> &[u64] {
        &self.query_signature_words
    }

    #[must_use]
    pub fn key_signature_words(&self) -> &[u64] {
        &self.key_signature_words
    }

    #[must_use]
    pub const fn boolean_kv_generation(&self) -> Option<u64> {
        self.boolean_kv_generation
    }

    #[must_use]
    pub const fn boolean_kv_schema_version(&self) -> Option<u32> {
        self.boolean_kv_schema_version
    }
}

fn validate_packed(
    contract: &'static str,
    bit_len: usize,
    words: &[u64],
) -> Result<(), CooperationError> {
    if bit_len == 0 {
        return Err(CooperationError::ZeroM13bWidth { contract });
    }
    let expected_words = bit_len.div_ceil(PACKED_WORD_BITS);
    if words.len() != expected_words {
        return Err(CooperationError::M13bWordCountMismatch {
            contract,
            bit_len,
            expected_words,
            actual_words: words.len(),
        });
    }
    let tail_bits = bit_len % PACKED_WORD_BITS;
    if tail_bits != 0 {
        let valid_mask = (1u64 << tail_bits) - 1;
        if words.last().copied().unwrap_or_default() & !valid_mask != 0 {
            return Err(CooperationError::M13bNonZeroTailBits { contract });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttentionAlgebraNeeds {
    pub eligibility_logic: bool,
    pub parity_or_binary_linear: bool,
    pub nonlinear_boolean_interaction: bool,
    pub precedence_or_critical_path: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicRoute {
    domains: Vec<AlgebraDomain>,
    boolean_evidence: Option<M13bBooleanRoutingEvidence>,
}

impl AlgebraicRoute {
    #[must_use]
    pub fn domains(&self) -> &[AlgebraDomain] {
        &self.domains
    }

    #[must_use]
    pub fn contains(&self, domain: AlgebraDomain) -> bool {
        self.domains.contains(&domain)
    }

    #[must_use]
    pub const fn boolean_evidence(&self) -> Option<&M13bBooleanRoutingEvidence> {
        self.boolean_evidence.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CooperationError {
    NoRequestedSemantics,
    MissingBooleanRoutingEvidence,
    UnexpectedBooleanRoutingEvidence,
    UnsupportedM13bSchema {
        contract: &'static str,
        expected: u32,
        actual: u32,
    },
    ZeroM13bWidth {
        contract: &'static str,
    },
    M13bWordCountMismatch {
        contract: &'static str,
        bit_len: usize,
        expected_words: usize,
        actual_words: usize,
    },
    M13bNonZeroTailBits {
        contract: &'static str,
    },
    IncompleteBooleanKvProvenance,
    Zhegalkin(ZhegalkinError),
}

impl fmt::Display for CooperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRequestedSemantics => write!(
                formatter,
                "multi-algebra routing requires at least one requested semantic capability"
            ),
            Self::MissingBooleanRoutingEvidence => write!(
                formatter,
                "Boolean eligibility routing requires canonical M13B survivor evidence"
            ),
            Self::UnexpectedBooleanRoutingEvidence => write!(
                formatter,
                "M13B Boolean evidence was supplied without Boolean eligibility routing"
            ),
            Self::UnsupportedM13bSchema {
                contract,
                expected,
                actual,
            } => write!(
                formatter,
                "unsupported {contract} schema version: expected {expected}, got {actual}"
            ),
            Self::ZeroM13bWidth { contract } => {
                write!(formatter, "{contract} width must be non-zero")
            }
            Self::M13bWordCountMismatch {
                contract,
                bit_len,
                expected_words,
                actual_words,
            } => write!(
                formatter,
                "{contract} with {bit_len} bits requires {expected_words} u64 words, got {actual_words}"
            ),
            Self::M13bNonZeroTailBits { contract } => write!(
                formatter,
                "unused high bits in the final {contract} word must be zero"
            ),
            Self::IncompleteBooleanKvProvenance => write!(
                formatter,
                "Boolean-KV schema version and generation must be supplied together"
            ),
            Self::Zhegalkin(error) => write!(formatter, "Zhegalkin cooperation failed: {error}"),
        }
    }
}

impl std::error::Error for CooperationError {}

impl From<ZhegalkinError> for CooperationError {
    fn from(error: ZhegalkinError) -> Self {
        Self::Zhegalkin(error)
    }
}

pub fn route_attention_needs(
    needs: AttentionAlgebraNeeds,
    boolean_evidence: Option<M13bBooleanRoutingEvidence>,
) -> Result<AlgebraicRoute, CooperationError> {
    if needs.eligibility_logic && boolean_evidence.is_none() {
        return Err(CooperationError::MissingBooleanRoutingEvidence);
    }
    if !needs.eligibility_logic && boolean_evidence.is_some() {
        return Err(CooperationError::UnexpectedBooleanRoutingEvidence);
    }

    let mut domains = Vec::with_capacity(4);
    if needs.eligibility_logic {
        domains.push(AlgebraDomain::Boolean);
    }
    if needs.parity_or_binary_linear {
        domains.push(AlgebraDomain::F2);
    }
    if needs.nonlinear_boolean_interaction {
        domains.push(AlgebraDomain::Zhegalkin);
    }
    if needs.precedence_or_critical_path {
        domains.push(AlgebraDomain::MaxPlus);
    }

    if domains.is_empty() {
        return Err(CooperationError::NoRequestedSemantics);
    }

    Ok(AlgebraicRoute {
        domains,
        boolean_evidence,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveBooleanMaxPlus {
    evidence: AlgebraicEvidence,
    value: MaxPlusValue,
}

impl PositiveBooleanMaxPlus {
    #[must_use]
    pub const fn evidence(&self) -> AlgebraicEvidence {
        self.evidence
    }

    #[must_use]
    pub const fn value(&self) -> MaxPlusValue {
        self.value
    }
}

#[must_use]
pub const fn positive_boolean_to_max_plus(value: bool) -> PositiveBooleanMaxPlus {
    PositiveBooleanMaxPlus {
        evidence: AlgebraicEvidence {
            source: AlgebraDomain::Boolean,
            target: CooperationTarget::Single(AlgebraDomain::MaxPlus),
            quality: ConversionQuality::Restricted,
            restriction: Some(BridgeRestriction::PositiveBooleanSemiring),
        },
        value: MaxPlusValue::from_positive_boolean(value),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZhegalkinF2Split {
    evidence: AlgebraicEvidence,
    affine: F2AffinePredicate,
    nonlinear: ZhegalkinPolynomial,
}

impl ZhegalkinF2Split {
    #[must_use]
    pub const fn evidence(&self) -> AlgebraicEvidence {
        self.evidence
    }

    #[must_use]
    pub fn affine(&self) -> &F2AffinePredicate {
        &self.affine
    }

    #[must_use]
    pub fn nonlinear(&self) -> &ZhegalkinPolynomial {
        &self.nonlinear
    }
}

pub fn split_zhegalkin_for_f2(
    polynomial: &ZhegalkinPolynomial,
) -> Result<ZhegalkinF2Split, CooperationError> {
    let decomposition = polynomial.affine_decomposition()?;
    Ok(ZhegalkinF2Split {
        evidence: AlgebraicEvidence {
            source: AlgebraDomain::Zhegalkin,
            target: CooperationTarget::Split {
                primary: AlgebraDomain::F2,
                residual: AlgebraDomain::Zhegalkin,
            },
            quality: ConversionQuality::Exact,
            restriction: None,
        },
        affine: decomposition.affine().clone(),
        nonlinear: decomposition.nonlinear().clone(),
    })
}

#[must_use]
pub const fn supports_direct_single_domain_bridge(
    source: AlgebraDomain,
    target: AlgebraDomain,
) -> bool {
    matches!(
        (source, target),
        (AlgebraDomain::Boolean, AlgebraDomain::MaxPlus)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2::F2Vector;

    fn boolean_evidence() -> M13bBooleanRoutingEvidence {
        M13bBooleanRoutingEvidence::new(
            1,
            4,
            vec![0b1101],
            1,
            4,
            vec![0b1011],
            vec![0b1001],
            Some(1),
            Some(7),
        )
        .unwrap()
    }

    #[test]
    fn route_can_select_all_engines_without_implying_execution_order() {
        let route = route_attention_needs(
            AttentionAlgebraNeeds {
                eligibility_logic: true,
                parity_or_binary_linear: true,
                nonlinear_boolean_interaction: true,
                precedence_or_critical_path: true,
            },
            Some(boolean_evidence()),
        )
        .unwrap();

        assert_eq!(
            route.domains(),
            &[
                AlgebraDomain::Boolean,
                AlgebraDomain::F2,
                AlgebraDomain::Zhegalkin,
                AlgebraDomain::MaxPlus,
            ]
        );
        for domain in [
            AlgebraDomain::Boolean,
            AlgebraDomain::F2,
            AlgebraDomain::Zhegalkin,
            AlgebraDomain::MaxPlus,
        ] {
            assert!(route.contains(domain));
        }
        let evidence = route.boolean_evidence().unwrap();
        assert_eq!(evidence.blocks(), 4);
        assert_eq!(evidence.mask_words(), &[0b1101]);
        assert_eq!(evidence.signature_bits(), 4);
        assert_eq!(evidence.boolean_kv_generation(), Some(7));
    }

    #[test]
    fn boolean_route_without_m13b_evidence_fails_closed() {
        assert_eq!(
            route_attention_needs(
                AttentionAlgebraNeeds {
                    eligibility_logic: true,
                    ..AttentionAlgebraNeeds::default()
                },
                None,
            ),
            Err(CooperationError::MissingBooleanRoutingEvidence)
        );
    }

    #[test]
    fn unrelated_boolean_evidence_is_rejected() {
        assert_eq!(
            route_attention_needs(
                AttentionAlgebraNeeds {
                    parity_or_binary_linear: true,
                    ..AttentionAlgebraNeeds::default()
                },
                Some(boolean_evidence()),
            ),
            Err(CooperationError::UnexpectedBooleanRoutingEvidence)
        );
    }

    #[test]
    fn malformed_m13b_evidence_fails_closed() {
        assert!(matches!(
            M13bBooleanRoutingEvidence::new(1, 65, vec![1], 1, 4, vec![1], vec![1], None, None),
            Err(CooperationError::M13bWordCountMismatch { .. })
        ));
        assert!(matches!(
            M13bBooleanRoutingEvidence::new(1, 4, vec![0b1_0000], 1, 4, vec![1], vec![1], None, None),
            Err(CooperationError::M13bNonZeroTailBits { .. })
        ));
        assert_eq!(
            M13bBooleanRoutingEvidence::new(1, 4, vec![1], 1, 4, vec![1], vec![1], Some(1), None),
            Err(CooperationError::IncompleteBooleanKvProvenance)
        );
    }

    #[test]
    fn empty_semantic_request_fails_closed() {
        assert_eq!(
            route_attention_needs(AttentionAlgebraNeeds::default(), None),
            Err(CooperationError::NoRequestedSemantics)
        );
    }

    #[test]
    fn positive_boolean_bridge_is_explicitly_restricted() {
        let false_bridge = positive_boolean_to_max_plus(false);
        let true_bridge = positive_boolean_to_max_plus(true);

        assert_eq!(false_bridge.value(), MaxPlusValue::ZERO);
        assert_eq!(true_bridge.value(), MaxPlusValue::ONE);
        assert_eq!(
            true_bridge.evidence(),
            AlgebraicEvidence {
                source: AlgebraDomain::Boolean,
                target: CooperationTarget::Single(AlgebraDomain::MaxPlus),
                quality: ConversionQuality::Restricted,
                restriction: Some(BridgeRestriction::PositiveBooleanSemiring),
            }
        );
    }

    #[test]
    fn zhegalkin_split_is_exact_and_keeps_nonlinear_residual() {
        let polynomial = ZhegalkinPolynomial::from_variable_sets(
            3,
            vec![vec![], vec![0], vec![2], vec![0, 1], vec![0, 1, 2]],
        )
        .unwrap();
        let split = split_zhegalkin_for_f2(&polynomial).unwrap();

        assert_eq!(
            split.evidence(),
            AlgebraicEvidence {
                source: AlgebraDomain::Zhegalkin,
                target: CooperationTarget::Split {
                    primary: AlgebraDomain::F2,
                    residual: AlgebraDomain::Zhegalkin,
                },
                quality: ConversionQuality::Exact,
                restriction: None,
            }
        );
        assert_eq!(
            split.affine().coefficients(),
            &F2Vector::from_bools(&[true, false, true]).unwrap()
        );
        assert!(split.affine().bias());
        assert_eq!(split.nonlinear().terms().len(), 2);

        for assignment in 0u8..8 {
            let input = F2Vector::from_bools(&[
                assignment & 0b001 != 0,
                assignment & 0b010 != 0,
                assignment & 0b100 != 0,
            ])
            .unwrap();
            let reconstructed = split.affine().evaluate(&input).unwrap()
                ^ split.nonlinear().evaluate(&input).unwrap();
            assert_eq!(reconstructed, polynomial.evaluate(&input).unwrap());
        }
    }

    #[test]
    fn f2_to_max_plus_is_not_an_implicit_direct_bridge() {
        assert!(!supports_direct_single_domain_bridge(
            AlgebraDomain::F2,
            AlgebraDomain::MaxPlus
        ));
    }

    #[test]
    fn zhegalkin_to_f2_is_a_split_not_a_direct_single_domain_bridge() {
        assert!(!supports_direct_single_domain_bridge(
            AlgebraDomain::Zhegalkin,
            AlgebraDomain::F2
        ));
    }
}
