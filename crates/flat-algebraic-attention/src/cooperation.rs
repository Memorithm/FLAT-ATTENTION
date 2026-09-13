use core::fmt;

use crate::f2::F2AffinePredicate;
use crate::max_plus::MaxPlusValue;
use crate::zhegalkin::{ZhegalkinError, ZhegalkinPolynomial};

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CooperationError {
    NoRequestedSemantics,
    Zhegalkin(ZhegalkinError),
}

impl fmt::Display for CooperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRequestedSemantics => write!(
                formatter,
                "multi-algebra routing requires at least one requested semantic capability"
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
) -> Result<AlgebraicRoute, CooperationError> {
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

    Ok(AlgebraicRoute { domains })
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
    matches!((source, target), (AlgebraDomain::Boolean, AlgebraDomain::MaxPlus))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f2::F2Vector;

    #[test]
    fn route_can_select_all_engines_without_implying_execution_order() {
        let route = route_attention_needs(AttentionAlgebraNeeds {
            eligibility_logic: true,
            parity_or_binary_linear: true,
            nonlinear_boolean_interaction: true,
            precedence_or_critical_path: true,
        })
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
    }

    #[test]
    fn empty_semantic_request_fails_closed() {
        assert_eq!(
            route_attention_needs(AttentionAlgebraNeeds::default()),
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
