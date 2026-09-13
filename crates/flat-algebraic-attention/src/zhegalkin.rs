use core::fmt;

use crate::f2::{F2AffinePredicate, F2Error, F2Vector};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZhegalkinError {
    ZeroVariables,
    VariableOutOfBounds {
        variable: usize,
        variable_count: usize,
    },
    InputDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    F2(F2Error),
}

impl fmt::Display for ZhegalkinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVariables => write!(
                formatter,
                "Zhegalkin polynomials require at least one variable"
            ),
            Self::VariableOutOfBounds {
                variable,
                variable_count,
            } => write!(
                formatter,
                "Zhegalkin variable x{variable} is outside 0..{variable_count}"
            ),
            Self::InputDimensionMismatch { expected, actual } => write!(
                formatter,
                "Zhegalkin input dimension mismatch: expected {expected} bits, got {actual}"
            ),
            Self::F2(error) => write!(formatter, "F2 bridge failed: {error}"),
        }
    }
}

impl std::error::Error for ZhegalkinError {}

impl From<F2Error> for ZhegalkinError {
    fn from(error: F2Error) -> Self {
        Self::F2(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ZhegalkinMonomial {
    variables: Vec<usize>,
}

impl ZhegalkinMonomial {
    pub fn new(variable_count: usize, mut variables: Vec<usize>) -> Result<Self, ZhegalkinError> {
        require_variable_count(variable_count)?;
        variables.sort_unstable();
        variables.dedup();
        validate_variables(variable_count, &variables)?;
        Ok(Self { variables })
    }

    #[must_use]
    pub fn variables(&self) -> &[usize] {
        &self.variables
    }

    #[must_use]
    pub fn degree(&self) -> usize {
        self.variables.len()
    }

    pub fn evaluate(&self, input: &F2Vector) -> Result<bool, ZhegalkinError> {
        for &variable in &self.variables {
            if !input.bit(variable)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZhegalkinPolynomial {
    variable_count: usize,
    terms: Vec<ZhegalkinMonomial>,
}

impl ZhegalkinPolynomial {
    pub fn new(
        variable_count: usize,
        mut terms: Vec<ZhegalkinMonomial>,
    ) -> Result<Self, ZhegalkinError> {
        require_variable_count(variable_count)?;
        for term in &terms {
            validate_variables(variable_count, term.variables())?;
        }

        terms.sort_unstable();
        let mut canonical = Vec::with_capacity(terms.len());
        let mut index = 0;
        while index < terms.len() {
            let mut end = index + 1;
            while end < terms.len() && terms[end] == terms[index] {
                end += 1;
            }
            if (end - index) % 2 == 1 {
                canonical.push(terms[index].clone());
            }
            index = end;
        }

        Ok(Self {
            variable_count,
            terms: canonical,
        })
    }

    pub fn from_variable_sets(
        variable_count: usize,
        terms: Vec<Vec<usize>>,
    ) -> Result<Self, ZhegalkinError> {
        let terms = terms
            .into_iter()
            .map(|variables| ZhegalkinMonomial::new(variable_count, variables))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(variable_count, terms)
    }

    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.variable_count
    }

    #[must_use]
    pub fn terms(&self) -> &[ZhegalkinMonomial] {
        &self.terms
    }

    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    #[must_use]
    pub fn algebraic_degree(&self) -> Option<usize> {
        self.terms.iter().map(ZhegalkinMonomial::degree).max()
    }

    pub fn evaluate(&self, input: &F2Vector) -> Result<bool, ZhegalkinError> {
        self.require_input(input)?;
        let mut value = false;
        for term in &self.terms {
            value ^= term.evaluate(input)?;
        }
        Ok(value)
    }

    pub fn affine_decomposition(&self) -> Result<ZhegalkinDecomposition, ZhegalkinError> {
        let mut coefficients = vec![false; self.variable_count];
        let mut bias = false;
        let mut nonlinear_terms = Vec::new();

        for term in &self.terms {
            match term.degree() {
                0 => bias = true,
                1 => coefficients[term.variables()[0]] = true,
                _ => nonlinear_terms.push(term.clone()),
            }
        }

        let affine = F2AffinePredicate::new(F2Vector::from_bools(&coefficients)?, bias);
        let nonlinear = Self::new(self.variable_count, nonlinear_terms)?;
        Ok(ZhegalkinDecomposition { affine, nonlinear })
    }

    fn require_input(&self, input: &F2Vector) -> Result<(), ZhegalkinError> {
        if input.bit_len() != self.variable_count {
            return Err(ZhegalkinError::InputDimensionMismatch {
                expected: self.variable_count,
                actual: input.bit_len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZhegalkinDecomposition {
    affine: F2AffinePredicate,
    nonlinear: ZhegalkinPolynomial,
}

impl ZhegalkinDecomposition {
    #[must_use]
    pub fn affine(&self) -> &F2AffinePredicate {
        &self.affine
    }

    #[must_use]
    pub fn nonlinear(&self) -> &ZhegalkinPolynomial {
        &self.nonlinear
    }

    pub fn evaluate(&self, input: &F2Vector) -> Result<bool, ZhegalkinError> {
        Ok(self.affine.evaluate(input)? ^ self.nonlinear.evaluate(input)?)
    }
}

fn require_variable_count(variable_count: usize) -> Result<(), ZhegalkinError> {
    if variable_count == 0 {
        return Err(ZhegalkinError::ZeroVariables);
    }
    Ok(())
}

fn validate_variables(variable_count: usize, variables: &[usize]) -> Result<(), ZhegalkinError> {
    for &variable in variables {
        if variable >= variable_count {
            return Err(ZhegalkinError::VariableOutOfBounds {
                variable,
                variable_count,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(bits: &[bool]) -> F2Vector {
        F2Vector::from_bools(bits).unwrap()
    }

    #[test]
    fn monomials_are_square_free_and_canonical() {
        let monomial = ZhegalkinMonomial::new(4, vec![3, 1, 3, 1]).unwrap();
        assert_eq!(monomial.variables(), &[1, 3]);
        assert_eq!(monomial.degree(), 2);
    }

    #[test]
    fn duplicate_terms_cancel_modulo_two() {
        let polynomial =
            ZhegalkinPolynomial::from_variable_sets(3, vec![vec![0, 1], vec![1, 0], vec![2]])
                .unwrap();

        assert_eq!(polynomial.terms().len(), 1);
        assert_eq!(polynomial.terms()[0].variables(), &[2]);
    }

    #[test]
    fn or_identity_matches_boolean_truth_table() {
        let polynomial =
            ZhegalkinPolynomial::from_variable_sets(2, vec![vec![0], vec![1], vec![0, 1]]).unwrap();

        for left in [false, true] {
            for right in [false, true] {
                assert_eq!(
                    polynomial.evaluate(&input(&[left, right])).unwrap(),
                    left || right
                );
            }
        }
    }

    #[test]
    fn not_identity_uses_constant_plus_variable() {
        let polynomial = ZhegalkinPolynomial::from_variable_sets(1, vec![vec![], vec![0]]).unwrap();

        assert!(polynomial.evaluate(&input(&[false])).unwrap());
        assert!(!polynomial.evaluate(&input(&[true])).unwrap());
    }

    #[test]
    fn decomposition_preserves_exact_function() {
        let polynomial = ZhegalkinPolynomial::from_variable_sets(
            3,
            vec![vec![], vec![0], vec![2], vec![0, 1], vec![0, 1, 2]],
        )
        .unwrap();
        let decomposition = polynomial.affine_decomposition().unwrap();

        assert!(decomposition.affine().bias());
        assert_eq!(
            decomposition.affine().coefficients(),
            &F2Vector::from_bools(&[true, false, true]).unwrap()
        );
        assert_eq!(decomposition.nonlinear().terms().len(), 2);
        assert_eq!(decomposition.nonlinear().algebraic_degree(), Some(3));

        for assignment in 0u8..8 {
            let vector = input(&[
                assignment & 0b001 != 0,
                assignment & 0b010 != 0,
                assignment & 0b100 != 0,
            ]);
            assert_eq!(
                decomposition.evaluate(&vector).unwrap(),
                polynomial.evaluate(&vector).unwrap()
            );
        }
    }

    #[test]
    fn zero_polynomial_is_valid_and_evaluates_false() {
        let polynomial = ZhegalkinPolynomial::new(2, Vec::new()).unwrap();
        assert!(polynomial.is_zero());
        assert_eq!(polynomial.algebraic_degree(), None);
        assert!(!polynomial.evaluate(&input(&[true, true])).unwrap());
    }

    #[test]
    fn invalid_variable_fails_closed() {
        assert_eq!(
            ZhegalkinPolynomial::from_variable_sets(2, vec![vec![2]]),
            Err(ZhegalkinError::VariableOutOfBounds {
                variable: 2,
                variable_count: 2,
            })
        );
    }

    #[test]
    fn wrong_input_width_fails_closed() {
        let polynomial = ZhegalkinPolynomial::from_variable_sets(2, vec![vec![0]]).unwrap();
        assert_eq!(
            polynomial.evaluate(&input(&[true])),
            Err(ZhegalkinError::InputDimensionMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }
}
