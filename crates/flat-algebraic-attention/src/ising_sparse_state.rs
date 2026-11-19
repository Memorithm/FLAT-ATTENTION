//! Research-only sparse event-driven Ising state with packed spins.
//!
//! The two bitplanes have distinct semantics: eligibility (structural mask)
//! and spin orientation (+1 versus -1). A masked vertex is not a negative spin
//! and cannot influence an eligible vertex. Couplings and biases are fixed-point
//! integers; exact local field updates use i64. No Q/K/V, softmax, GPU kernel,
//! model-quality or speedup claim is implied.
//!
//! E = -sum_i h_i*s_i - sum_(i<j) J_ij*s_i*s_j, restricted to eligible spins.
//! Flipping i costs delta E = 2*s_i*(h_i + sum_j J_ij*s_j).
//! Neighbor fields change by J_ij*(new_s_i-old_s_i).
use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseIsingEdge {
    pub left: usize,
    pub right: usize,
    pub coupling: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseIsingError {
    EmptyGraph,
    DimensionMismatch,
    InvalidPaddingBits,
    SpinOutsideEligibility,
    NoncanonicalEdge { index: usize },
    TooManyVertices,
    TooManyEdges,
    IndexOutOfBounds,
}
impl fmt::Display for SparseIsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SparseIsingError {}

/// Packed-state Ising oracle for sparse fixed-point, *undirected* coupling graphs.
///
/// Each allowed vertex has spin +1 when its state bit is set and -1 otherwise.
/// A vertex excluded by the eligibility bitplane has no spin and no interaction.
/// The graph uses CSR with each undirected edge stored twice in adjacency. This
/// is a deterministic correctness oracle, not yet a minimized memory layout.
#[derive(Debug, Clone)]
pub struct SparseIsingState {
    eligible: Vec<u64>,
    positive_spins: Vec<u64>,
    biases: Vec<i32>,
    local_fields: Vec<i64>,
    offsets: Vec<usize>,
    adjacency: Vec<(usize, i16)>,
}

impl SparseIsingState {
    pub fn new(
        biases: &[i32],
        eligibility: &[u64],
        positive_spins: &[u64],
        edges: &[SparseIsingEdge],
    ) -> Result<Self, SparseIsingError> {
        let n = biases.len();
        if n == 0 {
            return Err(SparseIsingError::EmptyGraph);
        }
        if n > u32::MAX as usize {
            return Err(SparseIsingError::TooManyVertices);
        }
        let words = n.div_ceil(64);
        if eligibility.len() != words || positive_spins.len() != words {
            return Err(SparseIsingError::DimensionMismatch);
        }
        let last_bits = n % 64;
        if last_bits != 0 {
            let padding_mask = !((1_u64 << last_bits) - 1);
            if eligibility[words - 1] & padding_mask != 0
                || positive_spins[words - 1] & padding_mask != 0
            {
                return Err(SparseIsingError::InvalidPaddingBits);
            }
        }
        for (&allowed, &positive) in eligibility.iter().zip(positive_spins) {
            if positive & !allowed != 0 {
                return Err(SparseIsingError::SpinOutsideEligibility);
            }
        }
        let mut previous: Option<(usize, usize)> = None;
        let mut degrees = vec![0usize; n];
        for (index, edge) in edges.iter().enumerate() {
            let pair = (edge.left, edge.right);
            if edge.left >= edge.right || edge.right >= n || previous.is_some_and(|old| pair <= old)
            {
                return Err(SparseIsingError::NoncanonicalEdge { index });
            }
            previous = Some(pair);
            degrees[edge.left] = degrees[edge.left]
                .checked_add(1)
                .ok_or(SparseIsingError::TooManyEdges)?;
            degrees[edge.right] = degrees[edge.right]
                .checked_add(1)
                .ok_or(SparseIsingError::TooManyEdges)?;
        }
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0usize);
        for degree in degrees {
            offsets.push(
                offsets
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(degree)
                    .ok_or(SparseIsingError::TooManyEdges)?,
            );
        }
        let mut adjacency = vec![(0usize, 0i16); offsets[n]];
        let mut cursor = offsets[..n].to_vec();
        for edge in edges {
            adjacency[cursor[edge.left]] = (edge.right, edge.coupling);
            cursor[edge.left] += 1;
            adjacency[cursor[edge.right]] = (edge.left, edge.coupling);
            cursor[edge.right] += 1;
        }
        let mut output = Self {
            eligible: eligibility.to_vec(),
            positive_spins: positive_spins.to_vec(),
            biases: biases.to_vec(),
            local_fields: biases.iter().map(|&h| i64::from(h)).collect(),
            offsets,
            adjacency,
        };
        for i in 0..n {
            if output.is_eligible(i) {
                for &(j, weight) in &output.adjacency[output.offsets[i]..output.offsets[i + 1]] {
                    if output.is_eligible(j) {
                        output.local_fields[i] +=
                            i64::from(weight) * i64::from(output.spin_sign(j));
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn vertex_count(&self) -> usize {
        self.biases.len()
    }

    /// Returns None when the vertex is excluded by the structural mask.
    pub fn spin(&self, i: usize) -> Result<Option<i8>, SparseIsingError> {
        if i >= self.vertex_count() {
            return Err(SparseIsingError::IndexOutOfBounds);
        }
        Ok(self.is_eligible(i).then(|| self.spin_sign(i)))
    }

    pub fn field(&self, i: usize) -> Result<Option<i64>, SparseIsingError> {
        if i >= self.vertex_count() {
            return Err(SparseIsingError::IndexOutOfBounds);
        }
        Ok(self.is_eligible(i).then_some(self.local_fields[i]))
    }

    /// Exact integer delta E if vertex i flips; masked vertices return None.
    pub fn flip_energy_delta(&self, i: usize) -> Result<Option<i128>, SparseIsingError> {
        Ok(self
            .spin(i)?
            .map(|s| 2 * i128::from(s) * i128::from(self.local_fields[i])))
    }

    /// Flip only if energy decreases strictly. Equal-energy proposals do not flip.
    /// Cost is proportional to the degree of the flipped vertex.
    pub fn flip_if_improves(&mut self, i: usize) -> Result<bool, SparseIsingError> {
        let Some(delta) = self.flip_energy_delta(i)? else {
            return Ok(false);
        };
        if delta >= 0 {
            return Ok(false);
        }
        let old_sign = i64::from(self.spin_sign(i));
        self.positive_spins[i / 64] ^= 1_u64 << (i % 64);
        for &(j, weight) in &self.adjacency[self.offsets[i]..self.offsets[i + 1]] {
            if self.is_eligible(j) {
                self.local_fields[j] -= 2 * old_sign * i64::from(weight);
            }
        }
        Ok(true)
    }

    /// Deterministic ascending-index sweep. This guarantees only local
    /// non-increasing energy, not convergence to the global Ising optimum.
    pub fn descent_sweep(&mut self) -> usize {
        let mut flips = 0;
        for i in 0..self.vertex_count() {
            if self.flip_if_improves(i).expect("valid internal index") {
                flips += 1;
            }
        }
        flips
    }

    /// Exact integer reference energy, independently recomputed from CSR.
    pub fn energy(&self) -> i128 {
        let mut e = 0i128;
        for i in 0..self.vertex_count() {
            if !self.is_eligible(i) {
                continue;
            }
            let s = i128::from(self.spin_sign(i));
            e -= i128::from(self.biases[i]) * s;
            for &(j, weight) in &self.adjacency[self.offsets[i]..self.offsets[i + 1]] {
                if j > i && self.is_eligible(j) {
                    e -= i128::from(weight) * s * i128::from(self.spin_sign(j));
                }
            }
        }
        e
    }

    fn is_eligible(&self, i: usize) -> bool {
        self.eligible[i / 64] & (1_u64 << (i % 64)) != 0
    }

    fn spin_sign(&self, i: usize) -> i8 {
        if self.positive_spins[i / 64] & (1_u64 << (i % 64)) != 0 {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_flip_preserves_exact_energy_delta_and_neighbor_fields() {
        let edges = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: 3,
        }];
        let mut state = SparseIsingState::new(&[0, 0], &[0b11], &[0b01], &edges).unwrap();
        assert_eq!(state.energy(), 3);
        assert_eq!(state.flip_energy_delta(0).unwrap(), Some(-6));
        assert!(state.flip_if_improves(0).unwrap());
        assert_eq!(state.energy(), -3);
        assert_eq!(state.spin(0).unwrap(), Some(-1));
        assert_eq!(state.field(1).unwrap(), Some(-3));
        assert!(!state.flip_if_improves(0).unwrap());
    }

    #[test]
    fn masked_spin_does_not_apply_couplings() {
        let edge = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: 20,
        }];
        let mut state = SparseIsingState::new(&[1, 0], &[0b01], &[0b01], &edge).unwrap();
        assert_eq!(state.spin(1).unwrap(), None);
        assert_eq!(state.field(0).unwrap(), Some(1));
        assert_eq!(state.energy(), -1);
        assert_eq!(state.descent_sweep(), 0);
        assert_eq!(state.energy(), -1);
    }

    #[test]
    fn descending_sweep_never_increases_energy() {
        let edges = [
            SparseIsingEdge {
                left: 0,
                right: 1,
                coupling: 2,
            },
            SparseIsingEdge {
                left: 1,
                right: 2,
                coupling: -3,
            },
        ];
        let mut state = SparseIsingState::new(&[0, 0, 1], &[0b111], &[0b111], &edges).unwrap();
        let before = state.energy();
        state.descent_sweep();
        assert!(state.energy() <= before);
    }

    #[test]
    fn invalid_masks_padding_and_edges_are_rejected() {
        assert_eq!(
            SparseIsingState::new(&[], &[], &[], &[]).unwrap_err(),
            SparseIsingError::EmptyGraph
        );
        assert_eq!(
            SparseIsingState::new(&[0], &[0b1], &[0b10], &[]).unwrap_err(),
            SparseIsingError::InvalidPaddingBits
        );
        assert_eq!(
            SparseIsingState::new(&[0, 0], &[0b01], &[0b10], &[]).unwrap_err(),
            SparseIsingError::SpinOutsideEligibility
        );
        assert_eq!(
            SparseIsingState::new(
                &[0, 0],
                &[0b11],
                &[0],
                &[SparseIsingEdge {
                    left: 1,
                    right: 0,
                    coupling: 1
                }]
            )
            .unwrap_err(),
            SparseIsingError::NoncanonicalEdge { index: 0 }
        );
    }
}
