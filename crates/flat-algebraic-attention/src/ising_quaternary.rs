//! Quaternary Boolean control for a sparse Ising-like activation network.
//!
//! Two independent bits encode activity and remembered polarity:
//! 00 dormant-negative, 01 dormant-positive, 10 active-negative, 11 active-positive.
//! Eligibility uses a separate Boolean bitplane. Masked sites never contribute.
//!
//! z_i = active_i * sign_i in {-1, 0, +1}.
//! E = sum_i cost_i * active_i - sum_i bias_i * z_i
//!     - sum_(i<j) J_ij * z_i * z_j.
//!
//! Dormant polarity is remembered without contributing to the energy. This is
//! a purpose-built four-state controller, not the Ashkin-Teller Hamiltonian.
//! All updates strictly lower integer energy; no learning or performance claim.

use core::fmt;

use crate::ising_sparse_state::SparseIsingEdge;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spin4 {
    DormantNegative = 0,
    DormantPositive = 1,
    ActiveNegative = 2,
    ActivePositive = 3,
}

impl Spin4 {
    pub const ALL: [Self; 4] = [
        Self::DormantNegative,
        Self::DormantPositive,
        Self::ActiveNegative,
        Self::ActivePositive,
    ];

    pub const fn active(self) -> bool {
        (self as u8) & 2 != 0
    }

    pub const fn positive(self) -> bool {
        (self as u8) & 1 != 0
    }

    pub const fn contribution(self) -> i64 {
        if !self.active() {
            0
        } else if self.positive() {
            1
        } else {
            -1
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Spin4Error {
    Empty,
    Shape,
    Padding,
    MaskViolation,
    NoncanonicalEdge(usize),
    Overflow,
    Index,
}
impl fmt::Display for Spin4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Spin4Error {}

/// Deterministic reference with three bitplanes and a sparse CSR graph.
///
/// Stored state costs three logical bits per vertex, plus independent i32
/// biases and activity penalties, i64 fields, adjacency and CSR offsets.
/// This layout is not a claim about total allocated bytes or throughput.
#[derive(Debug, Clone)]
pub struct BooleanQuaternary {
    n: usize,
    eligible: Vec<u64>,
    active: Vec<u64>,
    positive: Vec<u64>,
    biases: Vec<i32>,
    costs: Vec<i32>,
    fields: Vec<i64>,
    offsets: Vec<usize>,
    adjacency: Vec<(usize, i16)>,
}

impl BooleanQuaternary {
    pub fn new(
        biases: &[i32],
        costs: &[i32],
        eligible: &[u64],
        active: &[u64],
        positive: &[u64],
        edges: &[SparseIsingEdge],
    ) -> Result<Self, Spin4Error> {
        let n = biases.len();
        if n == 0 {
            return Err(Spin4Error::Empty);
        }
        if n > u32::MAX as usize {
            return Err(Spin4Error::Overflow);
        }
        let words = n.div_ceil(64);
        if costs.len() != n
            || eligible.len() != words
            || active.len() != words
            || positive.len() != words
        {
            return Err(Spin4Error::Shape);
        }
        if n % 64 != 0 {
            let padding = !((1u64 << (n % 64)) - 1);
            if (eligible[words - 1] | active[words - 1] | positive[words - 1]) & padding != 0 {
                return Err(Spin4Error::Padding);
            }
        }
        for (&a, &e) in active.iter().zip(eligible) {
            if a & !e != 0 {
                return Err(Spin4Error::MaskViolation);
            }
        }
        let mut degrees = vec![0usize; n];
        let mut previous = None;
        for (index, edge) in edges.iter().enumerate() {
            let pair = (edge.left, edge.right);
            if edge.left >= edge.right || edge.right >= n || previous.is_some_and(|p| pair <= p) {
                return Err(Spin4Error::NoncanonicalEdge(index));
            }
            previous = Some(pair);
            degrees[edge.left] = degrees[edge.left]
                .checked_add(1)
                .ok_or(Spin4Error::Overflow)?;
            degrees[edge.right] = degrees[edge.right]
                .checked_add(1)
                .ok_or(Spin4Error::Overflow)?;
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
                    .ok_or(Spin4Error::Overflow)?,
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
        let mut result = Self {
            n,
            eligible: eligible.to_vec(),
            active: active.to_vec(),
            positive: positive.to_vec(),
            biases: biases.to_vec(),
            costs: costs.to_vec(),
            fields: biases.iter().copied().map(i64::from).collect(),
            offsets,
            adjacency,
        };
        for i in 0..n {
            if result.allowed(i) {
                for &(j, weight) in &result.adjacency[result.offsets[i]..result.offsets[i + 1]] {
                    result.fields[i] += i64::from(weight) * result.z(j);
                }
            }
        }
        Ok(result)
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Borrow all three frozen state planes for topology and temporal observers.
    ///
    /// The returned order is (eligibility, activity, polarity). This is a
    /// read-only view, and does not expose the cached energetic fields.
    pub fn bitplanes(&self) -> (&[u64], &[u64], &[u64]) {
        (&self.eligible, &self.active, &self.positive)
    }

    /// None means structurally excluded, not a negative spin.
    pub fn state(&self, i: usize) -> Result<Option<Spin4>, Spin4Error> {
        if i >= self.n {
            return Err(Spin4Error::Index);
        }
        if !self.allowed(i) {
            return Ok(None);
        }
        let state = match (self.test(&self.active, i), self.test(&self.positive, i)) {
            (false, false) => Spin4::DormantNegative,
            (false, true) => Spin4::DormantPositive,
            (true, false) => Spin4::ActiveNegative,
            (true, true) => Spin4::ActivePositive,
        };
        Ok(Some(state))
    }

    pub fn field(&self, i: usize) -> Result<Option<i64>, Spin4Error> {
        if i >= self.n {
            return Err(Spin4Error::Index);
        }
        Ok(self.allowed(i).then_some(self.fields[i]))
    }

    /// Exact local energy difference without changing the state.
    pub fn delta(&self, i: usize, next: Spin4) -> Result<Option<i128>, Spin4Error> {
        let Some(previous) = self.state(i)? else {
            return Ok(None);
        };
        let da = i128::from(next.active() as i8) - i128::from(previous.active() as i8);
        let dz = i128::from(next.contribution() - previous.contribution());
        Ok(Some(
            i128::from(self.costs[i]) * da - i128::from(self.fields[i]) * dz,
        ))
    }

    /// Strict energy descent. Does not erase a latent polarity on deactivation.
    pub fn transition(&mut self, i: usize, next: Spin4) -> Result<bool, Spin4Error> {
        let Some(delta) = self.delta(i, next)? else {
            return Ok(false);
        };
        if delta >= 0 {
            return Ok(false);
        }
        let old = self.state(i)?.expect("previously eligible");
        let dz = next.contribution() - old.contribution();
        let bit = 1u64 << (i % 64);
        if next.active() {
            self.active[i / 64] |= bit;
        } else {
            self.active[i / 64] &= !bit;
        }
        if next.positive() {
            self.positive[i / 64] |= bit;
        } else {
            self.positive[i / 64] &= !bit;
        }
        if dz != 0 {
            for &(j, weight) in &self.adjacency[self.offsets[i]..self.offsets[i + 1]] {
                if self.allowed(j) {
                    self.fields[j] += i64::from(weight) * dz;
                }
            }
        }
        Ok(true)
    }

    /// Greedy local candidate search with a deterministic tie rule.
    pub fn update_one(&mut self, i: usize) -> Result<bool, Spin4Error> {
        let Some(old) = self.state(i)? else {
            return Ok(false);
        };
        let mut best = old;
        let mut best_delta = 0i128;
        for candidate in Spin4::ALL {
            let delta = self.delta(i, candidate)?.expect("eligible vertex");
            if delta < best_delta
                || (delta == best_delta
                    && delta < 0
                    && candidate.positive() == old.positive()
                    && best.positive() != old.positive())
            {
                best = candidate;
                best_delta = delta;
            }
        }
        self.transition(i, best)
    }

    pub fn sweep(&mut self) -> usize {
        let mut changes = 0;
        for i in 0..self.n {
            if self.update_one(i).expect("internal vertex index is valid") {
                changes += 1;
            }
        }
        changes
    }

    /// Independent full reference energy; integer arithmetic is exact.
    pub fn energy(&self) -> i128 {
        let mut energy = 0i128;
        for i in 0..self.n {
            if !self.allowed(i) {
                continue;
            }
            let z = i128::from(self.z(i));
            if z != 0 {
                energy += i128::from(self.costs[i]);
            }
            energy -= i128::from(self.biases[i]) * z;
            for &(j, weight) in &self.adjacency[self.offsets[i]..self.offsets[i + 1]] {
                if j > i {
                    energy -= i128::from(weight) * z * i128::from(self.z(j));
                }
            }
        }
        energy
    }

    fn test(&self, bits: &[u64], i: usize) -> bool {
        (bits[i / 64] >> (i % 64)) & 1 != 0
    }

    fn allowed(&self, i: usize) -> bool {
        self.test(&self.eligible, i)
    }

    fn z(&self, i: usize) -> i64 {
        if !self.allowed(i) || !self.test(&self.active, i) {
            0
        } else if self.test(&self.positive, i) {
            1
        } else {
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(
        states: &[Spin4],
        biases: &[i32],
        costs: &[i32],
        edges: &[SparseIsingEdge],
    ) -> BooleanQuaternary {
        let mut a = 0u64;
        let mut p = 0u64;
        for (i, &state) in states.iter().enumerate() {
            a |= u64::from(state.active()) << i;
            p |= u64::from(state.positive()) << i;
        }
        BooleanQuaternary::new(
            biases,
            costs,
            &[(1u64 << states.len()) - 1],
            &[a],
            &[p],
            edges,
        )
        .unwrap()
    }

    #[test]
    fn four_codes_are_distinct_and_roundtrip() {
        let s = Spin4::ALL;
        let m = build(&s, &[0; 4], &[0; 4], &[]);
        for (i, &state) in s.iter().enumerate() {
            assert_eq!(m.state(i).unwrap(), Some(state));
        }
        assert_eq!(m.energy(), 0);
    }

    #[test]
    fn latent_polarity_is_preserved() {
        let mut m = build(&[Spin4::ActivePositive], &[-2], &[0], &[]);
        assert_eq!(m.energy(), 2);
        assert_eq!(m.delta(0, Spin4::DormantPositive).unwrap(), Some(-2));
        assert!(m.transition(0, Spin4::DormantPositive).unwrap());
        assert_eq!(m.state(0).unwrap(), Some(Spin4::DormantPositive));
        assert!(!m.transition(0, Spin4::DormantNegative).unwrap());
        assert_eq!(m.energy(), 0);
    }

    #[test]
    fn all_local_deltas_equal_recomputed_energy() {
        let edges = [
            SparseIsingEdge {
                left: 0,
                right: 1,
                coupling: 5,
            },
            SparseIsingEdge {
                left: 0,
                right: 2,
                coupling: -3,
            },
            SparseIsingEdge {
                left: 1,
                right: 2,
                coupling: 2,
            },
        ];
        for a in Spin4::ALL {
            for b in Spin4::ALL {
                for c in Spin4::ALL {
                    let states = [a, b, c];
                    let m = build(&states, &[2, -1, 3], &[1, 2, -1], &edges);
                    for i in 0..3 {
                        for target in Spin4::ALL {
                            let mut changed = states;
                            changed[i] = target;
                            let other = build(&changed, &[2, -1, 3], &[1, 2, -1], &edges);
                            assert_eq!(
                                m.delta(i, target).unwrap(),
                                Some(other.energy() - m.energy())
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_fields_and_mask_are_correct() {
        let edges = [SparseIsingEdge {
            left: 0,
            right: 1,
            coupling: 3,
        }];
        let mut m = build(
            &[Spin4::ActivePositive, Spin4::ActiveNegative],
            &[0, 0],
            &[0, 0],
            &edges,
        );
        assert_eq!(m.energy(), 3);
        assert!(m.transition(0, Spin4::ActiveNegative).unwrap());
        assert_eq!(m.field(1).unwrap(), Some(-3));
        assert_eq!(m.energy(), -3);
        let masked =
            BooleanQuaternary::new(&[1, 0], &[0, 0], &[0b01], &[0b01], &[0b11], &edges).unwrap();
        assert_eq!(masked.state(1).unwrap(), None);
        assert_eq!(masked.field(0).unwrap(), Some(1));
    }

    #[test]
    fn rejects_invalid_bitplanes() {
        assert_eq!(
            BooleanQuaternary::new(&[], &[], &[], &[], &[], &[]).unwrap_err(),
            Spin4Error::Empty
        );
        assert_eq!(
            BooleanQuaternary::new(&[0], &[0], &[1], &[2], &[0], &[]).unwrap_err(),
            Spin4Error::Padding
        );
        assert_eq!(
            BooleanQuaternary::new(&[0], &[0], &[0], &[1], &[0], &[]).unwrap_err(),
            Spin4Error::MaskViolation
        );
    }
}
