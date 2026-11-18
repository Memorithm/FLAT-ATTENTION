//! Research-only spatial hierarchy for the TDI-22 torsor pairing.
//!
//! The bottom-up torsor reduction is linear and preserves the total wrench
//! (up to floating-point rounding). Replacing several keys by one averaged
//! key in a softmax, however, is an APPROXIMATION. This module does not
//! change FLAT public routing, the TDI f64 oracle, or any GPU path.
//!
//! TDI-22 signs: M(Q) = M(P) + (P - Q) x R.

use crate::tdi22_torsor_projection::{TorsorKey, TorsorQuery};

/// Explicit research-only implementation contract.
pub const TDI22_TORSOR_OCTREE_V1: &str = "tdi22-spatial-octree-f64-reference-v1";
/// Optional, non-production query-specific score-spread gate.
pub const TDI22_TORSOR_OCTREE_SCORE_GUARD_V1: &str = "tdi22-octree-score-spread-guard-v1";
/// Optional absolute componentwise output-error screen in exact arithmetic.
pub const TDI22_TORSOR_OCTREE_OUTPUT_BUDGET_V1: &str = "tdi22-octree-output-budget-v1";
/// Bound construction memory and prohibit accidental unbounded experiments.
pub const MAX_OCTREE_KEYS: usize = 4096;
const MAX_DEPTH: usize = 32;

type Vec3 = [f64; 3];

fn convert(v: [f32; 3]) -> Vec3 {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: Vec3, weight: f64) -> Vec3 {
    [a[0] * weight, a[1] * weight, a[2] * weight]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn finite(v: [f32; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorAggregate {
    /// Weighted arithmetic center of the covered key positions.
    pub center: Vec3,
    /// Sum of key resultants, not a normalized/averaged resultant.
    pub resultant: Vec3,
    /// Sum of transported key moments at center.
    pub moment_at_center: Vec3,
    /// Number of distinct input keys covered by this aggregate.
    pub mass: usize,
}

impl TorsorAggregate {
    /// Move a reduced moment without modifying its resultant.
    pub fn moment_at(&self, point: Vec3) -> Vec3 {
        add(
            self.moment_at_center,
            cross(sub(self.center, point), self.resultant),
        )
    }

    /// R dot M is independent of the reduction point in exact arithmetic.
    pub fn scalar_invariant(&self) -> f64 {
        dot(self.resultant, self.moment_at_center)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TorsorTreeError {
    EmptyBank,
    TooManyKeys { count: usize, limit: usize },
    InvalidLeafCapacity,
    InvalidDepth,
    InvalidOpeningRatio,
    InvalidScoreSpreadLimit,
    InvalidOutputErrorBudget,
    NonFiniteOutputRange,
    InvalidScale,
    NonFiniteKey { index: usize },
    NonFiniteQuery,
    NonFiniteScore,
}

/// Axis-aligned numerical envelopes over the per-key origin-reduced pair (R, C).
///
/// TDI convention: C_i = M_i(P_i) + P_i x R_i, so the score for query
/// (v, omega, Q) is scale * [(v + Q x omega) . R_i + omega . C_i].
/// These envelopes are a conservative algebraic screening mechanism in exact
/// arithmetic; IEEE-754 rounding is not treated as a formal interval proof.
#[derive(Clone, Copy, Debug)]
struct ScoreEnvelope {
    r_min: Vec3,
    r_max: Vec3,
    c_min: Vec3,
    c_max: Vec3,
}

impl ScoreEnvelope {
    fn empty() -> Self {
        Self {
            r_min: [f64::INFINITY; 3],
            r_max: [f64::NEG_INFINITY; 3],
            c_min: [f64::INFINITY; 3],
            c_max: [f64::NEG_INFINITY; 3],
        }
    }

    fn include_key(&mut self, key: TorsorKey) {
        let r = convert(key.resultant);
        let c = add(
            convert(key.moment_at_reference),
            cross(convert(key.reference), r),
        );
        for axis in 0..3 {
            self.r_min[axis] = self.r_min[axis].min(r[axis]);
            self.r_max[axis] = self.r_max[axis].max(r[axis]);
            self.c_min[axis] = self.c_min[axis].min(c[axis]);
            self.c_max[axis] = self.c_max[axis].max(c[axis]);
        }
    }

    fn from_leaf(keys: &[TorsorKey], indices: &[usize]) -> Self {
        let mut envelope = Self::empty();
        for &index in indices {
            envelope.include_key(keys[index]);
        }
        envelope
    }

    fn include_child(&mut self, child: Self) {
        for axis in 0..3 {
            self.r_min[axis] = self.r_min[axis].min(child.r_min[axis]);
            self.r_max[axis] = self.r_max[axis].max(child.r_max[axis]);
            self.c_min[axis] = self.c_min[axis].min(child.c_min[axis]);
            self.c_max[axis] = self.c_max[axis].max(child.c_max[axis]);
        }
    }

    /// Upper estimate on max(score_i) - min(score_i) across the node.
    /// Returns None for non-finite arithmetic so no summary is authorized.
    fn score_span(&self, projected_linear: Vec3, angular: Vec3, scale: f64) -> Option<f64> {
        let mut span = 0.0;
        for axis in 0..3 {
            let r_range = self.r_max[axis] - self.r_min[axis];
            let c_range = self.c_max[axis] - self.c_min[axis];
            span += projected_linear[axis].abs() * r_range + angular[axis].abs() * c_range;
        }
        let bound = scale * span;
        if bound.is_finite() && bound >= 0.0 {
            Some(bound)
        } else {
            None
        }
    }

    /// Global component ranges in query-reduced coordinates, computed only
    /// from the envelopes stored in the root. Not a floating-point interval
    /// certificate: the bound is algebraic in exact arithmetic.
    fn output_ranges(&self, point: Vec3) -> Option<OutputRanges> {
        let mut resultant = [0.0; 3];
        let mut moment = [0.0; 3];
        for (axis, value) in resultant.iter_mut().enumerate() {
            *value = self.r_max[axis] - self.r_min[axis];
        }
        moment[0] = (self.c_max[0] - self.c_min[0])
            + point[1].abs() * resultant[2]
            + point[2].abs() * resultant[1];
        moment[1] = (self.c_max[1] - self.c_min[1])
            + point[2].abs() * resultant[0]
            + point[0].abs() * resultant[2];
        moment[2] = (self.c_max[2] - self.c_min[2])
            + point[0].abs() * resultant[1]
            + point[1].abs() * resultant[0];
        if resultant
            .iter()
            .chain(moment.iter())
            .all(|x| x.is_finite() && *x >= 0.0)
        {
            Some(OutputRanges { resultant, moment })
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OutputRanges {
    resultant: Vec3,
    moment: Vec3,
}

/// Per-coordinate absolute tolerances for the full normalized torsor output.
/// All six tolerances must be finite and nonnegative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorOutputErrorBudget {
    pub resultant_abs: Vec3,
    pub moment_at_query_abs: Vec3,
}

impl TorsorOutputErrorBudget {
    fn valid(self) -> bool {
        self.resultant_abs
            .iter()
            .chain(self.moment_at_query_abs.iter())
            .all(|x| x.is_finite() && *x >= 0.0)
    }

    fn admits(self, score_span: f64, ranges: OutputRanges) -> bool {
        let tv_bound = score_span.tanh();
        (0..3).all(|axis| {
            let r_bound = ranges.resultant[axis] * tv_bound;
            let m_bound = ranges.moment[axis] * tv_bound;
            r_bound.is_finite()
                && m_bound.is_finite()
                && r_bound <= self.resultant_abs[axis]
                && m_bound <= self.moment_at_query_abs[axis]
        })
    }
}

/// Conservative algebraic bounds, valid for exact arithmetic only.
/// Not a directed-rounding IEEE-754 proof and not a model-quality guarantee.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorAnalyticalErrorBounds {
    pub resultant_abs: Vec3,
    pub moment_at_query_abs: Vec3,
    pub log_sum_exp_abs: f64,
    pub max_compressed_score_span: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorBudgetedOutput {
    pub attention: TorsorTreeOutput,
    pub analytical_bounds: TorsorAnalyticalErrorBounds,
}

#[derive(Debug)]
struct Node {
    aggregate: TorsorAggregate,
    bounds_min: Vec3,
    bounds_max: Vec3,
    /// Largest axis-aligned extent of this node's occupied key positions.
    extent: f64,
    score_envelope: ScoreEnvelope,
    children: [Option<usize>; 8],
    /// Only leaves retain individual key indices.
    leaf_keys: Vec<usize>,
}

/// An immutable f64 reference hierarchy built from f32 TDI-22 torsor keys.
/// No implicit causality: every query can see every bank key.
#[derive(Debug)]
pub struct PreparedTorsorOctree {
    keys: Vec<TorsorKey>,
    nodes: Vec<Node>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsorTreeOutput {
    pub resultant: Vec3,
    pub moment_at_query: Vec3,
    pub log_sum_exp: f64,
    /// Number of individual keys scored exactly by the query.
    pub exact_keys: usize,
    /// Number of distant internal nodes scored as a compressed surrogate.
    pub summarized_nodes: usize,
    /// Sum of masses of the exact keys and summarized nodes.
    pub covered_keys: usize,
}

#[derive(Default)]
struct OnlineSoftmax {
    max: f64,
    sum: f64,
    resultant: Vec3,
    moment: Vec3,
    entries: usize,
}

impl OnlineSoftmax {
    fn new() -> Self {
        Self {
            max: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    fn push(&mut self, score: f64, resultant: Vec3, moment: Vec3) {
        let next = self.max.max(score);
        let previous_weight = if self.entries == 0 {
            0.0
        } else {
            (self.max - next).exp()
        };
        let weight = (score - next).exp();
        self.sum = self.sum * previous_weight + weight;
        self.resultant = add(mul(self.resultant, previous_weight), mul(resultant, weight));
        self.moment = add(mul(self.moment, previous_weight), mul(moment, weight));
        self.max = next;
        self.entries += 1;
    }
}

impl PreparedTorsorOctree {
    /// Build the tree using at most 32 levels and `leaf_capacity >= 1`.
    /// Duplicate/degenerate coordinates remain exact leaves rather than
    /// creating arbitrarily deep chains.
    pub fn build(
        keys: &[TorsorKey],
        leaf_capacity: usize,
        max_depth: usize,
    ) -> Result<Self, TorsorTreeError> {
        if keys.is_empty() {
            return Err(TorsorTreeError::EmptyBank);
        }
        if keys.len() > MAX_OCTREE_KEYS {
            return Err(TorsorTreeError::TooManyKeys {
                count: keys.len(),
                limit: MAX_OCTREE_KEYS,
            });
        }
        if leaf_capacity == 0 {
            return Err(TorsorTreeError::InvalidLeafCapacity);
        }
        if max_depth == 0 || max_depth > MAX_DEPTH {
            return Err(TorsorTreeError::InvalidDepth);
        }
        for (index, key) in keys.iter().enumerate() {
            if !finite(key.reference) || !finite(key.resultant) || !finite(key.moment_at_reference)
            {
                return Err(TorsorTreeError::NonFiniteKey { index });
            }
        }
        let mut tree = Self {
            keys: keys.to_vec(),
            nodes: Vec::new(),
        };
        tree.build_node((0..keys.len()).collect(), leaf_capacity, max_depth, 0);
        Ok(tree)
    }

    /// Root's exact linear resultant and moment reduction.
    pub fn root_aggregate(&self) -> TorsorAggregate {
        self.nodes[0].aggregate
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    fn leaf_aggregate(&self, indices: &[usize], center: Vec3) -> TorsorAggregate {
        let mut resultant = [0.0; 3];
        let mut moment = [0.0; 3];
        for &index in indices {
            let key = self.keys[index];
            let r = convert(key.resultant);
            resultant = add(resultant, r);
            moment = add(
                moment,
                add(
                    convert(key.moment_at_reference),
                    cross(sub(convert(key.reference), center), r),
                ),
            );
        }
        TorsorAggregate {
            center,
            resultant,
            moment_at_center: moment,
            mass: indices.len(),
        }
    }

    fn build_node(
        &mut self,
        indices: Vec<usize>,
        leaf_capacity: usize,
        max_depth: usize,
        depth: usize,
    ) -> usize {
        let mut lower = [f64::INFINITY; 3];
        let mut upper = [f64::NEG_INFINITY; 3];
        let mut positions_sum = [0.0; 3];
        for &index in &indices {
            let point = convert(self.keys[index].reference);
            positions_sum = add(positions_sum, point);
            for axis in 0..3 {
                lower[axis] = lower[axis].min(point[axis]);
                upper[axis] = upper[axis].max(point[axis]);
            }
        }
        let mass = indices.len();
        let center = mul(positions_sum, 1.0 / mass as f64);
        let extent = (0..3)
            .map(|axis| upper[axis] - lower[axis])
            .fold(0.0_f64, f64::max);
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            aggregate: TorsorAggregate {
                center,
                resultant: [0.0; 3],
                moment_at_center: [0.0; 3],
                mass,
            },
            bounds_min: lower,
            bounds_max: upper,
            extent,
            score_envelope: ScoreEnvelope::empty(),
            children: [None; 8],
            leaf_keys: Vec::new(),
        });
        if mass <= leaf_capacity || depth == max_depth || extent == 0.0 {
            self.nodes[node_index].aggregate = self.leaf_aggregate(&indices, center);
            self.nodes[node_index].score_envelope = ScoreEnvelope::from_leaf(&self.keys, &indices);
            self.nodes[node_index].leaf_keys = indices;
            return node_index;
        }

        let midpoint = mul(add(lower, upper), 0.5);
        let mut buckets: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
        for &index in &indices {
            let point = convert(self.keys[index].reference);
            let mut octant = 0;
            for axis in 0..3 {
                if point[axis] >= midpoint[axis] {
                    octant |= 1 << axis;
                }
            }
            buckets[octant].push(index);
        }
        if buckets.iter().filter(|bucket| !bucket.is_empty()).count() < 2 {
            self.nodes[node_index].aggregate = self.leaf_aggregate(&indices, center);
            self.nodes[node_index].score_envelope = ScoreEnvelope::from_leaf(&self.keys, &indices);
            self.nodes[node_index].leaf_keys = indices;
            return node_index;
        }

        let mut total_r = [0.0; 3];
        let mut total_m = [0.0; 3];
        for (octant, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let child_index = self.build_node(bucket, leaf_capacity, max_depth, depth + 1);
            let child = self.nodes[child_index].aggregate;
            let child_envelope = self.nodes[child_index].score_envelope;
            self.nodes[node_index].children[octant] = Some(child_index);
            self.nodes[node_index]
                .score_envelope
                .include_child(child_envelope);
            total_r = add(total_r, child.resultant);
            total_m = add(
                total_m,
                add(
                    child.moment_at_center,
                    cross(sub(child.center, center), child.resultant),
                ),
            );
        }
        self.nodes[node_index].aggregate.resultant = total_r;
        self.nodes[node_index].aggregate.moment_at_center = total_m;
        node_index
    }

    /// Query a non-causal bank. `theta = 0` opens every internal node and
    /// computes all individual scores. Positive theta permits spatially distant
    /// node surrogates, scoring the *mean* torsor with a log(mass) correction.
    ///
    /// The reduction itself is exact in real arithmetic; this attention output
    /// is approximate whenever a node compresses nonidentical key scores.
    pub fn forward(
        &self,
        query: TorsorQuery,
        scale: f64,
        theta: f64,
    ) -> Result<TorsorTreeOutput, TorsorTreeError> {
        self.forward_internal(query, scale, theta, None, None)
            .map(|(output, _)| output)
    }

    /// Optional semantic score-spread guard on top of the spatial opening test.
    ///
    /// A summary is admitted only if a per-node O(1) componentwise envelope
    /// bounds the spread of scaled key logits by `max_score_spread`.
    /// In exact arithmetic, this also bounds the global LSE discrepancy
    /// caused by mass-aware mean-logit summaries by that amount.
    /// This does NOT bound resultant/moment output errors or model quality,
    /// and it is not a directed-rounding interval certificate.
    ///
    /// No public runtime routing change; causal masking is not supported.
    pub fn forward_with_score_spread_guard(
        &self,
        query: TorsorQuery,
        scale: f64,
        theta: f64,
        max_score_spread: f64,
    ) -> Result<TorsorTreeOutput, TorsorTreeError> {
        if !max_score_spread.is_finite() || max_score_spread < 0.0 {
            return Err(TorsorTreeError::InvalidScoreSpreadLimit);
        }
        self.forward_internal(query, scale, theta, Some(max_score_spread), None)
            .map(|(output, _)| output)
    }

    /// Require both a score-spread limit and six normalized output-error
    /// budgets before summarizing any distant internal node.
    ///
    /// In exact arithmetic, if all admitted key logits differ from their
    /// node mean by at most S, the exact-versus-surrogate softmax TV distance
    /// is at most tanh(S). Multiplication by the global query-reduced
    /// coordinate range bounds each output-coordinate discrepancy. This is
    /// a conservative analytical screen, NOT a floating-point proof.
    /// No causal masking, model integration, or default runtime promotion.
    pub fn forward_with_output_budgets(
        &self,
        query: TorsorQuery,
        scale: f64,
        theta: f64,
        max_score_spread: f64,
        budget: TorsorOutputErrorBudget,
    ) -> Result<TorsorBudgetedOutput, TorsorTreeError> {
        if !max_score_spread.is_finite() || max_score_spread < 0.0 {
            return Err(TorsorTreeError::InvalidScoreSpreadLimit);
        }
        if !budget.valid() {
            return Err(TorsorTreeError::InvalidOutputErrorBudget);
        }
        if !finite(query.evaluation_point) {
            return Err(TorsorTreeError::NonFiniteQuery);
        }
        let ranges = self.nodes[0]
            .score_envelope
            .output_ranges(convert(query.evaluation_point))
            .ok_or(TorsorTreeError::NonFiniteOutputRange)?;
        let (attention, accepted_span) = self.forward_internal(
            query,
            scale,
            theta,
            Some(max_score_spread),
            Some((budget, ranges)),
        )?;
        let tv_bound = accepted_span.tanh();
        Ok(TorsorBudgetedOutput {
            attention,
            analytical_bounds: TorsorAnalyticalErrorBounds {
                resultant_abs: mul(ranges.resultant, tv_bound),
                moment_at_query_abs: mul(ranges.moment, tv_bound),
                log_sum_exp_abs: accepted_span,
                max_compressed_score_span: accepted_span,
            },
        })
    }

    fn forward_internal(
        &self,
        query: TorsorQuery,
        scale: f64,
        theta: f64,
        max_score_spread: Option<f64>,
        output_guard: Option<(TorsorOutputErrorBudget, OutputRanges)>,
    ) -> Result<(TorsorTreeOutput, f64), TorsorTreeError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TorsorTreeError::InvalidScale);
        }
        if !theta.is_finite() || theta < 0.0 {
            return Err(TorsorTreeError::InvalidOpeningRatio);
        }
        if !finite(query.linear) || !finite(query.angular) || !finite(query.evaluation_point) {
            return Err(TorsorTreeError::NonFiniteQuery);
        }
        let linear = convert(query.linear);
        let angular = convert(query.angular);
        let point = convert(query.evaluation_point);
        let projected_linear = add(linear, cross(point, angular));
        let mut softmax = OnlineSoftmax::new();
        let mut stack = vec![0_usize];
        let mut exact_keys = 0;
        let mut summarized_nodes = 0;
        let mut covered_keys = 0;
        let mut accepted_max_score_span = 0.0_f64;
        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];
            let has_children = node.children.iter().any(Option::is_some);
            let inside = (0..3).all(|axis| {
                point[axis] >= node.bounds_min[axis] && point[axis] <= node.bounds_max[axis]
            });
            let offset = sub(node.aggregate.center, point);
            let distance = dot(offset, offset).sqrt();
            let is_distant = has_children
                && theta > 0.0
                && !inside
                && distance > 0.0
                && node.extent / distance < theta;
            let gated = max_score_spread.is_some() || output_guard.is_some();
            let span = if is_distant && gated {
                node.score_envelope
                    .score_span(projected_linear, angular, scale)
            } else {
                None
            };
            let collapse = is_distant
                && max_score_spread.is_none_or(|limit| span.is_some_and(|value| value <= limit))
                && output_guard.is_none_or(|(budget, ranges)| {
                    span.is_some_and(|value| budget.admits(value, ranges))
                });
            if collapse {
                if let Some(value) = span {
                    accepted_max_score_span = accepted_max_score_span.max(value);
                }
                let aggregate = node.aggregate;
                let inverse_mass = 1.0 / aggregate.mass as f64;
                let resultant = mul(aggregate.resultant, inverse_mass);
                let moment = mul(aggregate.moment_at(point), inverse_mass);
                let score = scale * (dot(linear, resultant) + dot(angular, moment))
                    + (aggregate.mass as f64).ln();
                if !score.is_finite() {
                    return Err(TorsorTreeError::NonFiniteScore);
                }
                softmax.push(score, resultant, moment);
                summarized_nodes += 1;
                covered_keys += aggregate.mass;
            } else if has_children {
                for child in node.children.iter().rev().flatten() {
                    stack.push(*child);
                }
            } else {
                for &key_index in &node.leaf_keys {
                    let key = self.keys[key_index];
                    let resultant = convert(key.resultant);
                    let moment = add(
                        convert(key.moment_at_reference),
                        cross(sub(convert(key.reference), point), resultant),
                    );
                    let score = scale * (dot(linear, resultant) + dot(angular, moment));
                    if !score.is_finite() {
                        return Err(TorsorTreeError::NonFiniteScore);
                    }
                    softmax.push(score, resultant, moment);
                    exact_keys += 1;
                    covered_keys += 1;
                }
            }
        }
        let inverse_sum = 1.0 / softmax.sum;
        let result = TorsorTreeOutput {
            resultant: mul(softmax.resultant, inverse_sum),
            moment_at_query: mul(softmax.moment, inverse_sum),
            log_sum_exp: softmax.max + softmax.sum.ln(),
            exact_keys,
            summarized_nodes,
            covered_keys,
        };
        if !result.log_sum_exp.is_finite()
            || !result.resultant.iter().all(|v| v.is_finite())
            || !result.moment_at_query.iter().all(|v| v.is_finite())
        {
            return Err(TorsorTreeError::NonFiniteScore);
        }
        Ok((result, accepted_max_score_span))
    }

    pub fn forward_batch(
        &self,
        queries: &[TorsorQuery],
        scale: f64,
        theta: f64,
    ) -> Result<Vec<TorsorTreeOutput>, TorsorTreeError> {
        queries
            .iter()
            .copied()
            .map(|query| self.forward(query, scale, theta))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key(i: usize) -> TorsorKey {
        let t = i as f32;
        TorsorKey {
            resultant: [0.4 + t * 0.07, -0.15 + t * 0.02, (0.2 * t).cos()],
            moment_at_reference: [0.1 * t, -0.3, 0.12 * t],
            reference: [t * 0.3, (t * 0.4).sin(), (t * 0.6).cos()],
        }
    }

    fn sample_query(point: [f32; 3]) -> TorsorQuery {
        TorsorQuery {
            linear: [0.8, -0.2, 0.1],
            angular: [0.2, 0.3, -0.4],
            evaluation_point: point,
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1.0e-10 * (1.0 + b.abs()), "{a} != {b}");
    }

    /// Independent per-token softmax does not use the tree, subtree reductions
    /// or stored origin-reduced moments.
    fn independent_dense(keys: &[TorsorKey], query: TorsorQuery, scale: f64) -> (Vec3, Vec3, f64) {
        let mut scored = Vec::new();
        for key in keys {
            let r = convert(key.resultant);
            let delta = sub(convert(key.reference), convert(query.evaluation_point));
            let moment = add(convert(key.moment_at_reference), cross(delta, r));
            let score =
                scale * (dot(convert(query.linear), r) + dot(convert(query.angular), moment));
            scored.push((score, r, moment));
        }
        let largest = scored.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
        let normalizer: f64 = scored.iter().map(|x| (x.0 - largest).exp()).sum();
        let mut out_r = [0.0; 3];
        let mut out_m = [0.0; 3];
        for (score, r, m) in scored {
            let weight = (score - largest).exp() / normalizer;
            out_r = add(out_r, mul(r, weight));
            out_m = add(out_m, mul(m, weight));
        }
        (out_r, out_m, largest + normalizer.ln())
    }

    #[test]
    fn bottom_up_reduction_matches_direct_varignon_and_invariant() {
        let keys: Vec<_> = (0..97).map(sample_key).collect();
        for capacity in [1, 3, 12, 128] {
            let tree = PreparedTorsorOctree::build(&keys, capacity, 20).unwrap();
            assert!(tree.node_count() >= 1);
            assert!(tree.node_count() < 2 * keys.len());
            let root = tree.root_aggregate();
            assert_eq!(root.mass, keys.len());
            let direct = tree.leaf_aggregate(&(0..keys.len()).collect::<Vec<_>>(), root.center);
            for axis in 0..3 {
                close(root.resultant[axis], direct.resultant[axis]);
                close(root.moment_at_center[axis], direct.moment_at_center[axis]);
            }
            let moved = root.moment_at([13.0, -8.0, 0.7]);
            close(root.scalar_invariant(), dot(root.resultant, moved));
        }
    }

    #[test]
    fn theta_zero_matches_independent_dense_all_keys() {
        let keys: Vec<_> = (0..65).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 2, 24).unwrap();
        for point in [[0.0, 0.0, 0.0], [500.0, 0.1, -80.0], [2.0, 3.0, 4.0]] {
            let query = sample_query(point);
            let actual = tree.forward(query, 0.32, 0.0).unwrap();
            let (r, m, lse) = independent_dense(&keys, query, 0.32);
            assert_eq!(actual.exact_keys, keys.len());
            assert_eq!(actual.summarized_nodes, 0);
            assert_eq!(actual.covered_keys, keys.len());
            for axis in 0..3 {
                close(actual.resultant[axis], r[axis]);
                close(actual.moment_at_query[axis], m[axis]);
            }
            close(actual.log_sum_exp, lse);
        }
    }

    #[test]
    fn distant_identical_wrenches_compress_with_mass_and_keep_softmax() {
        // All keys have the same origin-reduced (R,C), despite distinct P.
        let r = [0.3_f32, 0.0, 0.2];
        let c = [0.1_f32, 0.5, -0.2];
        let keys: Vec<_> = (0..128)
            .map(|i| {
                let p = [i as f32 * 0.02, (i % 8) as f32 * 0.1, (i % 4) as f32 * 0.1];
                let pxr = cross(convert(p), convert(r));
                TorsorKey {
                    resultant: r,
                    reference: p,
                    moment_at_reference: [
                        (f64::from(c[0]) - pxr[0]) as f32,
                        (f64::from(c[1]) - pxr[1]) as f32,
                        (f64::from(c[2]) - pxr[2]) as f32,
                    ],
                }
            })
            .collect();
        let tree = PreparedTorsorOctree::build(&keys, 4, 20).unwrap();
        let query = sample_query([10000.0, -1200.0, 230.0]);
        let actual = tree.forward(query, 0.2, 0.5).unwrap();
        let (dense_r, dense_m, dense_lse) = independent_dense(&keys, query, 0.2);
        assert_eq!(actual.exact_keys, 0);
        assert_eq!(actual.summarized_nodes, 1);
        assert_eq!(actual.covered_keys, keys.len());
        for axis in 0..3 {
            // The f32 storage of each M(P) introduces slight individual rounding.
            assert!((actual.resultant[axis] - dense_r[axis]).abs() < 1.0e-9);
            assert!((actual.moment_at_query[axis] - dense_m[axis]).abs() < 2.0e-5);
        }
        assert!((actual.log_sum_exp - dense_lse).abs() < 2.0e-5);
    }

    #[test]
    fn nearby_query_forces_opening_and_every_key_is_covered_once() {
        let keys: Vec<_> = (0..256).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 25).unwrap();
        let query = sample_query(keys[10].reference);
        let row = tree.forward(query, 0.2, 1000.0).unwrap();
        assert!(row.exact_keys > 0);
        assert_eq!(row.covered_keys, keys.len());
        assert!(row.exact_keys + row.summarized_nodes <= keys.len());
        let far = tree
            .forward(sample_query([1.0e5, 1.0e5, 1.0e5]), 0.2, 0.5)
            .unwrap();
        assert_eq!(far.covered_keys, keys.len());
        assert_eq!(far.summarized_nodes, 1);
        assert_eq!(far.exact_keys, 0);
    }

    #[test]
    fn variable_attention_scores_are_approximate_not_dense_equivalent() {
        let keys: Vec<_> = (0..80).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let q = sample_query([1000.0, 1000.0, 1000.0]);
        let exact = tree.forward(q, 0.5, 0.0).unwrap();
        let compressed = tree.forward(q, 0.5, 0.5).unwrap();
        assert!(compressed.summarized_nodes > 0);
        assert_eq!(compressed.covered_keys, keys.len());
        assert!(
            (exact.log_sum_exp - compressed.log_sum_exp).abs() > 1.0e-4,
            "a single mean score is not an exact log-sum-exp"
        );
    }

    #[test]
    fn guarded_adversarial_remote_keys_open_on_large_semantic_spread() {
        let keys: Vec<_> = (0..80).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let q = TorsorQuery {
            linear: [1.0, 0.0, 0.0],
            angular: [0.0; 3],
            evaluation_point: [10000.0, 10000.0, 10000.0],
        };
        let dense = tree.forward(q, 0.2, 0.0).unwrap();
        let spatial_only = tree.forward(q, 0.2, 1.0).unwrap();
        assert_eq!(spatial_only.summarized_nodes, 1);
        let guarded = tree
            .forward_with_score_spread_guard(q, 0.2, 1.0, 0.0)
            .unwrap();
        assert_eq!(guarded.exact_keys, keys.len());
        assert_eq!(guarded.summarized_nodes, 0);
        assert_eq!(guarded.covered_keys, keys.len());
        for axis in 0..3 {
            close(guarded.resultant[axis], dense.resultant[axis]);
            close(guarded.moment_at_query[axis], dense.moment_at_query[axis]);
        }
        close(guarded.log_sum_exp, dense.log_sum_exp);
    }

    #[test]
    fn guarded_lse_remains_within_budget_for_nonuniform_logit_sets() {
        let keys: Vec<_> = (0..80).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let q = TorsorQuery {
            linear: [1.0, -0.25, 0.4],
            angular: [0.0; 3],
            evaluation_point: [1000.0, 2000.0, 3000.0],
        };
        let dense = tree.forward(q, 0.3, 0.0).unwrap();
        for budget in [0.01, 0.05, 0.2, 1.0] {
            let row = tree
                .forward_with_score_spread_guard(q, 0.3, 1.0, budget)
                .unwrap();
            assert_eq!(row.covered_keys, keys.len());
            assert!((row.log_sum_exp - dense.log_sum_exp).abs() <= budget + 1.0e-10);
        }
        let unrestricted = tree.forward(q, 0.3, 1.0).unwrap();
        let loose = tree
            .forward_with_score_spread_guard(q, 0.3, 1.0, 1.0e9)
            .unwrap();
        assert_eq!(unrestricted, loose);
    }

    #[test]
    fn zero_logit_spread_allows_heterogeneous_values_with_exact_mean() {
        let keys: Vec<_> = (0..64)
            .map(|i| TorsorKey {
                reference: [i as f32 * 0.2, 0.0, 0.0],
                resultant: [1.0, i as f32 * 0.5, 0.0],
                moment_at_reference: [0.0, 0.1 * i as f32, 0.0],
            })
            .collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let q = TorsorQuery {
            linear: [1.0, 0.0, 0.0],
            angular: [0.0; 3],
            evaluation_point: [10000.0, 0.0, 0.0],
        };
        let dense = tree.forward(q, 0.4, 0.0).unwrap();
        let row = tree
            .forward_with_score_spread_guard(q, 0.4, 0.5, 0.0)
            .unwrap();
        assert_eq!(row.summarized_nodes, 1);
        assert_eq!(row.covered_keys, keys.len());
        for axis in 0..3 {
            close(row.resultant[axis], dense.resultant[axis]);
            close(row.moment_at_query[axis], dense.moment_at_query[axis]);
        }
        close(row.log_sum_exp, dense.log_sum_exp);
    }

    #[test]
    fn output_budget_reports_componentwise_bounds_against_independent_dense() {
        let keys: Vec<_> = (0..128).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 2, 24).unwrap();
        let budget = TorsorOutputErrorBudget {
            resultant_abs: [1.0e8; 3],
            moment_at_query_abs: [1.0e8; 3],
        };
        let queries = [
            TorsorQuery {
                linear: [1.0, -0.4, 0.7],
                angular: [0.0; 3],
                evaluation_point: [200.0, 400.0, 100.0],
            },
            TorsorQuery {
                linear: [0.8, 0.1, -0.2],
                angular: [0.02, -0.015, 0.01],
                evaluation_point: [500.0, -400.0, 80.0],
            },
            TorsorQuery {
                linear: [0.1, 0.5, -0.3],
                angular: [-0.01, 0.005, 0.02],
                evaluation_point: [-300.0, 250.0, 700.0],
            },
        ];
        for query in queries {
            let bounded = tree
                .forward_with_output_budgets(query, 0.04, 0.5, 1.0e8, budget)
                .unwrap();
            let (dense_r, dense_m, dense_lse) = independent_dense(&keys, query, 0.04);
            assert!(bounded.attention.summarized_nodes > 0);
            assert_eq!(bounded.attention.covered_keys, keys.len());
            let error = bounded.analytical_bounds;
            for axis in 0..3 {
                let r_delta = (bounded.attention.resultant[axis] - dense_r[axis]).abs();
                let m_delta = (bounded.attention.moment_at_query[axis] - dense_m[axis]).abs();
                assert!(
                    r_delta <= error.resultant_abs[axis] + 1.0e-8,
                    "R axis {axis}: {r_delta} > {}",
                    error.resultant_abs[axis]
                );
                assert!(
                    m_delta <= error.moment_at_query_abs[axis] + 1.0e-8,
                    "M axis {axis}: {m_delta} > {}",
                    error.moment_at_query_abs[axis]
                );
                assert!(error.resultant_abs[axis] <= budget.resultant_abs[axis]);
                assert!(error.moment_at_query_abs[axis] <= budget.moment_at_query_abs[axis]);
            }
            assert!(
                (bounded.attention.log_sum_exp - dense_lse).abs() <= error.log_sum_exp_abs + 1.0e-8
            );
        }
    }

    #[test]
    fn tight_component_budgets_expand_remote_semantically_heterogeneous_tree() {
        let keys: Vec<_> = (0..100).map(sample_key).collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let query = TorsorQuery {
            linear: [1.0, 0.0, 0.0],
            angular: [0.0; 3],
            evaluation_point: [5000.0, -2000.0, 700.0],
        };
        let exact = tree.forward(query, 0.2, 0.0).unwrap();
        let score_only = tree
            .forward_with_score_spread_guard(query, 0.2, 1.0, 1.0e6)
            .unwrap();
        assert!(score_only.summarized_nodes > 0);
        let bounded = tree
            .forward_with_output_budgets(
                query,
                0.2,
                1.0,
                1.0e6,
                TorsorOutputErrorBudget {
                    resultant_abs: [0.0; 3],
                    moment_at_query_abs: [0.0; 3],
                },
            )
            .unwrap();
        assert_eq!(bounded.attention.exact_keys, keys.len());
        assert_eq!(bounded.attention.summarized_nodes, 0);
        assert_eq!(bounded.attention, exact);
        assert_eq!(bounded.analytical_bounds.resultant_abs, [0.0; 3]);
        assert_eq!(bounded.analytical_bounds.moment_at_query_abs, [0.0; 3]);
        assert_eq!(bounded.analytical_bounds.log_sum_exp_abs, 0.0);
    }

    #[test]
    fn identical_logits_with_heterogeneous_output_allow_zero_budget_summary() {
        let keys: Vec<_> = (0..64)
            .map(|i| TorsorKey {
                reference: [i as f32, 0.0, 0.0],
                resultant: [1.0, i as f32, 0.0],
                moment_at_reference: [0.0; 3],
            })
            .collect();
        let tree = PreparedTorsorOctree::build(&keys, 1, 20).unwrap();
        let query = TorsorQuery {
            linear: [1.0, 0.0, 0.0],
            angular: [0.0; 3],
            evaluation_point: [1024.0, 0.0, 0.0],
        };
        let bounded = tree
            .forward_with_output_budgets(
                query,
                0.25,
                0.5,
                0.0,
                TorsorOutputErrorBudget {
                    resultant_abs: [0.0; 3],
                    moment_at_query_abs: [0.0; 3],
                },
            )
            .unwrap();
        assert_eq!(bounded.attention.summarized_nodes, 1);
        assert_eq!(bounded.attention.exact_keys, 0);
        let exact = tree.forward(query, 0.25, 0.0).unwrap();
        for axis in 0..3 {
            close(bounded.attention.resultant[axis], exact.resultant[axis]);
            close(
                bounded.attention.moment_at_query[axis],
                exact.moment_at_query[axis],
            );
        }
        close(bounded.attention.log_sum_exp, exact.log_sum_exp);
        assert_eq!(bounded.analytical_bounds.max_compressed_score_span, 0.0);
    }

    #[test]
    fn invalid_six_component_output_budgets_are_rejected() {
        let tree = PreparedTorsorOctree::build(&[sample_key(0)], 1, 2).unwrap();
        let query = sample_query([5.0, 0.0, 0.0]);
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let mut budget = TorsorOutputErrorBudget {
                resultant_abs: [1.0; 3],
                moment_at_query_abs: [1.0; 3],
            };
            budget.resultant_abs[1] = invalid;
            assert!(matches!(
                tree.forward_with_output_budgets(query, 0.3, 0.5, 0.5, budget),
                Err(TorsorTreeError::InvalidOutputErrorBudget)
            ));
            budget.resultant_abs[1] = 1.0;
            budget.moment_at_query_abs[2] = invalid;
            assert!(matches!(
                tree.forward_with_output_budgets(query, 0.3, 0.5, 0.5, budget),
                Err(TorsorTreeError::InvalidOutputErrorBudget)
            ));
        }
        let valid = TorsorOutputErrorBudget {
            resultant_abs: [1.0; 3],
            moment_at_query_abs: [1.0; 3],
        };
        assert!(matches!(
            tree.forward_with_output_budgets(query, 0.3, 0.5, f64::NAN, valid),
            Err(TorsorTreeError::InvalidScoreSpreadLimit)
        ));
    }

    #[test]
    fn invalid_score_spread_budget_fails_closed() {
        let tree = PreparedTorsorOctree::build(&[sample_key(0)], 1, 2).unwrap();
        let q = sample_query([100.0, 0.0, 0.0]);
        for budget in [f64::NAN, f64::INFINITY, -1.0] {
            assert!(matches!(
                tree.forward_with_score_spread_guard(q, 0.3, 0.5, budget),
                Err(TorsorTreeError::InvalidScoreSpreadLimit)
            ));
        }
    }

    #[test]
    fn degenerate_coordinates_and_invalid_inputs_fail_closed() {
        let mut keys = vec![sample_key(0); 100];
        let tree = PreparedTorsorOctree::build(&keys, 1, 5).unwrap();
        assert_eq!(tree.node_count(), 1);
        let row = tree.forward(sample_query([0.0; 3]), 0.5, 1.0).unwrap();
        assert_eq!(row.exact_keys, 100);
        assert_eq!(row.covered_keys, 100);
        assert!(matches!(
            PreparedTorsorOctree::build(&[], 1, 5),
            Err(TorsorTreeError::EmptyBank)
        ));
        assert!(matches!(
            PreparedTorsorOctree::build(&keys, 0, 5),
            Err(TorsorTreeError::InvalidLeafCapacity)
        ));
        assert!(matches!(
            PreparedTorsorOctree::build(&keys, 1, 0),
            Err(TorsorTreeError::InvalidDepth)
        ));
        assert!(matches!(
            tree.forward(sample_query([0.0; 3]), 0.5, -0.5),
            Err(TorsorTreeError::InvalidOpeningRatio)
        ));
        assert!(matches!(
            tree.forward(sample_query([0.0; 3]), 0.0, 0.5),
            Err(TorsorTreeError::InvalidScale)
        ));
        keys[4].resultant[0] = f32::NAN;
        assert!(matches!(
            PreparedTorsorOctree::build(&keys, 1, 4),
            Err(TorsorTreeError::NonFiniteKey { index: 4 })
        ));
    }
}
