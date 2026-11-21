# Q4-TOPO-4 — Matched signed-cycle control panel (frozen research fixture)

Status: deterministic bounded experiment, Rust CPU oracle only. Neither trained
memory, attention quality, stochastic stability, predictive accuracy, nor any
physical speed, bandwidth or energy advantage is claimed.

## Preregistered intervention and held-fixed variables

The graph has five eligible sites and five canonical undirected edges:

    edge 0: (0,1)      edge 1: (0,3)
    edge 2: (1,2)      edge 3: (2,3)      edge 4: (3,4)

The first four edges constitute the unique four-edge cycle. Edge 4 is a
pendant bridge. All five edge magnitudes are exactly 2 (integer J=+/-2).
Vertex degrees, vertex order, total edges, and all coupling magnitudes are
fixed across the entire panel.

Exhaustively enumerate every way of placing exactly TWO negative signs among
the FIVE edges (C(5,2)=10 assignments):

- Six balanced assignments: both negative couplings are on the four-cycle
  (even negative sign parity). The bridge is positive.
- Four frustrated assignments: one negative sign is on the cycle and one
  on the pendant bridge (odd negative cycle parity).

This use of a bridge makes **negative-coupling counts equal** between both
families while changing the cycle sign invariant. These assignments are NOT
ten independent stochastic draws or a random dataset. The bridge's role
cannot be ignored when interpreting any family difference.

Three preregistered scalar activation costs, applied to all five sites:
- -1: activity intrinsically rewarded
-  0: no local activation penalty
- +2: activity penalized

All five local field biases are zero, eligibility is full, J magnitudes
are unchanged, and there is no parameter optimization or threshold tuning.

## Initialization, attractors and metrics

For each signed assignment, construct the canonical BFS-tree-aligned ACTIVE
configuration rooted at site 0 with positive polarity. Transport edge signs
through tree edges 0,1,2,4 to derive all five signs. Construct the second
initial state by flipping every sign. These paired initializations avoid
conflating a fixed all-positive convention with the gauge of the signed
graph. Descend via the unchanged BooleanQuaternary strict energy minimizer,
in ascending vertex order, with a hard 64-sweep budget. If no fixed point
is reached, return an explicit error instead of silently dropping the case.

For each of the 30 configurations (10 signs x 3 costs):
1. Report both fixed-point endpoints (all five Q4 codes) and their settling
   sweep counts.
2. Compute **exact minimum communication height** and directional Q4
   barriers between the endpoints, exhaustively over 4^5=1024 Q4 states.
3. Where BOTH endpoints are fully active, also calculate a genuinely
   restricted **binary-active** 2^5=32-state minimax barrier. No binary
   value is manufactured for a dormant endpoint. Under comparable
   endpoints, the Q4 state graph includes every binary path, so the Q4
   communication height must be no greater than the binary height.
4. Independently flip each vertex's activity bit and polarity bit, then
   apply the same deterministic strict descent. Measure EXACT stored-bit
   recall and effective z-value recall separately for each of the two
   starting fixed points. Each case has 2 starts x 5 vertices x 2
   perturbation types = 20 trials. Total 600 deterministic trials.
5. Aggregate by structural-balance family and activation cost using **raw
   counts and denominators**, preserving unequal group sizes (6 vs 4
   per cost). Do not report a p-value or probability based on these
   deterministic observations.

In the absence of a task, effective-spin restoration does NOT establish
semantic memory. Exactly how to perturb, and the tie-breaking rule, matter.
Q4 state values 0/1 are dormant with retained latent polarity and 2/3
are active negative/positive.

## Execution: machine-readable reporting

Cases (30 rows):

    cargo run -p flat-algebraic-attention --bin q4_signed_cycle_panel -- --cases

Aggregates (6 rows):

    cargo run -p flat-algebraic-attention --bin q4_signed_cycle_panel -- --summary

Both modes print a deterministic CSV header with a schema column
Q4-TOPO-4-v1. Missing binary controls appear as blank CSV fields,
not zeros. Re-running with the same Rust code/commit must reproduce the
same metrics; the test suite directly checks run equality.

Source:
- crates/flat-algebraic-attention/src/ising_matched_cycle_panel.rs
- crates/flat-algebraic-attention/src/ising_binary_barrier_control.rs
- crates/flat-algebraic-attention/src/bin/q4_signed_cycle_panel.rs

## Qualification, validity and next steps

- Unit tests verify 6/4 cycle parity, graph/rank/magnitude/count matching,
  case and perturbation counts, reproducibility, and the bound that
  restricted-binary communication height is >= Q4 height when the endpoints
  coincide and are fully active.
- A separate analytic J=+3, two-spin witness checks binary barrier=6
  versus Q4 barrier=3 with zero fields and costs.
- Every PR HEAD must pass fmt, Clippy, rustdoc, host and WGPU CI,
  security/supply-chain and semver before merging.

The panel isolates structural sign parity better than an unbalanced
negative-count control, but it does NOT demonstrate causality for real
networks: sign placement on a bridge versus a cycle, the fixed
degree-sequence, the small graph, artificial initializations, and
deterministic descent all matter. No external confirmatory data or real
network/model context has been evaluated.

A subsequent **separately preregistered** holdout experiment should vary
graph topology and graph size (still under the exhaustive limit), include
degree-preserving rewiring and balanced/unbalanced controls with the same
negative-edge counts and full metadata, and measure actual task-specific
recall under multiple perturbation strengths. Any physical cost comparison
must include complete routing and numerical attention, not this host
research oracle alone.

Relevant scientific and code references:
- Facchetti, Iacono, Altafini (2011). Computing global structural balance
  in large-scale signed social networks. PNAS.
  https://pubmed.ncbi.nlm.nih.gov/22167802/
- Jacquier, Vanessa (2025). Exploring Metastability in Ising Models:
  Critical Droplets, Energy Barriers and Exit Time.
  https://doi.org/10.1007/s11040-025-09514-2
- docs/research/ISING_Q4_TOPOLOGICAL_DEFECT_MONITOR.md
- docs/research/ISING_Q4_TEMPORAL_DEFECTS.md
- docs/research/ISING_Q4_BARRIER_RECALL.md
