# Q4-TOPO-1 — Signed-cycle frustration and state-dependent defect monitoring

Status: research-only, host-only, read-only Rust observer. The scope is the
frozen Boolean-quaternary Ising controller and its causal FLAT attention
candidate topology, NOT a trained or production routing policy.

## Why topology is analytically distinct from local neuron scores

A Boolean-quaternary site has an activity bit and a remembered polarity bit.
The structural admission mask is a third independent bit. Edges have signed
nonzero i16 couplings. This creates two different classes of observations:

1. **Structural signed-graph frustration.** For a complete undirected cycle C,
   the sign product P(C) = product_{edge in C} sign(J_edge) is invariant under
   a gauge/switching transformation (flipping a vertex's spin convention and
   all incident coupling signs). If P(C) = -1, no binary spin assignment can
   satisfy every edge of C. Existence of negative cycles is independent of
   the current Q4 activity or polarity. Zero coupling edges are omitted.

2. **State-dependent interfaces.** Activity boundaries connect one active and
   one dormant eligible site. Active polarity walls connect opposite active
   orientations. Latent polarity boundaries connect opposite remembered
   polarities with at least one dormant site. Unsatisfied active couplings
   connect active spins whose product disagrees with their coupling sign;
   their local excess above the pair optimum is 2*abs(J). These quantities
   change when Q4 state changes. They are not all gauge-invariant.

An unsatisfied edge on an entirely balanced graph is not an intrinsic
structural defect. Conversely, a negative cycle remains structurally
frustrated even when all of its vertices are dormant and its instantaneous
energy contribution vanishes. Do not conflate these cases.

## Implemented read-only observer

The Rust module is
crates/flat-algebraic-attention/src/ising_topological_defects.rs.

- Validates exact bitplane dimensions, out-of-range padding bits, forbidden
  active-but-masked bits, canonical unique sorted edges.
- Removes structurally masked vertices, incident edges, and zero couplings
  *before* finding components and signed cycles.
- Computes connected components (including isolated eligible vertices),
  isolated count, active count, effective edges and cycle rank.
- Builds a deterministic BFS spanning forest; each non-tree edge defines
  one *fundamental cycle*. Reports the number with negative coupling sign
  product, and whether the whole signed graph is structurally balanced.
  IMPORTANT: the **count** of negative fundamental cycles depends on the
  chosen cycle basis; the balanced/unbalanced predicate does not. Neither
  value is the minimal frustration index.
- Returns localized flagged edges for activation interfaces, active/latent
  polarity mismatches, unsatisfied active couplings and negative basis-cycle
  closures. A closure edge's LOCATION is forest-dependent.
- A query-scoped adapter applies the exact structural and causal eligibility
  mask to the frozen head plan before observing the graph; it reports the
  initial configuration **before** energy-descent sweeps.
- No mutation or implicit repair of the routing policy. A future temporal
  monitor may compare consecutive snapshots, but must explicitly control
  for changing masks and cycle bases.

All-accept dense FLAT remains the numerical baseline. Neither the observer
nor the quaternary controller is evidence of topological error protection,
robust associative memory, downstream accuracy improvement, or speedup.

## Experimental controls required before any promotion

Compare signed topology under fixed matched graph size/density and signs:
balanced trees and cycles; negative triangles; disconnected components;
zero-weight edges; causal erosion of a future-only negative cycle; independent
gauge transformations; fully dormant configurations; active-state defects on
balanced cycles; random-signed and degree-preserving rewiring controls.

Retain candidate identity and attention O/LSE parity when plugging a later
topology-conditioned policy into the qualified sparse execution path. Measure
wall-clock cost including topology diagnostics, route preparation, transfers,
numeric kernels, and output quality; no post hoc changes to holdout policies.

## Reference sources

- Facchetti, G.; Iacono, G.; Altafini, C. (2011). Computing global structural
  balance in large-scale signed social networks. PNAS.
  https://pubmed.ncbi.nlm.nih.gov/22167802/
- Facchetti, G.; Iacono, G.; Altafini, C. (2012). Exploring the low-energy
  landscape of large-scale signed social networks. Physical Review E.
  https://journals.aps.org/pre/abstract/10.1103/PhysRevE.86.036116
- Gallo et al. (2024). Testing structural balance theories in heterogeneous
  signed networks. Communications Physics.
  https://www.nature.com/articles/s42005-024-01640-7
- Phase Diagram of the Ashkin-Teller Model (2024), Communications in
  Mathematical Physics: https://doi.org/10.1007/s00220-023-04925-0

The references motivate graph-signed topology and its diagnostic measures,
not a claim that this Q4 controller is an Ashkin-Teller system or that
topological memory has been experimentally demonstrated.
