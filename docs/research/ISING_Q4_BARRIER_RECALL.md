# Q4-TOPO-3 — Exact bounded energetic communication barriers and deterministic recall

Status: research-only, host-only Rust scalar oracle, not an inference kernel, training mechanism or production routing policy. The baseline Q4 Hamiltonian and its causal masking contract remain unchanged. No NVIDIA runtime or SDK is introduced.

## Exact mathematical question

For two specified Q4 configurations x and y on the **same frozen signed graph and eligibility mask**, define the communication height

    Phi(x,y) = min_{path from x to y} max_{s in path} E(s)
    B(x,y)   = Phi(x,y) - E(x)

where every path step may update **exactly one eligible vertex** to any of its other three Q4 states. This matches the local proposals considered by the existing Q4 update_one oracle, *not* their strictly downhill acceptance restriction. The paths may temporarily increase energy. The reverse-direction barrier need not have the same magnitude because E(x) and E(y) may differ, even though the communication height is symmetric.

The Rust oracle **exhaustively enumerates up to 4^8 = 65,536 configurations** and applies a bottleneck version of Dijkstra to find the smallest possible maximal path energy and a valid explicit witness. The hard limit is 8 total vertices, masked or eligible. Integer i128 evaluation uses exactly the existing signed coupling, activity cost and bias formula. Each vertex's two bits remain distinct even when both dormant signs have equal energy; structural masking is held fixed.

The result is exact **only for the declared local transition graph**. It is not a measured physical energy barrier, escape time, thermal transition probability, degree of topological protection or generalized barrier over continuous/synchronized transitions.

## Independent deterministic perturbation recovery

Function single_bit_perturbation_recall:

1. Validate the initial state as a Q4 fixed point under a complete strictly energy-decreasing sweep (reject non-attractors).
2. For each eligible vertex, independently flip **one** of its two Boolean coordinates: activity or polarity. Restore the same baseline before each trial.
3. Evolve each perturbed model using the unchanged greedy Q4 sweep with an explicit maximum budget, bounded to 1–1024 sweeps.
4. Report per trial: perturbed and final energies; sweeps; explicit fixed-point status; exact recovery of **all stored Q4 bits**; and recovery of **all energy-contributing spin values** z (which ignores the remembered sign of dormant vertices).

These are operationally distinct observables. A dormant unit may return to the same signed contribution z=0 without restoring its latent polarity bit. This is deliberately counted as effective-state recovery but **not** exact-bit recall. Neither one establishes useful next-token accuracy or long-term stochastic memory.

## Controlled fixtures

- Two ferromagnetically coupled sites J=+3, both initially active-positive, target both active-negative, zero fields and costs. Exact start/target energies are -3; the minimum-height path reaches E=0, yielding B=3. The path is checked to change only one vertex per step.
- A signed-coupling gauge transformation with corresponding spin-polarity redefinition preserves the exact communication height.
- A dormant-positive state with a positive activation penalty can preserve z=0 after an external polarity flip while failing exact latent-bit recall.
- A field-stabilized active-positive state returns from both single-bit perturbation types.
- Balanced versus sign-frustrated triangles with identical vertex count, topology and absolute edge weights: topology is verified independently and barriers are measured without claiming one outperforms the other.
- Explicit errors for oversized graphs, invalid endpoints, masked-state contamination, invalid sweep budgets and baselines that are not Q4 fixed points.

## Qualification and scientific boundary

Required: cargo fmt, workspace Clippy and rustdoc with warnings denied, focused unit tests, full host/WGPU CI, CodeQL, supply-chain and semver on the exact PR head.

The next honest experiment is a preregistered matched panel with graph-degree-preserving signed rewiring, controlled activation penalties and field strengths. Compare barrier values, exact stored-bit recall and effective-state recovery with and without frustrated cycles. Use separate held-out graph families, report all failures and avoid post hoc selection. Only measured real-device telemetry can substantiate any runtime gain.

Sources for the **pathwise energy-barrier definition**, not for model performance:

- Exploring Metastability in Ising Models: Critical Droplets, Energy Barriers and Exit Time (2025), Mathematics of Operations Research / Springer article https://doi.org/10.1007/s11040-025-09514-2
- Facchetti, Iacono & Altafini (2011), Computing global structural balance in large-scale signed social networks, PNAS https://pubmed.ncbi.nlm.nih.gov/22167802/

Related contracts:
- docs/research/ISING_Q4_BOOLEAN.md
- docs/research/ISING_Q4_TOPOLOGICAL_DEFECT_MONITOR.md
- docs/research/ISING_Q4_TEMPORAL_DEFECTS.md
