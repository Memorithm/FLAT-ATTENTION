# ISING-Q4 — Boolean quaternary attention admission (research only)

Research-only host oracle under flat-algebraic-attention. This does not change
the public attention API or default execution route, Q/K/V semantics or SML.

## Four states, two Boolean coordinates

| Activation | Polarity | State | Contribution z |
| --- | --- | --- | --- |
| 0 | 0 | Dormant-negative | 0 |
| 0 | 1 | Dormant-positive | 0 |
| 1 | 0 | Active-negative | -1 |
| 1 | 1 | Active-positive | +1 |

A third Boolean eligibility plane acts as an independent causal/structural
mask. A masked vertex cannot be active or contribute to other vertices.
The dormant states preserve a polarity memory but have the same energy.
Strict downhill dynamics reject energy-neutral polarity changes.

Let a_i be 0/1, p_i be -1/+1 and z_i = a_i*p_i.
With bias h_i, activity cost c_i and undirected J_ij:

    E = sum_i c_i*a_i - sum_i h_i*z_i - sum_(i<j) J_ij*z_i*z_j
    H_i = h_i + sum_j J_ij*z_j
    deltaE = c_i*(a'_i-a_i) - H_i*(z'_i-z_i)
    H_j(new) = H_j(old) + J_ij*(z'_i-z_i)

Energy and delta use i128; graph edges have signed i16 couplings, biases
and costs are i32, cached local fields i64. A sparse deterministic sweep
strictly decreases energy or leaves it unchanged. There is no claim of a global
minimum, differentiability, Boltzmann sampling, speedup or language quality.

## Memory and scope

At n=1,000,000 vertices the three bitplanes consume exactly 375,000 bytes
(before alignment and allocator overhead). This excludes biases, costs, cached
fields, CSR offsets and duplicated undirected adjacency, which may dominate
the true memory footprint. Retained binary polarity is not a learned weight.

This is a purpose-built two-Boolean *activity x polarity* controller, NOT the
standard Ashkin-Teller Hamiltonian. The latter has two Ising spins per site
and additional interaction terms.

For FLAT, quaternary states would only propose candidates or pages for the
existing exact numerical attention. An active negative spin does not mean a
negative attention probability. Compare against all-accept, dense FLAT,
density-matched random masks and topology controls with quality/O/LSE evidence.
SML remains non-attention: a possible circuit/page controller requires separate
evaluation, not an attention implementation imported into the SML core.

## Host validation

    cargo test -p flat-algebraic-attention --lib ising_quaternary
    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings

Tests cover all four states, latent polarity retention, all local energy
deltas against independently recomputed total energy on a frozen 3-site
graph, incremental neighbor fields, masks and malformed bitplanes.
GPU performance, memory advantages and model quality are not established.

References:
- https://mathworld.wolfram.com/Ashkin-TellerModel.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10858004/
- docs/research/ISING_ATTENTION_VECTORS.md
