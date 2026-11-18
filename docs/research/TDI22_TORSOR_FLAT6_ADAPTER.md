# TDI-22 × FLAT-ATTENTION: six-channel torsor adapter, research V1

Status: **implementation-only experimental CPU scalar bridge**, not a production
route, confirmatory scientific experiment, FLAT GPU implementation, or evidence
that torsor attention improves training, inference, latency, bandwidth or memory.

## Source contract and ownership

Scientific origin: Memorithm/TDI, merged
[TDI PR #797](https://github.com/Memorithm/TDI/pull/797) as
`a8b62858aa7d8009322fad0d3306c59db00612dd`. Frozen TDI-22.0
convention remains authoritative. This adapter does not copy the TDI
scientific evaluator or its protected result material; it implements a
separate, versioned FLAT-side **f32** projection and checks its numerical
behavior against an independent f64 direct-Varignon dense oracle.

Research module:
`flat_algebraic_attention::tdi22_torsor_projection`,
contract string `tdi22-origin-reduced-flat6-f32-v1`.
It lives outside `flat_attention::api::v1`; it adds no GPU dispatch and
does not modify default or public runtime routing.

## Exact real-arithmetic factorization

An input torsor has resultant R, reduced moment M(P), and reduction
point P. An input query has linear component v, angular component
omega, and evaluation point Q. In TDI's sign convention:

    M(Q) = M(P) + (P - Q) x R
    C = M(P) + P x R
    M(Q) = C - Q x R

The TDI pairing is:

    score = scale * [v . R + omega . M(Q)]
          = scale * [(v + Q x omega) . R + omega . C]

A six-channel, standard FLAT dense-attention call is therefore formed as:

    FLAT Q_i = [v_i + Q_i x omega_i, omega_i]
    FLAT K_j = [R_j, C_j]
    FLAT V_j = [R_j, C_j]
    FLAT O_i = [sum_j alpha_ij R_j, sum_j alpha_ij C_j]

Only **once per query row**, reconstruct:

    R_out = O_i[0..3]
    M_out(Q_i) = O_i[3..6] - Q_i x R_out

The FLAT native f32 implementation is called with
`AttentionShape {batch:1,heads:1,seq_len:N,head_dim:6}`.
The existing `FlatAttentionConfig` owns the optional positive scale
(default 1/sqrt(6)), causal masking, normalized weights and log-sum-exp.

## Resource and numerical boundary

- Input square query/key sequences must have identical, nonzero length;
  the reference bounds N to at most 4,096.
- The prepared key bank is reusable; the **same f32 storage** supplies K
  and V, avoiding a second materialized V copy. Q uses 6N scalars.
- The underlying FLAT oracle remains full O(N²) work in the worst case;
  no hierarchical approximation or TDI's f64 tree is implicitly selected.
- No N×N score matrix is materialized by the FLAT online-softmax reference.
- Every input component, derived cross product, origin-reduced moment,
  projected query, and reconstructed output is checked for finite values.
  FLAT validates scale and scaled scores independently.
- Differences from the TDI f64 row are expected from f32 rounding;
  tests compare against a separately computed f64 **direct Varignon**
  per-key and per-query oracle. This is not bitwise scientific equivalence.
- Masked future keys cannot contribute to eligible causal outputs.
  The entire bank is still validated at build time.
- No dtype, multi-head, rectangular, training/backward or WGPU claim
  is made; these require separate scoped adapters and parity gates.
- Raw O/LSE values and physical performance are distinct:
  the tests do not measure latency, energy or effective data traffic.

## Implementation and reproduction

    cargo test -p flat-algebraic-attention tdi22_torsor_projection
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

Tests: full/causal independent f64 dense parity, source reduction point
invariance, joint origin translation, changed masked future keys, extreme
finite logits and invalid/nonfinite/overflow failures.

## Planned paired physical qualification

The next step is a strictly non-final, read-only RemoteOps-controlled CPU
comparison on an isolated runner. A valid paired report must pin both repo
SHAs, toolchain, CPU, compiler profile, N, score scale, causal policy,
warm-up and iteration counts and capture output error and median/p95
latencies for separately named arms. Device measurements must not be
inferred from logical score counts or from the standard GitHub-hosted CI.

Promotion to a FLAT production kernel is **not** authorized by this bridge.
