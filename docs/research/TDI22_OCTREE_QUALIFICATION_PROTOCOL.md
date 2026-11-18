# TDI-22 spatial-octree qualification — deterministic host-only V1

Status: **research CPU correctness gate; not physical performance qualification**.
Target: `crates/flat-algebraic-attention/examples/tdi22_octree_qualification.rs`.
Reference code: `crates/flat-algebraic-attention/src/torsor_octree.rs`.
Versioned approximation gate: `tdi22-octree-output-budget-v1`.

## What is implemented

The executable builds one octree per corpus and compares four **paired**
paths for each query:

1. `exact_tree` with `theta=0` (all keys evaluated exactly);
2. `spatial_only` with `theta=0.5` (no semantic bound);
3. `score_guard` with a within-node score-span budget `0.25`;
4. `six_output_budget`, which additionally requires per-component
   resultant and moment absolute-error budgets of `0.05` each.

An independent direct f64 Varignon + stable softmax implementation
provides a reference for all four. The reference does **not** use the
octree, stored origin-reduced envelopes, aggregate tree nodes, or
the `forward(theta=0)` implementation.

The fixture is **deterministic**, needs no network access, and contains
four panels:

- `uniform_projected_score`: varied wrench values but identical
  projection onto the distant linear query; eligible for near-exact
  mass-corrected summary;
- `smooth_semantic`: spatially distributed keys with gradually
  varying resultant and moment values;
- `far_adversarial`: spatially close keys with alternating opposing
  high-magnitude linear logits; distance alone is deliberately
  untrustworthy;
- `coincident_coordinates`: an unsplittable leaf with identical
  spatial positions, which must not fabricate a compressible branch.

Each panel is queried at one near point with nonzero angular velocity
and one distant point with a linear-only twist. The complete gate
requires that every key be covered exactly once, that exact-tree
results match the independent oracle within an explicit floating-point
tolerance, that the score-guard LSE error is within its declared
budget, and that every reported six-component analytical upper bound
is respected within floating-point comparison slack.

**Important numerical caveat:** the analytical bounds rely on exact
real arithmetic. The comparison tolerance is
`1e-9 * (1 + |exact| + |observed|)` and detects large deviations;
it is **not** directed-rounding interval arithmetic and does not
constitute a formal IEEE-754 error certificate.

## How to reproduce host-only validation

From the checked-out `Memorithm/FLAT-ATTENTION` repository:

```bash
cargo test -p flat-algebraic-attention --example tdi22_octree_qualification
cargo run -p flat-algebraic-attention --example tdi22_octree_qualification -- --keys 128 --samples 3 > /tmp/tdi22-octree-qualification.csv
```

The CLI accepts `--keys` between 16 and 4096 (default 256) and
`--samples` between 1 and 25 (default 5). Unsupported options,
negative values and out-of-range counts fail closed. A CI gate
executes the tests and the 128-key/3-sample fixture, then requires
33 CSV lines (header + four panels × two queries × four modes).

## CSV fields and interpretation

The CSV includes a panel/query/mode identity, key count, constructed
node count, owned reserved buffer bytes, build duration, independent
dense median query duration, mode median query duration, exact keys,
summarized nodes, summarized-key coverage, maximum absolute resultant
error, maximum absolute moment error, LSE error, a presence flag
for analytical bounds, and the maxima of the reported analytical
resultant/moment/LSE error bounds.

`reserved_buffer_bytes` measures only the capacity of the tree's
three owned Vec buffer families: keys, nodes, and leaf-key indices.
It **excludes** allocator headers, stack, temporary allocations and
process RSS. Its meaning is neither peak memory nor resident memory.

The durations are short, unpinned host-only observations. A median of
multiple repeated calls suppresses a single noisy timestamp but
**does not** prove a physical performance improvement: there is no
independent workload isolation, CPU frequency control, load capture,
hardware counter capture, cross-run confidence estimate, or declared
paired device occupancy. The reported ratios are intentionally left
to an external qualified analysis.

The exact CPU oracle and bounds do not make this a causal attention,
trained model, gradient, quantized KV, WGPU, GPU, or production route.

## Physical qualification gate through RemoteOps only

The host-only PR is not an authorization to run a new physical
experiment or to alter the default FLAT routing.

For a follow-on controlled experiment, the sole remote control plane
is `Memorithm/RemoteOps`. Before scheduling, preserve the exact
FLAT SHA, Rust 1.89.0 toolchain, fixture version, machine identity,
runner availability, load/occupancy, allocation limits and repetitions;
avoid competing Dell/Thor/Mac workloads. Use an isolated temporary
workspace and unprivileged CPU process with a bounded key count and
sample count. Record each raw CSV, host provenance, build flags,
process exit status and an immutable checksum; re-run as a genuinely
paired baseline/candidate study under the same isolation policy.

For a follow-on process **dispatched by RemoteOps after isolation
approval**, the workload command is:

```bash
cargo run --release -p flat-algebraic-attention --example tdi22_octree_qualification -- --keys 1024 --samples 7 > tdi22-octree-qualification.csv
```

The workload never requires CUDA/NVIDIA products or a resident inference
service. Distinguish CPU build/warmup cost, query cost, storage buffer
reservations, actual peak RSS, energy and downstream model quality:
this harness alone measures only the first three categories partially.
No physical measurements, speedup or production-quality claims are
established by adding or merging this research harness.
