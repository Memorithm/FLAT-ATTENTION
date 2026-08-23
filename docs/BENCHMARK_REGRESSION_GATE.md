# Same-Device Kernel Regression Gate

`examples/regression_gate.rs` is a CI-gated performance guard. It compares
opt-in kernel generations against the qualified Q4 portable baseline **inside
one process on one adapter**, so its thresholds are hardware independent: any
algorithmic regression of a candidate relative to the reference is visible on
every adapter, from Mesa lavapipe in CI to physical GPUs.

## What it measures

End-to-end wall time (upload + fused dispatch + readback) for fixed workloads,
median over `FLAT_REGRESSION_GATE_ITERATIONS` iterations (default 15) after a
fixed warm-up:

| Workload | Context | Gate |
|---|---|---|
| `q4_portable_d64` | vectorization off | reference |
| `q4_vec4_d64` | vectorization on | median ratio vs reference ≤ 1.20 |
| `q4_portable_d128` | vectorization off | median ratio vs reference ≤ 2.50 |

Both contexts verify via `kernel_variant_for_head_dim` that the intended
generation is actually selected before timing; a selection mismatch fails the
gate instead of measuring the wrong kernel.

## Why relative, same-device gates

Absolute timings vary across adapters, drivers and machines; a committed
absolute baseline would be wrong almost everywhere and flaky where it is
right. Ratios between kernels measured back-to-back on the *same* adapter are
stable across hardware because both sides scale with the device's speed.
This catches:

- a candidate generation becoming slower than the qualified baseline;
- pathological scaling (head-dimension doubling blowing up);
- kernel-selection drift (a workload silently running the wrong kernel).

It does not claim any absolute speed. Absolute performance evidence stays
governed by the physical-hardware qualification protocol
(`docs/RELEASE_BENCHMARK_SNAPSHOT.md`, Thor workflows) and reproducible
benchmark manifests (`src/benchmark_manifest.rs`).

## Environment

- `FLAT_REQUIRE_WGPU=1`: fail instead of skipping when no adapter exists.
- `FLAT_REGRESSION_GATE_ITERATIONS`: iteration count (default 15).
- `FLAT_REGRESSION_GATE_VEC4_RATIO` / `FLAT_REGRESSION_GATE_D128_RATIO`:
  threshold overrides for experiments; CI keeps defaults.

Exit code is non-zero with `regression_gate_verdict=fail` plus a machine-
readable reason on stderr.
