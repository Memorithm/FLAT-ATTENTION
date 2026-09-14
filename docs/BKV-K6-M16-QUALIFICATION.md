# BKV-K6 versus M16 WGPU qualification

Status: research-only evidence harness. No runtime routing change and no performance claim.

This harness is the first executable consumer of the BKV-K6.1 qualification record. It compares the Boolean-selected BKV-K6 numerical consumer against the dense M16 paged decode on the same already-resident Q/K/V buffers.

## What is qualified

The integration harness `tests/bkv6_m16_qualification.rs` verifies before interpreting timing:

1. BKV-K6 with an all-accept Boolean selection matches M16 O/LSE within the existing WGPU tolerance.
2. Sparse BKV-K6 matches a scalar oracle restricted to exactly the selected original token positions.
3. The BKV-K6.1 record reproduces the exact full, selected, and avoided logical numerical K/V byte counts computed by BKV-K5 selection metadata.

The Boolean signatures in this first harness are a deterministic matched-density control. They are not a learned semantic index and therefore do not establish model quality.

## Timing scope

The harness follows the repository's existing host-observed WGPU benchmark convention. It reports medians for:

- query Boolean-signature generation on the host;
- Boolean search plus selected-page-table materialization on the host;
- selected K6 encode plus `queue.submit` return;
- the subsequent explicit `device.poll(...wait...)` synchronization;
- dense M16 encode plus submit plus poll.

These are host-observed wall-clock phases. They are **not GPU timestamp measurements** and must not be described as pure kernel execution time.

Uploads and readbacks are excluded from the timed phases because K6 and M16 consume the same already-resident Q/K/V buffers. Readback is used only for correctness/quality evidence outside the timing region.

## Running the harness

```bash
cargo test --release --features wgpu \
  --test bkv6_m16_qualification \
  -- --nocapture
```

Optional controls:

```text
FLAT_BKV_QUAL_PAGES          default 16
FLAT_BKV_QUAL_PAGE_SIZE      default 8
FLAT_BKV_QUAL_MAX_DISTANCE   default 1, clamped to 8
FLAT_BKV_QUAL_WARMUP         default 3
FLAT_BKV_QUAL_ITERS          default 9
```

Set `FLAT_REQUIRE_WGPU=1` when absence of a WGPU adapter must be a hard failure.

## Evidence interpretation

The output records the exact Git commit visible to the harness, WGPU adapter identity, geometry, selection density, Boolean-index bytes read, logical numerical K/V bytes touched/avoided, all timing phases, correctness gates, a synthetic-fixture quality gate, and the BKV-K6.1 promotion disposition.

A `Promote` result from this synthetic fixture is not sufficient to enable BKV-7 adaptive tiering. Promotion beyond the research harness additionally requires representative workload/model quality evidence and hardware-qualified results on the target platform. Negative or no-effect measurements are retained as valid evidence.
