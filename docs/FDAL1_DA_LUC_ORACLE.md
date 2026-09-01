# FDAL1 — deterministic DA-LUC representation oracle

Status: research-only host oracle. No runtime promotion, GPU-performance claim, novelty claim, or production compression claim is made by this milestone.

Base contract: FDAL0 merged in PR #174 at `ddaaa45f0f1b8e6d53be755934a9fe9dba546d94`.

## Scope

FDAL1 turns the versioned FDAL0 descriptor into a deterministic, byte-backed host representation that can be encoded, decoded, compared with dense K/V, and accounted exactly before any direct compressed-attention kernel exists.

The implementation lives at `flat_attention::api::research_da_luc_oracle`, deliberately outside stable `api::v1`. It consumes the FDAL0 contract but does not change production routing or defaults.

## Canonical dense input/output

Oracle `K` and `V` inputs and decoded outputs use canonical logical order:

```text
[batch, kv_heads, kv_len, feature]
```

The physical payload follows FDAL0 `row_order`, storage topology, packed bit order, codebook scope, padding rule, and plane alignment.

## Key oracle

- The caller provides the K codebook. FDAL1 does **not** train or calibrate codebooks.
- The codebook is first rounded to its declared storage dtype (`f16`, `bf16`, or `f32`).
- Each K subspace selects the stored codebook entry with minimum squared L2 error.
- Ties are deterministic: the lowest codebook index wins.
- Indices are packed using the declared width and bit order.
- Sparse residuals are selected from the largest absolute full-vector reconstruction errors.
- Residual coordinate ties prefer the lower logical coordinate.
- The host oracle stores a fixed `max_entries_per_vector` residual budget, avoiding hidden count metadata.
- Coordinate and bitmap residual forms both obey the FDAL0 v1 interpretation.

## Value oracle

Dense V stores the declared floating dtype directly.

Groupwise affine V is deterministic:

- symmetric mode (`zero_point=None`) uses signed two's-complement packed integers;
- affine `U8`/`U16` modes use unsigned packed integers plus one zero point per group;
- scales are rounded to the declared scale dtype before integer selection;
- a non-zero group whose stored scale underflows or becomes non-finite fails closed;
- residuals are selected after primary V reconstruction using the same deterministic full-vector rule as K.

## Paged host semantics

FDAL0 intentionally leaves backend page-table ABI ownership to adapters. FDAL1 therefore does not invent a backend ABI.

For host evidence only, the oracle accepts a per-batch logical-page -> physical-page map and serializes each entry as little-endian `u32`. If no map is supplied, a deterministic identity map is used. Aliasing or out-of-range pages fail closed.

This page map is evidence payload metadata, not an NNIS, SLHAv2, WGPU, or CUDA layout contract.

## Exact storage accounting

`DalucOraclePayload::storage_report` counts the actual owned byte planes and reports:

- K codebook bytes;
- packed K index bytes;
- K residual value bytes;
- K residual coordinate/bitmap bytes;
- V data bytes;
- V scale bytes;
- V zero-point bytes;
- V residual value bytes;
- V residual coordinate/bitmap bytes;
- page-map bytes;
- non-byte-aligned packing tail bits;
- zero-filled alignment padding bytes;
- explicit external metadata bytes supplied by the caller;
- total representation bytes;
- logical K+V scalar count;
- dense baseline dtype and bytes under the same capacity/topology;
- effective bits/value;
- compression ratio against that declared dense baseline.

The oracle deliberately does not guess a runtime descriptor serialization size. A surrounding runtime/evidence protocol must pass its real metadata byte count to the accounting call.

## Error evidence

`reconstruction_report` produces separate K and V:

- sample count;
- maximum absolute error;
- mean absolute error;
- RMSE.

These are isolated representation metrics only. They are not model-quality evidence and do not authorize compressed attention routing.

## Reproducible tests

Focused public integration gate:

```bash
cargo test --test fdal1_da_luc_oracle -- --nocapture
```

Full workspace regression matrix:

```bash
cargo test --workspace --all-features
```

Deterministic error/storage sweep:

```bash
cargo run --release --example fdal1_da_luc_oracle_sweep
```

The dedicated GitHub Actions workflow can also be invoked manually:

```bash
gh workflow run fdal1-da-luc-oracle.yml --ref main
```

It executes the public integration contract and runs the deterministic sweep twice, requiring byte-for-byte identical output and all 12 declared cases.

## Non-goals

FDAL1 does not provide:

- codebook training or calibration;
- q_len=1 LUT attention scoring (FDAL2);
- direct compressed WGPU/open-codegen execution (FDAL3);
- NNIS physical layout or kernel integration (FDAL4);
- dynamic precision routing;
- PHR-Lite;
- real-model quality evidence;
- DA-LUC-specific Thor performance evidence;
- an 8x-16x memory claim;
- a "zero dequantization" claim.

## Next gate

FDAL2 may consume this exact oracle representation to implement direct scalar q_len=1 compressed scoring and value accumulation. It must prove equivalence to the mathematical dense reconstruction semantics before any portable GPU candidate is attempted.
