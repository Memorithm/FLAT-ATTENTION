# FDAL3 — DA-LUC direct-compressed WGPU candidate

FDAL3 is the first portable GPU correctness candidate for the research-only
DA-LUC program. It is **not** a production route, performance result, novelty
claim, or complete implementation of the FDAL0 representation schema.

## Candidate property

The candidate executes `q_len = 1` attention without calling the FDAL1
`decode_keys()` or `decode_values()` helpers and without constructing a full
dense K or V tensor.

For keys, the shader builds a query-local lookup table over the stored K
codebook, consumes packed K indices, and accumulates LUT entries into attention
scores. For values, the shader reads packed U8 values plus per-group F32 scales
and U8 zero points and performs the scalar conversion in registers during the
online-softmax accumulation.

The absence of dense K/V materialization is **not** a claim of zero
dequantization. Scalar/register conversion still occurs for V.

## Exact supported subset

FDAL3 candidate version 1 accepts only:

- `q_len = 1` decode semantics from FDAL2;
- contiguous storage;
- `BatchHeadToken` physical row order;
- zero-filled aligned planes with at least 4-byte alignment;
- key and value head dimensions at most 128;
- F32 K codebooks, shared across KV heads or per KV head;
- 8-bit LSB0 K indices;
- no K sparse residual;
- groupwise-affine 8-bit LSB0 V;
- F32 V scales;
- U8 V zero points;
- no V sparse residual;
- a query-local `subspaces * codebook_entries` LUT of at most 2048 entries.

Everything else fails closed before dispatch. In particular, v1 rejects paged
storage, MSB0 streams, sub-byte K/V, dense V, F16/BF16 representation planes,
and sparse residuals. Those are future candidate extensions, not implicit
fallbacks.

## Device staging

The FDAL1 byte planes for K indices, V values and V zero points are staged as
word-aligned `u32` buffers so WGSL can address their bytes portably. This
preserves the compressed payload bytes; it is not dense K/V reconstruction.

F32 codebook and scale scalars are copied into native F32 device buffers for the
portable candidate. Therefore FDAL3 v1 is not a zero-copy representation ABI.
A future external/runtime adapter may define a backend-owned resident layout only
through an explicit versioned contract.

## Correctness authority

The comparison target is
`DalucOraclePayload::q_len1_attention_direct`, the FDAL2 deterministic scalar
oracle over the same encoded payload. Output and LSE are checked independently.
The dense reconstructed comparator from FDAL2 remains available above that for
semantic auditing.

Vulkan lavapipe is accepted here only as portable shader/device correctness
evidence. It is not real-GPU performance evidence.

## Reproducible commands

Fail-closed host capability contract:

```bash
cargo test --features wgpu --test fdal3_da_luc_wgpu \
  fdal3_plan_is_narrow_fail_closed_and_declares_no_dense_materialization \
  -- --nocapture
```

Run the direct-compressed candidate on the available WGPU device and require an
adapter:

```bash
FLAT_REQUIRE_WGPU=1 cargo test --features wgpu \
  --test fdal3_da_luc_wgpu \
  direct_compressed_wgpu_matches_fdal2_scalar_oracle \
  -- --nocapture
```

Run both FDAL3 tests:

```bash
FLAT_REQUIRE_WGPU=1 cargo test --features wgpu \
  --test fdal3_da_luc_wgpu -- --nocapture
```

After merge, the dedicated GitHub qualification can be dispatched from any
working directory:

```bash
gh workflow run fdal3-da-luc-wgpu.yml \
  --repo Memorithm/FLAT-ATTENTION \
  --ref main
```

On the physical Thor host, the second or third command above is the first
DA-LUC-specific GPU correctness trial. A latency/tokens-per-second/memory trial
must use a separate committed benchmark/evidence protocol and is intentionally
not inferred from these correctness tests.
