# M8 — Mixed-precision I/O contract

M8 reduces attention input/output storage width without weakening the numerical core of FLAT-ATTENTION.

## Why packed binary16

SciRust currently aligns FLAT-ATTENTION with WGPU 0.20 / Naga 0.20. CI proved that this frontend does not parse native WGSL `f16` syntax: both the later `enable f16;` form and direct `f16` scalar types are rejected.

M8 therefore does **not** upgrade the GPU stack and does not introduce a vendor compiler. Instead, two IEEE-754 binary16 values are packed in each `u32` storage word. WGSL converts those words at the storage boundary with `unpack2x16float` and `pack2x16float`.

This keeps the actual Q/K/V/O representation at two bytes per scalar while the shader itself uses only baseline `u32` and `f32` scalar types.

## Binary16 scope

The initial portable mixed-precision specialization supports `head_dim = 64` and `128`.

- Q, K and V are true IEEE-754 binary16 values, two scalars per `u32` storage word.
- each loaded pair is unpacked directly to `vec2<f32>`;
- Q·K dot products accumulate in `f32`;
- score scaling, row maxima, exponentials and online-softmax sums are `f32`;
- P·V output accumulation is `f32`;
- O is rounded and packed to IEEE binary16 only during final writeback;
- LSE remains `f32` because it is part of the future backward/recomputation contract.

The kernel never materializes an `N × N` score/probability matrix.

## Rust storage type

`F16` is implemented inside this repository with no external half-precision dependency. It is a two-byte IEEE binary16 storage value with deterministic conversion to/from `f32` using round-to-nearest, ties-to-even.

It is deliberately an interchange/storage type rather than a general arithmetic type. FLAT-ATTENTION performs the attention algorithm in `f32`.

Before upload, Rust combines each pair explicitly into one `u32`:

```text
word[15:0]  = first F16 bits
word[31:16] = second F16 bits
```

The shader applies `unpack2x16float` to recover the corresponding two `f32` values.

## Packed output

The M8 storage output remains one shader binding:

```text
[ O pairs as pack2x16float(vec2<f32>) | LSE words as f32 bit patterns ]
```

For D64/D128, O always has an even number of elements. Relative to the f32 path:

- Q bytes: 50%;
- K bytes: 50%;
- V bytes: 50%;
- O bytes: 50%;
- LSE bytes: unchanged.

These are representation-size facts, not claims about physical DRAM transactions, cache behavior or runtime speed.

## Runtime policy

`WgpuF16Attention` is the explicit resident packed-binary16 executor. It requests no native shader-f16 feature.

`WgpuPreferredAttention` is the shape/backend convenience router:

- D64/D128 + accepted packed-binary16 pipeline: use `PackedF16`;
- unsupported head dimension: use the already-qualified f32 WGPU path;
- backend rejection of the packed-binary16 shader: use the already-qualified f32 WGPU path;
- finite f32 values that overflow binary16 during quantization: use the f32 WGPU path;
- no path falls back to CPU.

The direct `WgpuF16Attention` constructor reports shader rejection explicitly so tests cannot hide a broken packed path.

## Numerical qualification

Parity is measured against the deterministic Rust oracle after first quantizing Q/K/V to binary16, because those are the exact numerical inputs observed by the shader after unpacking.

The reference O is then quantized to binary16 before comparison with GPU O. LSE is compared directly in `f32`.

Current test tolerances:

- O: `atol = 2e-3`, `rtol = 5e-3`;
- LSE: `atol = 8e-4`, `rtol = 2e-3`.

Stress tests reject non-finite input before dispatch and require finite outputs.

## Native f16 and BF16

M8 does **not** claim native f16 arithmetic. Packed binary16 is an I/O compression format around an FP32 attention core.

A future native-f16 kernel may be added only after the project's open WGPU/Naga stack supports and validates the required language/runtime capability without destabilizing the qualified paths.

M8 also does **not** claim a portable WGSL bf16 implementation. A future bf16 path must preserve the same architecture:

1. two-byte bf16 storage representation;
2. explicit conversion to f32 after load;
3. f32 Q·K accumulation;
4. f32 online-softmax state;
5. f32 P·V accumulation;
6. explicit final conversion policy for O;
7. f32 LSE;
8. GPU fallback rather than a hidden CPU path.

Until those conditions are met, no bf16 GPU executor or performance claim is exposed.

### Host bf16 contract (FML1 slice)

The host-side half of that architecture now exists and fixes the semantics a future
GPU path must match:

- `Bf16` (`src/bf16.rs`) is a dependency-free two-byte storage value equal to the upper
  16 bits of an IEEE binary32. Decode is exact (`bits << 16`). Encode is
  round-to-nearest, ties-to-even; finite values rounding past `0x7f7f` become infinity;
  NaN stays NaN with its sign kept and the quiet bit forced, so rounding can never
  turn NaN into infinity or zero.
- `forward_reference_bf16` (MHA) and `forward_reference_grouped_bf16` (GQA/MQA) are
  *defined* as exact decode → the FP32 scalar oracle → RNE encode of O. Q·K,
  online-softmax state, P·V and LSE are FP32 and bit-identical to `forward_reference`
  / `forward_reference_grouped` on the decoded inputs. The only error the bf16 path adds
  relative to that FP32 oracle is the final O rounding, bounded by `2^-8 · |O|`
  (`tests/bf16_oracle.rs`). Input quantization error relative to full-precision
  inputs is data-dependent and is not claimed.
- bfloat16 shares binary32's exponent range, so finite bf16 inputs can overflow the
  FP32 score; those cases fail closed with `NonFiniteScore` (see M9).

Because bf16 decode is a 16-bit shift, a future WGSL path can unpack two bf16 values
per `u32` with baseline integer operations and no `f16`/`bf16` shader capability. That
kernel, its device parity against these oracles and any bandwidth measurement remain
open work; nothing here is a GPU or performance claim.

## Benchmark policy

`examples/f16_bench.rs` compares resident f32 and packed-binary16 paths using the same binary16-representable input values. The harness reports timings and logical representation bytes; repository documentation makes no universal speedup claim without physical-GPU measurements.
