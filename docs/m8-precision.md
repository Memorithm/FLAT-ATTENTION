# M8 — Mixed-precision I/O contract

M8 reduces attention input/output storage traffic without weakening the numerical core of FLAT-ATTENTION.

## Binary16 scope

The initial portable mixed-precision specialization supports `head_dim = 64` and `128`.

- Q, K and V are stored as IEEE-754 binary16.
- The shader contains `enable f16;` and is created only on an adapter exposing the matching WGPU feature.
- Every Q/K/V element is converted to `f32` immediately after its storage load.
- Q·K dot products accumulate in `f32`.
- score scaling, row maxima, exponentials and online-softmax sums are `f32`.
- P·V output accumulation is `f32`.
- O is converted to binary16 only during final writeback.
- LSE remains `f32` because it is part of the future backward/recomputation contract.

The kernel never materializes an `N × N` score/probability matrix.

## Rust storage type

`F16` is implemented inside this repository with no external half-precision dependency. It is a two-byte IEEE binary16 storage value with deterministic conversion to/from `f32` using round-to-nearest, ties-to-even.

It is deliberately not presented as a general arithmetic type. FLAT-ATTENTION uses it at the I/O boundary and performs the algorithm in `f32`.

## Packed output

The M8 storage output remains one shader binding:

```text
[ O pairs as u32(bitcast(vec2<f16>)) | LSE words as f32 bit patterns ]
```

For D64/D128, O always has an even number of elements. Relative to the f32 path:

- Q bytes: 50%
- K bytes: 50%
- V bytes: 50%
- O bytes: 50%
- LSE bytes: unchanged

These are storage-size facts, not claims about physical DRAM transactions or runtime speed.

## Runtime policy

`WgpuF16Attention` is the explicit resident binary16 executor. Construction fails explicitly when the selected adapter does not expose f16 shader support.

`WgpuPreferredAttention` is the capability-based convenience router:

- D64/D128 + f16-capable adapter: use M8 f16 I/O;
- unsupported head dimension: use the already-qualified f32 WGPU path;
- no f16 capability: use the already-qualified f32 WGPU path;
- finite f32 values that overflow binary16 during quantization: use the f32 WGPU path;
- no path falls back to CPU.

The f32 executor remains independently available and unchanged.

## Numerical qualification

Parity is measured against the deterministic Rust oracle after first quantizing Q/K/V to binary16, because that is the actual numerical input observed by the M8 shader.

The reference O is then quantized to binary16 before comparison with GPU O. LSE is compared directly in `f32`.

Current test tolerances:

- O: `atol = 2e-3`, `rtol = 5e-3`;
- LSE: `atol = 8e-4`, `rtol = 2e-3`.

Stress tests must reject non-finite input before dispatch and must not accept NaN/Inf output.

## BF16 contract

M8 does **not** claim a portable WGSL bf16 implementation.

The future bf16 path is reserved to follow the same architecture when the selected open backend exposes a capability that can be validated and dispatched without vendor SDK code:

1. two-byte bf16 storage representation;
2. immediate promotion to f32 after load;
3. f32 Q·K accumulation;
4. f32 online-softmax state;
5. f32 P·V accumulation;
6. explicit final conversion policy for O;
7. f32 LSE;
8. capability-based GPU fallback rather than a hidden CPU path.

Until those conditions are met, no `Bf16` GPU executor or performance claim is exposed.

## Benchmark policy

`examples/f16_bench.rs` compares resident f32 and resident f16 paths using the same binary16-representable input values. The harness reports timings; repository documentation makes no universal speedup claim without physical-GPU measurements.
