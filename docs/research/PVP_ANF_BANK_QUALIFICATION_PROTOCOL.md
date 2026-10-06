# PVP ANF-bank semantic qualification v1

Frozen before implementation on 2026-10-06. This is a deterministic functional
qualification, not a timing experiment or a model-side promotion.

## Question and fixed corpus

Do the existing scalar-u32, vec4, fused2 and tile8 WGPU realizations turn the
same explicit Boolean ANF coefficients into exactly the same truth tables as
an independent direct monomial evaluator?

Use these eight `(K, G)` geometries in order: `(1, 1)`, `(2, 31)`, `(4, 65)`,
`(8, 129)`, `(64, 257)`, `(256, 513)`, `(2048, 257)`, `(256, 2048)`.
For each geometry use three banks, making 24 cases:

- `boundary`: gate index modulo six chooses zero, constant one, lowest-variable
  monomial (constant at K=1), full-degree monomial, constant XOR lowest-variable
  monomial, or parity of all variables. Repeated masks cancel over GF(2).
- `sparse4`: exactly `min(4,K)` distinct terms per gate.
- `dense32`: exactly `min(32,K)` distinct terms per gate.

For the latter two banks, term `t` of gate `g` has mask
`(257*g + 73*t) mod K`. Because K is a power of two and 73 is odd, these
masks are distinct within each gate. Mask zero is the constant monomial.
The fixture must reject duplicate terms in these two banks.

## Oracle and gates

The direct oracle evaluates every gate at every assignment `x` by XORing
`(x & mask) == mask` for its explicit terms. It must not call a zeta/butterfly
implementation, enumerate submasks, or derive expected outputs by transforming
the packed input. Independently pack oracle truth bits into each physical
layout, including zero padding.

Host tests check scalar-u32 and vec4 against that oracle. An actual WGPU test
executes all four existing kernels, checks every output word against the oracle,
then executes the same kernel again and checks exact recovery of coefficients.
Compare complete words, including unused gate lanes. Small domains exercise
fused-kernel fallbacks; 31/65/129/257/513 gates exercise lane/vector boundaries;
G=2048 exercises a larger bank. Every frozen case is required, without adapting
the corpus after observing outcomes.

Unavailable adapters fail when `FLAT_REQUIRE_WGPU` is set. Software adapters
qualify correctness only. Preserve backend identity in test output. No timing,
GPU speedup, physical register-allocation, model quality, attention replacement,
cross-repository binary compatibility, or SML internalization follows.

## Reuse boundary

The formula, explicit term lists and direct oracle design can be ported to
SciRust and NNIS after destination review. This suite uses only FLAT's existing
Rust/WGPU kernels and adds no runtime dependency. SML retains ANF semantics and
the final model's self-sufficiency; the future harness remains separate.
