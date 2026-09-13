# Multi-Algebra Attention (MAA) research contract

Status: research-only. This document does not change `flat_attention::api::v1`, default routing, or any performance claim.

## Objective

Evaluate whether FLAT-ATTENTION can reduce unnecessary numerical attention work by coordinating four algebraic domains without conflating their laws:

1. classical Boolean admission and bit-packed routing;
2. linear/affine computation over the finite field `F2`;
3. algebraic-normal-form (Zhegalkin) polynomials for explicit nonlinear Boolean interactions;
4. max-plus constraints for readiness, priority, and scheduling decisions.

Dense qualified FLAT attention remains the semantic and numerical reference. A research decision layer may reject work only under an explicitly qualified policy; it must never silently weaken the existing correctness contract.

## Existing foundation

FLAT already owns Boolean masks, packed Q/K signatures, Hamming/XNOR-style admission, Boolean KV metadata, and WGPU routing before numerical K/V staging. MAA must reuse those contracts rather than duplicate them.

The first MAA implementation is therefore an algebraic foundation, not a new attention kernel.

## Algebraic domains

### Boolean control plane

Boolean logic owns discrete control decisions such as admitted/rejected, ready/not-ready, and feature predicates. Existing M13B contracts remain authoritative for bit-packed attention admission.

### `F2`

`F2 = {0, 1}` with addition `XOR` and multiplication `AND`.

For vectors `x, y in F2^n`:

```text
x + y = x XOR y
<x, y> = XOR_i (x_i AND y_i)
```

An affine predicate is:

```text
f(x) = b XOR <a, x>
```

The MAA `F2` implementation is a deterministic scalar/packed-bit oracle. It is not a performance claim and is not a replacement for the existing Boolean attention API.

### Zhegalkin / algebraic normal form

A Boolean function may be written uniquely as an algebraic-normal-form polynomial over `F2`:

```text
P(x) = a_empty XOR XOR_{S != empty} a_S * product_{i in S} x_i
```

with `x_i^2 = x_i`.

MAA will explicitly decompose:

```text
P(x) = L(x) XOR N(x)
```

where `L` contains degree 0 and 1 terms and can be evaluated through the `F2` layer, while `N` contains degree >= 2 interactions. Zhegalkin work must therefore reuse the qualified `F2` semantics rather than implement a second incompatible notion of Boolean arithmetic.

### Max-plus

The max-plus semiring uses:

```text
a (+_max) b = max(a, b)
a (*_plus) b = a + b
```

with an explicit negative-infinity element.

MAA will use max-plus only for temporal/readiness constraints unless later evidence justifies a broader role. Example:

```text
t_route = max(t_query + d_signature,
              t_metadata + d_metadata)

t_attention = max(t_route + d_route,
                  t_kv_ready + d_kv)
```

This is a scheduling model, not a claim that GPU concurrency or latency improvement exists. Such claims require trace/timing evidence under the existing M13B.4 evidence rules.

## Cooperation model

The intended research graph is:

```text
Q/K state
  -> existing Boolean signatures / admission metadata
  -> optional F2 linear/affine predicates
  -> optional Zhegalkin nonlinear predicates
  -> survivor set
  -> optional max-plus readiness / priority plan
  -> exact qualified FLAT attention for survivors
```

The domains are cooperative, not mandatory serial filters. A plan may legitimately use only a subset of them.

## Promotion rules

No MAA component may enter the stable public API or default runtime routing until all applicable gates are satisfied:

- mathematical definition is frozen;
- deterministic Rust oracle tests are green;
- all-accept behavior preserves the existing dense reference contract where applicable;
- malformed dimensions or metadata fail closed;
- no hidden CPU fallback is presented as GPU execution;
- portable GPU work, if added, is qualified against the Rust oracle;
- any performance claim is tied to reproducible real-device evidence;
- quality/sparsity claims include dense and matched-control baselines.

## Planned milestones

- **MAA-0**: mathematical contract and repository boundary.
- **MAA-1**: packed deterministic `F2` vector, linear-map, and affine-predicate oracle.
- **MAA-2**: canonical Zhegalkin representation with exact linear/nonlinear decomposition.
- **MAA-3**: explicit max-plus values and deterministic readiness/critical-path oracle.
- **MAA-4**: cooperation plan binding existing M13B Boolean metadata to the three new algebraic layers.
- **MAA-5**: evidence schema and matched experiments versus dense FLAT and Boolean-only routing.
- **MAA-GPU**: portable WGPU candidates only after host-oracle qualification justifies them.

## Non-goals for the foundation series

The first series does not:

- replace softmax;
- replace exact `Q.K` scoring;
- alter default kernel selection;
- claim lower latency, higher throughput, lower energy, or lower physical memory traffic;
- duplicate the M13B Boolean mask/signature contracts;
- promote any research type into `api::v1`.
