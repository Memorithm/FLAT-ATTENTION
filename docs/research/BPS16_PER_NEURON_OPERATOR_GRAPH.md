# BPS-16: per-neuron, operation-specific recurrent connections

**Research-only model.** 64 packed Boolean neurons; **four independent
incoming connections per neuron**, one for each BPS-15 operation.
This is 256 individually selectable recurrent sources (0..63), **not**
four shared rotations. Each destination independently chooses its
predecessor for retention, excitation, inhibition and toggling.

Source: `crates/flat-algebraic-attention/src/boolean_neuron_graph.rs`.
Executable: `crates/flat-algebraic-attention/examples/bps16_neuron_graph_smoke.rs`.

## Exact topology semantics

For destination neuron i and operator k, define shift s[k,i] in [0,63].
That neuron reads state bit (i-s[k,i]) mod 64 from the **previous** tick.
Its four contributions are input-gated and combined using the same fixed
order as BPS-15:

```
base[i] = retain_source[i] OR excite_source[i] OR direct_write[i]
next[i] = (base[i] AND NOT inhibit_source[i]) XOR toggle_source[i]
output[i] = next[i] AND output_gate[i]
```

All neurons update synchronously, never mixing old and new state within
the same tick. Gate truth tables remain input-only and are not learned
by the initial BPS-16 trainer. Per-neuron predecessor indices are
trainable via bounded, strict-improvement full-trajectory Hamming loss.
Held-out labels are not consulted during update selection.

## Compiled execution: group equal shifts

The independent `step_scalar` oracle reads the predecessor for each
output bit and channel explicitly (up to 256 source-bit reads/tick).
`step_grouped` instead compiles for each operator a table of 64
destination bitmasks grouped by equal source shift. Each active shift
requires one `rotate_left(u64)` and `AND`, then groups are OR-reduced
and the same gate/operation order applied.

- One shared shift per operator maps to **exactly BPS-15**; tested
  across randomized input gates and trajectories.
- Arbitrary per-neuron shifts are bit-exact with the independent scalar
  reference across randomized 64-bit state/input and topology.
- With U distinct shifts among the destinations of one operator, its
  grouping requires U rotations per tick rather than one shared BPS-15
  rotation; U is in [1,64]. Across four operations the upper bound is
  256 group rotations/tick, with sparse gates able to skip a channel.
- `step_grouped_sparse` skips **all** source-group rotations when an
  operation's full 64-bit gate word is zero, preserving exactness.
- The compiler does NOT guarantee SIMD or 256-way CPU parallelism:
  instructions in the reference are sequential. Grouping can reduce
  duplicated rotations, or increase work for very diverse topologies.

There is no generally valid compressed constant-size BPS-13-style
associative temporal scan of this multi-source update. We do not claim
sublinear state reads, increased throughput, efficiency or convergence
without independently measured evidence.

## Falsifiable qualification and next stage

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_neuron_graph
cargo run -p flat-algebraic-attention --example bps16_neuron_graph_smoke
```

The new CI gate executes unit tests and the smoke example on every
relevant HEAD. The 2-source synthetic teaching task learns separate
links for destination lanes 1 and 3 and reports training and held-out
Hamming losses; it is **not** evidence of general language modeling
or classical LSTM quality.

The next paired CPU benchmark must use an identical BPS-15 shared
topology between BPS-15 dense/sparse and BPS-16 grouped/sparse as its
**exact-output** comparable arm. Independent per-neuron topology may
need more work; measure its expressivity separately from throughput.
All Dell/Thor/Mac physical work must run only under
`Memorithm/RemoteOps` with resource admission, pinned main,
read-only host policy and retained measured evidence. No production
service or hardware policy changes are authorized.
