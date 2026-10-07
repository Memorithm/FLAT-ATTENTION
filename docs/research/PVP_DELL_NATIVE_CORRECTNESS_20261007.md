# Native Dell prepared-plan correctness — 7 October 2026

Verdict: three complete, exact native Vulkan processes; no performance admission.

[RemoteOps #301](https://github.com/Memorithm/RemoteOps/pull/301) added the read-only
readiness gate; [run 37635323320](https://github.com/Memorithm/RemoteOps/actions/runs/37635323320)
found native x86_64 host debian, driver 595.45.04 and memci without a Rust PATH or
render-node access. [RemoteOps #303](https://github.com/Memorithm/RemoteOps/pull/303)
merged after exact-head CI success and froze the bounded native protocol.
[Native run 37635882086](https://github.com/Memorithm/RemoteOps/actions/runs/37635882086)
completed successfully on dell-t430-remoteops-x64-01.

Source: b020d31711d75807eb0f8b13a6a967a8522c5d1c. Existing prepared-plan kernels
and fixture semantics are unchanged. Compiler: Rust 1.89.0, LLVM 20.1.7, native
x86_64, release profile; binary SHA256 fd6525068b492ccfba215b3941389b47be5d64eaedb8bf816edc35a3658a2ac5.
Every process records NVIDIA GeForce RTX 4060, DiscreteGpu, Vulkan, driver
595.45.04. Software adapters or changed identities are rejected.

| Coverage | Each process | Three processes |
|---|---:|---:|
| Cases: shape × bank × coefficient round × arm | 720 | 2160 |
| Complete paired phase comparisons | 6480 | 19440 |
| Failed comparisons | 0 | 0 |

The sixteen frozen geometries include (16384,2048), (65536,512) and
(262144,128), plus narrow and padding-sensitive cases. Five arms: vec4, fused2,
tile8, packed1, packed7; boundary/sparse4/dense32 banks, three changed coefficient
rounds. Uploaded source, transformed truth and inverse recovery each use two
independent explicit MAP_READ buffers, compare A/oracle, B/oracle and A/B, and
include all words, padding and four outside-binding sentinels.

RemoteOps staged an existing toolchain and ran cargo/GPU code as memci in a
transient systemd unit, with scoped supplementary render group, NoNewPrivileges,
read-only system/home, private temporary namespace, one writable task directory,
12 GiB RAM/four CPU-equivalent limits and a 40-minute lifetime limit. Runtime
identity confirms uid 984 and supplementary gid 105(render). No account group
change, service pause, NVIDIA SDK/management command or root GPU compute occurs
in the retained controller. This is process hardening, not hostile-code isolation.

Each numbered process independently obtained and released a cooperative device
lock. All process exit codes are 0; source/lockfile/binary identities remain
unchanged. The lock is not continuous idle/exclusive admission. No device-user
occupancy campaign or timing measurement was performed; accepted performance
rows remain zero. These release-profile Dell correctness results cannot be pooled
with Thor's debug-profile prepared qualification.

Evidence: [capture](evidence/pvp-dell-native-20261007/capture.tar.xz), SHA256
0a195ca81d98bac960a12ecf1c71eb952dc686cddce3944228ff29bad1f990b8, 25052 bytes, 28 retained files.
[File hashes](evidence/pvp-dell-native-20261007/FILES.json) and the independent
[standalone Rust verifier](evidence/pvp-dell-native-20261007/verify.rs) check
source/device identity, exact record-key coverage, lengths, wrong-word counts,
completion and process exits. Seven synthetic verifier tests cover valid coverage
and wrong source/device/words/length, duplicate and missing records. Those tests
are parser checks, not additional GPU experiments. Frozen script/protocol hashes
were recomputed from the downloaded bytes; all raw captures remain unedited.

This closes a bounded five-arm functional gate on native Dell hardware. A frozen
Dell timing/occupancy study, independently reviewed admission, SML-owned ANF
banks and destination requalification remain open. No speed, bandwidth, register
allocation, model quality, attention replacement or default-route claim follows.
SML remains self-sufficient; FLAT, SciRust and NNIS remain optional partners.
