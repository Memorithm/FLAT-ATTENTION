# PVP transfer diagnostic D1
Prospective diagnostic after rejected R3 correctness, not a timing campaign.
Standalone Rust/WGPU/Vulkan, no compute shader, 4 MiB buffers, fixed CPU pattern.
Six cells: upload flush false/true crossed with staging drop/reuse/retain.
Thirty fresh pattern rounds per cell. Retain failed-round count and first mismatch
without giant vector dumps. All six cells run regardless of observed correctness;
no performance sample or gain acceptance. Same operator-controlled reservation,
R3 kernel-lifecycle attribution, guarded service pause and independent recovery.
Source SHA-256 b750701ee1b309d7d12d1c14aa91421c8070b9eb70b8864b8aed9ab7a2ee4ad3;
binary SHA-256 4ff93dfc8188cfab43f99ff5b6ed25ae19ae98185880f9bcd65663ad9d3a528b.
The failed R3 left/right comparison used the pinned Rust CPU oracle:
current expected vector matches; 16864 wrong words all match the corresponding
previous geometry's expected words. This is evidence of stale content, not yet
proof of which transfer stage or driver layer caused it.
