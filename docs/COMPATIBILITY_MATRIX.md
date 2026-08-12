# FLAT-ATTENTION compatibility matrix

This matrix records correctness/portability qualification boundaries. It is not a performance table. A backend marked qualified means the corresponding gate executed the FLAT WGPU/WGSL implementation and parity checks; it does not imply equal speed across devices.

| Platform / backend | Evidence class | 1.0 candidate status | Boundary |
|---|---|---|---|
| Linux / Vulkan / Mesa lavapipe | software reference | qualified | Shader translation, pipeline/dispatch and numerical parity only; no physical-GPU performance claim. |
| Linux / Vulkan / NVIDIA Jetson Thor | physical NVIDIA | qualified for correctness on recorded exact candidate heads | Real NVIDIA Vulkan execution; benchmark claims require a separate idle-device M40 manifest. |
| Windows Server / D3D12 / Microsoft Basic Render Driver (WARP) | software D3D12 reference | qualified for M35 portability | Uses WGPU/DX12 and DXC; software-rendered timing is not performance evidence. |
| macOS / Metal | hosted Apple hardware | qualified for portability/correctness | Hardware identity and capability limits belong in exact workflow evidence; hosted timing is not promoted. |
| Linux / Vulkan / physical AMD | physical AMD | **pending** | Required by M37 before 1.0 release; no substitute device is accepted. |
| Linux/Windows / physical Intel GPU | physical Intel | **pending** | Required by M37 before 1.0 release; no substitute device is accepted. |

## Functional contract

The portable contract is intended to cover MHA/GQA/MQA, causal/non-causal attention, asymmetric Q/KV lengths, variable-length batches, RoPE/bias paths, resident and paged KV, chunked prefill, specialized decode, mixed precision where the backend exposes the required feature, and recomputation-based backward.

Backend capability limitations remain explicit. An optimized variant may fall back only to an already-qualified portable GPU variant when policy allows it; an explicitly requested GPU path must not silently substitute CPU execution.

## Integration status boundary

SciRust owns the surrounding model/tensor/runtime integration. Stable adapter and resident prefill integration have been qualified in SciRust. The 1.0 release checklist separately requires the final SciAgent resident decode/KV lifecycle gate on the intended FLAT release revision; this matrix does not mark that integration complete on behalf of SciRust.

## Performance boundary

No row in this file claims latency, throughput, bandwidth, memory efficiency, tokens/s, or speedup. Performance claims are accepted only from reproducible benchmark manifests tied to an exact commit, identified physical device, driver/backend, problem geometry, precision, warm-up count, measured iterations, and reported statistic.
