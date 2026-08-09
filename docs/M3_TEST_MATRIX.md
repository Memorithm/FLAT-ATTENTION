# M3 Test Matrix

- Head dimensions: 1, 8, 16, 32, 64, 80, 96, 128.
- Sequence boundaries: 1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129.
- Modes: causal and non-causal.
- Additional coverage: batch=2, heads=3, adversarial score ranges, causal future-token isolation, non-finite input rejection, unsupported head dimension rejection, packed resident output.
