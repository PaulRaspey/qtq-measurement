# Findings — Quantum-TurboQuant measurement spike (d = 1024, n = 20 samples / cell)

At the default 3-bit-per-magnitude / 3-bit-per-phase budget, the **full four-stage pipeline** achieved
F = 0.957 ± 0.001 on Haar-random states (8.75× compression),
F = 0.954 ± 0.003 on random MPS with χ = 16 (8.75×), and
F = 0.804 ± 0.007 on the 10-qubit transverse-field Ising critical ground state (8.75×). The MPS-at-χ=16
class behaved indistinguishably from Haar across all three bit budgets (2/2, 3/3, 4/4) and all four
configurations, indicating that χ = 16 on 10 sites already produces effectively unstructured statevectors
under this pipeline. The TFIM ground state did **not** compress better than Haar — it compressed
substantially worse — but the bottleneck localizes to magnitude PolarQuant: TFIM's `wht_mag` and
`wht_mag_phase` fidelities are identical to four decimal places (0.7570 / 0.7570 at 3/3), so phase
quantization adds zero error on the bimodal-phase state, while Lloyd-Max with 8 levels handles the
heavy-tailed post-WHT magnitude distribution poorly (one coefficient at 0.682 vs. median 5.3e-4 → 66%
relative error on the largest coefficient). The phase-distribution asymmetry from the original hypothesis
**is** observable in isolation (TFIM loses 0% from phase quant; Haar loses ~5% absolute fidelity going
from `wht_mag` to `wht_mag_phase` at 3 bits), but it is dominated in this pipeline by an MSE-vs-tail
mismatch in the magnitude stage. QJL residual correction recovered ~1.3% absolute fidelity for Haar/MPS
at 3/3 and ~4.7% for TFIM at 3/3; on the basis-state sanity check it correctly produced F ≈ 1.0 by
zeroing α (no-harm behavior).
