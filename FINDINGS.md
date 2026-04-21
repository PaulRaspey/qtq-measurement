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

## v2 — outlier-aware magnitude quantizers (n = 20 / cell, full pipeline)

Swapping the magnitude stage flipped the result. At the default 3/3 budget, **TFIM** went from F = 0.804
(Lloyd-Max baseline) to **F = 0.989 ± 0.001** with top-k (k = 8, 8.40× ratio) and **F = 0.986 ± 0.001**
with percentile-clip (8.30× ratio) — both **above** Haar's F = 0.957 ± 0.001 at 8.75×, confirming the
v1 diagnosis: the physical-state advantage was real but masked by an MSE-vs-tail mismatch in plain
Lloyd-Max. The best fidelity/compression point in the entire study is TFIM with **top-k k = 32** at
3/3: F = 0.9986 ± 0.0001 at 7.50× — essentially lossless. Haar and MPS were within ±0.005 fidelity
across all four quantizers at every bit budget (i.e., outlier-aware machinery neither helps nor hurts
when the distribution has no outliers; the small overhead from top-k positions / percentile metadata
trades a few percent of compression ratio for unchanged fidelity). **Log-domain Lloyd-Max degraded
TFIM** to F = 0.594 ± 0.010 at 3/3 (and to 0.420 at 2/2) because log-MSE is equivalent to
relative-MSE: it spends centroid budget on the tens-of-near-zero entries that contribute negligibly
to fidelity while still under-resolving the L2-mass-bearing top coefficient. Top-k k-sweep on TFIM
shows clean monotonic gains: F = 0.978 (k = 4), 0.989 (k = 8), 0.995 (k = 16), 0.999 (k = 32);
the same sweep on Haar moves F by < 0.001, consistent with Haar having no localized L2-mass to
capture. Net: under top-k or percentile, this pipeline does have a legitimate physical-state
compression story — TFIM beats Haar by ~3 percentage points of fidelity at the same bit budget,
or reaches near-lossless reconstruction at compression ratios where Haar caps at ~0.96.

