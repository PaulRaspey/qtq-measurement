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

## v3 — robustness sweep across physical state classes (n = 20 / cell, full pipeline, 3/3 bits, top-k k=8)

Extending v2 to four new state classes (10-site OBC Heisenberg AFM ground state; random MPS at χ=2 and
χ=4; random Clifford-circuit state, depth 200) sharpens the physical-state claim. Combined results
at 3/3 with top-k: **Haar 0.9582**, MPS-χ=16 0.9572, MPS-χ=4 0.9543, MPS-χ=2 0.9551, **Heisenberg
0.9805**, **TFIM 0.9890**, **Clifford 1.0000** (each ± 0.001–0.004 across n=20). Three findings:
**(1) Heisenberg generalizes the v2 result**: critical-point local-Hamiltonian ground states do
compress better than Haar under top-k (+2.2 pp), confirming the v2 asymmetry is not TFIM-specific —
but the route differs. Heisenberg's vanilla Lloyd-Max baseline is already 0.9695 (vs TFIM 0.8042),
so the post-WHT magnitude tail is much milder; top-k uplifts by only +0.011 here vs +0.185 for TFIM.
**(2) Lower-χ MPS does not recover any structure advantage** — χ=2, χ=4, χ=16 all sit indistinguishably
within the 0.95 Haar band under every quantizer, even though χ=2 has minimal entanglement. The top-k
uplift increases monotonically as χ shrinks (Δ = 0.003 → 0.005 → 0.006 for χ=16/4/2), but the
absolute fidelity does not improve over Haar at any χ. Random MPS being structured *as an MPS*
does not imply heavy-tailed magnitudes in the WHT basis. Bond dimension is the wrong axis for
predicting WHT-basis compressibility. **(3) Clifford reconstructs to F = 1.0000 across all four
quantizers, including baseline Lloyd-Max.** A stabilizer-prepared state has only one nonzero
magnitude value (1/√2^k for support size 2^k); post-WHT it remains so degenerate that any reasonable
codebook is exact. Outlier-aware machinery is irrelevant here. Net read: the v2 claim "physical
states compress better than random states with outlier-aware quantization" was correct but
conflated two different mechanisms. The pipeline detects *L2-mass localization in the
Walsh-Hadamard basis*. TFIM and Heisenberg have it (heavy tail → top-k essential). Clifford has
an extreme version of it (full degeneracy → trivial compression). Random MPS does not have it at
any tested χ. So "physicality" is not the relevant axis; "WHT-basis L2-mass concentration" is, and
that is satisfied by some but not all physical state classes.
