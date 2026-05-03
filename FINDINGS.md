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

## v4 — direct measurement of the predictor (n = 20 / cell, k* at 90% L2 mass)

v3 named *"WHT-basis L2-mass concentration"* as the operative variable; v4 measures it.
For each state class we compute k\*(0.9) — the smallest number of post-WHT magnitudes whose
squared sum reaches 90% of the L2 norm — and correlate it against the v3 top-k fidelity at
3/3 bits. Results: TFIM 42→0.989, Heisenberg 56→0.981, MPS-χ=2 303→0.955, MPS-χ=4
376→0.954, MPS-χ=16 536→0.957, Haar 604→0.958, Clifford 542→1.000. The data does **not**
fit a single continuous predictor; instead k\* discriminates **three regimes**.
**(1) Heavy-tail regime (k\* < ~100):** TFIM and Heisenberg. Top-k recovers fidelity by
exact preservation of the small set of L2-mass-dominant outliers, and within this regime
k\* sharply tracks fidelity (though the regime contains only two examples). **(2) Bulk
regime (k\* ~ 300–600):** Haar, MPS at χ=2/4/16. All four cluster at F ≈ 0.954–0.958
regardless of k\* — Spearman ρ(k\*, F) = −0.486 (p = 0.33) across the six non-Clifford
classes. The strong-looking Pearson r = −0.836 (p = 0.038) is misleading: it is almost
entirely leverage from the two heavy-tail points. **(3) Degenerate regime (Clifford):**
k\* = 542 (mid-range) but F = 1.000 across every quantizer. Magnitude degeneracy makes
baseline Lloyd-Max exact; k\* is the wrong metric here. Net read: the v3 asymmetry framing
holds — physical critical-point ground states do compress dramatically better — but the
mechanism summary tightens to *two distinct routes to compressibility (heavy-tailed
post-WHT magnitudes, magnitude degeneracy), and a bulk regime where amplitude-quantization
pipelines provide no state-class advantage over Haar*. That is a more nuanced and harder-
to-attack claim than "k\* predicts fidelity." It also identifies the next experiment cleanly:
v5 should look for a second metric that discriminates within the bulk, or test whether a
non-WHT basis (DCT, identity, learned rotation) shifts random MPS out of the bulk regime.

## v5 — basis-dependence of L2-mass concentration (n = 20 / cell, k* at 90% L2 mass, three bases)

v3/v4 framed the operative variable as *"WHT-basis L2-mass concentration."* v5 tests whether
that framing is intrinsic to the state classes or specific to the WHT basis. Three orthogonal
bases at d=1024: Walsh-Hadamard (the pipeline default), DCT-II (ortho-normalized), and identity
(no transform). Three findings.
**(1) Haar invariance: confirmed.** k\*(0.9) = 604 / 605 / 600 across WHT / DCT / Identity (spread
0.8% of mean), as predicted by unitary invariance of the Haar measure.
**(2) The bulk regime survives the basis change.** Random MPS at every tested χ stays in the
bulk (k\* > 270) in every basis. Largest basis-induced reduction is MPS-χ=2 dropping from
k\*=303 (WHT) to k\*=274 (identity) — only ~10%, not enough to leave the bulk. The v3 negative
result on MPS therefore strengthens: random MPS does not have heavy-tailed magnitudes in *any*
of WHT, DCT, or the computational basis. Bond dimension is the wrong axis for predicting
amplitude-quantization compressibility under any standard orthogonal basis tested.
**(3) The "right basis" is state-class-specific and is not always WHT.** TFIM is *most*
concentrated in **DCT** (k\* = 27) versus WHT's 42 — a 36% reduction. Heisenberg ties at
k\* = 56 in both WHT and identity (DCT worse at 95). So the two critical-point ground states
have *different* basis preferences despite both being local-Hamiltonian ground states. Clifford
is most concentrated in WHT (542); k\* is larger in DCT (599) and identity (668), and the cross-
seed std of k\* collapses in DCT (33 vs 272 in WHT) — DCT averages out the variable support
size of random Clifford circuits.
**Net read.** The earlier "WHT-basis concentration" framing was *almost* right but slightly
imprecise. The accurate framing: **physical states have state-class-specific bases in which
their amplitudes are concentrated, while random states (Haar) and random MPS at any χ are
spread out in every standard orthogonal basis tested.** The pipeline's WHT step is a
reasonable default but not optimal; a basis-adaptive compressor would strictly beat the
current pipeline on TFIM, tie or marginally beat on Heisenberg / MPS-χ=2, and is the natural
v6 direction (encoder picks among {WHT, DCT, identity} per vector at the cost of ~2 extra
bits per state for the basis tag).

## v6 — basis-adaptive pipeline (n = 20 / cell, top-k k=8, 3/3 bits, three strategies)

v5 said the right basis depends on the state class. v6 actually exploits that: the encoder
picks the basis per vector and sends a 2-bit tag; the decoder reads the tag and inverts. Two
adaptive strategies are compared against the WHT-only baseline. **k\*-adaptive** (realistic):
compute k\*(0.9) in WHT, DCT, identity; pick the smallest. **Oracle**: try all three bases at
full pipeline cost; pick the one with highest decoded fidelity. Three outcomes:
**(1) Wins:** TFIM 0.9890 → 0.9947 (+0.57 pp, k\* matches oracle perfectly — both always pick
DCT). Bulk states (Haar, MPS at any χ) gain modestly under oracle (+0.1–0.3 pp); k\*-adaptive
is essentially neutral or slightly negative because the 2-bit tag overhead isn't recovered when
the basis change doesn't help. **(2) k\*(0.9) misses Heisenberg by 1.05 pp:** WHT 0.9804 →
k\*-adaptive 0.9804 → oracle **0.9909**. k\*(0.9) for Heisenberg is 56 in WHT, 56 in identity,
95 in DCT — so k\* picks WHT — but the oracle picks DCT every single seed and gets a +1.05 pp
lift. DCT spreads Heisenberg's energy across more entries (95 vs 56) but the *largest* entries
are relatively bigger, and that is what top-k actually exploits. Concentration-at-90% does not
capture top-of-the-tail shape. **(3) k\*(0.9) destroys Clifford F = 1:** WHT 1.0000 →
k\*-adaptive 0.9934 → oracle 1.0000. Clifford's WHT k\* has high cross-seed variance (std 272);
on some seeds the encoder routes to DCT or identity, breaking the magnitude-degeneracy
mechanism that makes Lloyd-Max exact. The F=1 mechanism is invisible to k\*. **Net read:** the
basis-adaptive idea is real (TFIM +0.57 pp banked, Heisenberg +1.05 pp on the table, MPS
+0.1–0.3 pp on the table); the encoder-side predictor needs to be smarter than k\*(0.9). v7
directions: (a) better encoder-side scoring (top-of-tail metric like k\*(0.5), or a small
simulated-quantization probe per basis); (b) hybrid score combining concentration with a
degeneracy detector so Clifford-like states default safely to WHT. The Heisenberg gap is the
larger immediate target than Clifford regression — Clifford can be defended by routing to WHT
when k\* is uncertain.
