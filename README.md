# qtq-measurement

Empirical measurement study of a four-stage compression pipeline
(**WHT → magnitude PolarQuant → phase quantization → QJL-analog 1-bit residual**)
applied to quantum statevectors at d = 1024 (10 qubits).

The repo is a snapshot of three measurement spikes — v1, v2, v3 — that progressively
narrow the question of *what kind of structure this pipeline actually exploits*.
See `FINDINGS.md` for the running observations; the v3 paragraph is the current
headline.

## Headline finding (v3)

At 3 bits per magnitude / 3 bits per phase, full pipeline, n=20 samples per cell:

| State class                  | Lloyd-Max F        | Top-k (k=8) F      |
|------------------------------|--------------------|--------------------|
| Haar-random                  | 0.9572             | 0.9582             |
| Random MPS (χ = 2 / 4 / 16)  | 0.949 – 0.954     | 0.954 – 0.957     |
| Heisenberg AFM ground state  | 0.9695             | **0.9805**         |
| TFIM critical ground state   | 0.8042             | **0.9890**         |
| Random Clifford state        | 1.0000             | **1.0000**         |

The pipeline detects *L2-mass localization in the Walsh–Hadamard basis*. Local-Hamiltonian
critical ground states (TFIM, Heisenberg) have it via heavy-tailed magnitudes; stabilizer
states (Clifford) have an extreme version of it via post-WHT degeneracy; random MPS at any
tested bond dimension does not have it. So *physicality* is not the right axis for predicting
compressibility — *WHT-basis L2-mass concentration* is.

## Layout

| File                                | What it is                                                                  |
|-------------------------------------|-----------------------------------------------------------------------------|
| `pipeline.py`                       | Four pipeline stages with encode/decode + bit accounting (v1 + v2 quantizers) |
| `states.py`                         | Seven state generators: Haar, TFIM GS, Heisenberg GS, MPS at χ ∈ {2, 4, 16}, Clifford |
| `test_pipeline.py`                  | v1 + v2 sanity checks (no pytest dependency, just `python ...`)             |
| `test_states.py`                    | v3 state-generator sanity checks (Heisenberg energy, Clifford amplitudes, MPS shapes) |
| `measurement.ipynb`                 | v1 grid (3 states × 4 configs × 3 budgets × 20 samples = 720 measurements)  |
| `measurement_v2.ipynb`              | v2 grid (3 states × 4 quantizers × 3 budgets × 20 + topk k-sweep = 960)     |
| `measurement_v3.ipynb`              | v3 grid (4 new states × 4 quantizers × 3 budgets × 20 = 960)                |
| `results.csv` / `results_v2.csv` / `results_v3.csv` | Raw per-cell measurements                                  |
| `figures/*.png`                     | Per-state quantizer-comparison plots (v1, v2, v3) + cross-class comparison  |
| `FINDINGS.md`                       | One-paragraph observation per spike (v1, v2, v3)                            |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy matplotlib jupyter pandas

# Sanity checks (~3 s)
.venv/Scripts/python test_pipeline.py
.venv/Scripts/python test_states.py

# Full measurement (~1-2 min per notebook on a workstation)
.venv/Scripts/jupyter nbconvert --to notebook --execute measurement.ipynb     --inplace --ExecutePreprocessor.timeout=600
.venv/Scripts/jupyter nbconvert --to notebook --execute measurement_v2.ipynb  --inplace --ExecutePreprocessor.timeout=900
.venv/Scripts/jupyter nbconvert --to notebook --execute measurement_v3.ipynb  --inplace --ExecutePreprocessor.timeout=900
```

`results*.csv` and `figures/*.png` are regenerated from the notebooks.

## Pipeline notes

- **WHT** is the normalized (orthogonal, symmetric) Walsh–Hadamard transform; `iwht == wht`.
- **Magnitude PolarQuant** has four selectable quantizers (`pipeline.MAG_QUANTIZERS`):
  - `lloyd` — per-vector Lloyd-Max codebook initialized at quantiles (v1 baseline)
  - `topk` — k largest magnitudes stored exact, rest Lloyd-Max'd (v2; default k = 8)
  - `log` — Lloyd-Max in log-magnitude space (v2)
  - `percentile` — outliers above the 99th percentile stored exact, bulk-only Lloyd-Max (v2)
- **Phase quantization** is uniform on [0, 2π) with bin midpoints as decoded values.
- **QJL residual** projects the residual through a complex Gaussian Johnson–Lindenstrauss
  matrix (m = d, seeded), takes 1-bit signs of the real part, and sends an `alpha` scalar.
  The encoder picks `alpha` to maximize fidelity (gauge-invariant); if no `alpha` improves
  fidelity over the intermediate reconstruction, `alpha = 0` and the correction is a no-op.

## Bit accounting (3/3 full pipeline, d = 1024)

Original: 2 × d × 32 = 65,536 bits.
Compressed (Lloyd-Max baseline): 256 (codebook) + 1024 × 3 (mag idx) + 1024 × 3 (phase idx)
+ 1024 (QJL signs) + 32 (α) + 32 (seed) = 7,488 bits → **8.75×**.
Top-k (k = 8): adds 8 × 10 (positions) + 8 × 32 (exact values) − 8 × 3 (saved bulk indices)
= ~7,808 bits → **8.40×**, slightly more overhead in exchange for accurate top-of-tail.

## Out of scope (still)

- System sizes other than 10 qubits (d = 1024)
- Adaptive or learned codebooks (the Lloyd-Max codebook is per-vector but not amortized)
- Pipeline variations beyond the four magnitude quantizers and the {2/2, 3/3, 4/4} budgets
- Theoretical analysis of *why* WHT-basis L2-mass concentration tracks the observed
  fidelity gradient — v3 identifies this as the operative variable but does not derive it

## License

MIT — see `LICENSE`.
