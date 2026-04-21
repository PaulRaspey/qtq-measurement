# qtq-measurement

Research spike measuring whether a four-stage compression pipeline
(**WHT → magnitude PolarQuant → phase quantization → QJL-analog 1-bit residual**)
produces a fidelity/compression-ratio asymmetry between physical statevectors and
Haar-random statevectors at d = 1024.

This is a measurement spike, not a production library. See `FINDINGS.md` for a
one-paragraph observation of what we measured.

## Layout

| File                    | What it is                                                          |
|-------------------------|---------------------------------------------------------------------|
| `pipeline.py`           | Four pipeline stages with encode/decode + bit accounting            |
| `states.py`             | Three state generators: Haar-random, TFIM critical GS, random MPS   |
| `test_pipeline.py`      | Lightweight sanity checks (no pytest dependency, just `python ...`) |
| `measurement.ipynb`     | Experiment grid (3 × 4 × 3 × 20 = 720 measurements)                 |
| `results.csv`           | Raw results                                                         |
| `figures/*.png`         | Fidelity-vs-ratio plot per state class (default 3/3 bit budget)     |
| `FINDINGS.md`           | One-paragraph observation                                           |

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy matplotlib jupyter pandas

# Sanity checks (~3 s)
.venv/Scripts/python test_pipeline.py

# Full measurement (~1-2 min on a workstation)
.venv/Scripts/jupyter nbconvert --to notebook --execute measurement.ipynb --inplace --ExecutePreprocessor.timeout=600
```

`results.csv` and `figures/*.png` are regenerated from the notebook.

## Pipeline notes

- **WHT** is the normalized (orthogonal, symmetric) Walsh–Hadamard transform; `iwht == wht`.
- **Magnitude PolarQuant** trains a per-vector Lloyd-Max codebook (initialized at quantiles).
  The codebook (8 floats at 3 bits per magnitude) is included in the bit-budget accounting.
- **Phase quantization** is uniform on [0, 2π) with bin midpoints as decoded values.
- **QJL residual** projects the residual through a complex Gaussian Johnson–Lindenstrauss
  matrix (m = d, seeded), takes 1-bit signs of the real part, and sends an `alpha` scalar.
  The encoder picks `alpha` to maximize fidelity (gauge-invariant); if no `alpha` improves
  fidelity over the intermediate reconstruction, `alpha = 0` and the correction is a no-op.

## Bit accounting

Original: 2 × d × 32 = 65,536 bits at d = 1024.
Compressed (3/3 full): 256 (codebook) + 1024 × 3 (mag idx) + 1024 × 3 (phase idx) + 1024 (QJL signs) + 32 (α) + 32 (seed) = 7,488 bits → 8.75×.

## Out of scope

- Additional state classes
- Hyperparameter tuning beyond the three bit budgets {2/2, 3/3, 4/4}
- Pipeline variations (alternative quantizers, different JL m, etc.)
- Interpretation of findings beyond the one-paragraph observation
