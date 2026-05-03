"""v7: encoder-side basis selectors that predict end-to-end fidelity better than k*(0.9).

v6 used `k_star_in_basis(psi, 0.9, b)` as the encoder-side score: pick the basis with
smallest k*(0.9). That works for TFIM (matches oracle) but misses Heisenberg by 1.05 pp
(oracle picks DCT despite worse k*(0.9)) and destroys Clifford F=1 by routing away from
WHT when k*(0.9) is high-variance.

v7 tests four alternative selectors:
- `select_kstar_50`         — smallest k*(0.5); favors bases where the *top* is most concentrated
- `select_topk_mass`        — highest L2 mass in the top-k=8 entries; the natural top-k predictor
- `select_quant_probe`      — cheap probe: run the pipeline minus QJL in each basis, pick highest F
- `select_hybrid`           — degeneracy detector first (defaults to WHT for stabilizer-like states),
                              then quant_probe on the remainder

Each takes (psi, bases) and returns the chosen basis name.
"""

from __future__ import annotations

import numpy as np

import pipeline as pl
import concentration_basis as cb


# ---------------------------------------------------------------------------
# Cheap concentration-style selectors
# ---------------------------------------------------------------------------

def select_kstar_50(psi: np.ndarray,
                     bases: tuple[str, ...] = cb.SUPPORTED_BASES) -> str:
    """Smallest k*(0.5) wins (top-of-tail concentration)."""
    best_basis, best_k = None, None
    for b in bases:
        k = cb.k_star_in_basis(psi, 0.5, b)
        if best_k is None or k < best_k:
            best_k, best_basis = k, b
    return best_basis


def select_topk_mass(psi: np.ndarray,
                      bases: tuple[str, ...] = cb.SUPPORTED_BASES,
                      k: int = 8) -> str:
    """Highest L2 mass in the top-k entries wins (direct top-k predictor)."""
    best_basis, best_mass = None, -1.0
    for b in bases:
        coeffs = cb.transform_in_basis(psi, b)
        mags2 = np.abs(coeffs) ** 2
        mass = float(np.sort(mags2)[::-1][:k].sum())
        if mass > best_mass:
            best_mass, best_basis = mass, b
    return best_basis


# ---------------------------------------------------------------------------
# Cheap simulated-quantization probe
# ---------------------------------------------------------------------------

def select_quant_probe(psi: np.ndarray,
                        bases: tuple[str, ...] = cb.SUPPORTED_BASES,
                        bits_mag: int = 3, bits_phase: int = 3,
                        mag_quantizer: str = "topk", topk: int = 8) -> str:
    """Pick basis by running the pipeline at the target bit budget *minus* QJL
    in each candidate basis, and picking the highest reconstructed fidelity.

    QJL is skipped because (a) it's the most expensive stage (scipy.optimize per
    call), and (b) it's a residual correction whose contribution is approximately
    basis-independent for small-residual cases. The probe uses the `wht_mag_phase`
    config — same Lloyd-Max + phase quantization the full pipeline uses, but no
    QJL stage. After picking the basis, the caller runs the FULL pipeline (with
    QJL) in that basis once for actual encoding.
    """
    best_basis, best_fid = None, -1.0
    for b in bases:
        p = pl.compress(psi, "wht_mag_phase", bits_mag=bits_mag, bits_phase=bits_phase,
                        mag_quantizer=mag_quantizer, topk=topk, basis=b)
        f = pl.fidelity(psi, pl.decompress(p))
        if f > best_fid:
            best_fid, best_basis = f, b
    return best_basis


# ---------------------------------------------------------------------------
# Hybrid: degeneracy detector + quant_probe
# ---------------------------------------------------------------------------

def is_magnitude_degenerate(psi: np.ndarray, basis: str = "wht",
                             mag_tol: int = 6, frac_threshold: float = 0.05) -> bool:
    """True if the magnitude distribution is dominated by very few distinct values
    in the named basis -- a stabilizer-state-like signature.

    `mag_tol` rounds magnitudes to that many decimals before counting unique values.
    `frac_threshold` is the cutoff: if (n_unique_mags / d) < threshold, declare
    degenerate. For Clifford in WHT, n_unique = 1; ratio = 1/1024 ~= 0.001.
    For non-degenerate states, ratio ~ 1.0.
    """
    coeffs = cb.transform_in_basis(psi, basis)
    nz_mags = np.abs(coeffs)
    nz_mags = nz_mags[nz_mags > 1e-10]
    if nz_mags.size == 0:
        return False
    n_unique = len(np.unique(np.round(nz_mags, mag_tol)))
    return (n_unique / len(coeffs)) < frac_threshold


def select_hybrid(psi: np.ndarray,
                   bases: tuple[str, ...] = cb.SUPPORTED_BASES,
                   bits_mag: int = 3, bits_phase: int = 3,
                   mag_quantizer: str = "topk", topk: int = 8) -> str:
    """If WHT magnitudes are highly degenerate (Clifford-like), default to WHT.
    Otherwise use the quant_probe."""
    if is_magnitude_degenerate(psi, basis="wht"):
        return "wht"
    return select_quant_probe(psi, bases=bases, bits_mag=bits_mag,
                               bits_phase=bits_phase,
                               mag_quantizer=mag_quantizer, topk=topk)


SELECTORS = {
    "kstar_50": select_kstar_50,
    "topk_mass": select_topk_mass,
    "quant_probe": select_quant_probe,
    "hybrid": select_hybrid,
}
