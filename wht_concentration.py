"""WHT-basis L2-mass concentration: the v4 predictor.

For a unit-norm complex statevector, define k*(p) as the smallest k such that
the top-k magnitudes (after a normalized Walsh-Hadamard transform) hold at
least fraction p of the L2 norm. Small k* means the post-WHT representation
is dominated by a few outliers; large k* means the energy is spread out.

v3 hypothesis: k* (at p ~ 0.9) tightly predicts the fidelity that top-k exact
preservation can recover at a fixed bit budget. State classes with small k*
(TFIM, Heisenberg, Clifford) compress well; state classes with large k*
(Haar, random MPS at any chi) do not.
"""

from __future__ import annotations

import numpy as np

import pipeline as pl


def k_star(psi: np.ndarray, threshold: float) -> int:
    """Smallest k such that the top-k post-WHT magnitudes hold at least
    `threshold` fraction of the L2 norm.

    `psi` must be unit-norm. `threshold` in (0, 1]. Returns an integer in
    [1, len(psi)]; for unit-norm input a threshold of 1.0 returns len(psi)
    only if no leading prefix already sums to 1.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    transformed = pl.wht(psi)
    mags2 = np.abs(transformed) ** 2
    sorted_desc = np.sort(mags2)[::-1]
    cumulative = np.cumsum(sorted_desc)
    # The total may differ from 1 by numerical noise; renormalize for the
    # threshold comparison so threshold=1.0 doesn't always return len(psi).
    total = float(cumulative[-1])
    if total <= 0:
        return len(psi)
    idx = int(np.searchsorted(cumulative, threshold * total, side="left"))
    return idx + 1  # 1-indexed: k=1 means "top single entry already holds >= p"


def k_star_profile(psi: np.ndarray,
                    thresholds: tuple[float, ...] = (0.5, 0.9, 0.99)
                    ) -> dict[float, int]:
    """k* at multiple thresholds in one WHT pass."""
    transformed = pl.wht(psi)
    mags2 = np.abs(transformed) ** 2
    sorted_desc = np.sort(mags2)[::-1]
    cumulative = np.cumsum(sorted_desc)
    total = float(cumulative[-1])
    if total <= 0:
        return {p: len(psi) for p in thresholds}
    out = {}
    for p in thresholds:
        idx = int(np.searchsorted(cumulative, p * total, side="left"))
        out[p] = idx + 1
    return out
