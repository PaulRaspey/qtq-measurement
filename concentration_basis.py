"""Generalized k*(p) — measure WHT-basis L2-mass concentration in arbitrary orthogonal bases.

v4 measured concentration only in the Walsh-Hadamard basis (the basis the pipeline already uses).
v5 asks whether the bulk regime — random MPS at any chi sitting on top of Haar at F~0.955 —
is a property of those state classes intrinsically, or only of their interaction with WHT.

Three bases tested:
- "wht":      Normalized Walsh-Hadamard transform (the pipeline default).
- "dct":      DCT-II with norm='ortho'; another real orthogonal transform but with smooth-
              basis vectors instead of Walsh-Hadamard's binary structure.
- "identity": No transform — measure k*(p) on the raw computational-basis amplitudes.

All three are orthogonal/unitary, so they preserve L2 norm. For Haar-random states k*(p)
should be statistically identical across all three bases (Haar measure is unitarily invariant).
For state classes with structure that aligns with a particular basis, k*(p) should be smaller
in that basis.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct

import pipeline as pl


SUPPORTED_BASES = ("wht", "dct", "identity")


def transform_in_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    """Apply the named orthogonal transform to a complex statevector."""
    if basis == "identity":
        return psi.astype(np.complex128, copy=True)
    if basis == "wht":
        return pl.wht(psi)
    if basis == "dct":
        # DCT-II with norm='ortho' is orthogonal; apply to real and imag separately.
        real = dct(np.real(psi), type=2, norm="ortho")
        imag = dct(np.imag(psi), type=2, norm="ortho")
        return (real + 1j * imag).astype(np.complex128)
    raise ValueError(f"unknown basis {basis!r}; expected one of {SUPPORTED_BASES}")


def inverse_transform_in_basis(coeffs: np.ndarray, basis: str) -> np.ndarray:
    """Apply the inverse of `transform_in_basis(_, basis)` to coefficients."""
    if basis == "identity":
        return coeffs.astype(np.complex128, copy=True)
    if basis == "wht":
        # Normalized WHT is involutive.
        return pl.iwht(coeffs)
    if basis == "dct":
        # idct with type=2, norm='ortho' inverts dct(type=2, norm='ortho').
        from scipy.fft import idct
        real = idct(np.real(coeffs), type=2, norm="ortho")
        imag = idct(np.imag(coeffs), type=2, norm="ortho")
        return (real + 1j * imag).astype(np.complex128)
    raise ValueError(f"unknown basis {basis!r}; expected one of {SUPPORTED_BASES}")


def k_star_in_basis(psi: np.ndarray, threshold: float, basis: str) -> int:
    """Smallest k such that the top-k magnitudes of `psi` (in the named orthogonal
    basis) hold at least `threshold` fraction of the L2 norm.
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    coeffs = transform_in_basis(psi, basis)
    mags2 = np.abs(coeffs) ** 2
    sorted_desc = np.sort(mags2)[::-1]
    cumulative = np.cumsum(sorted_desc)
    total = float(cumulative[-1])
    if total <= 0:
        return len(psi)
    idx = int(np.searchsorted(cumulative, threshold * total, side="left"))
    return idx + 1


def k_star_profile_all_bases(psi: np.ndarray,
                              thresholds: tuple[float, ...] = (0.5, 0.9, 0.99)
                              ) -> dict[tuple[str, float], int]:
    """k*(p) in every supported basis at every requested threshold."""
    out: dict[tuple[str, float], int] = {}
    for basis in SUPPORTED_BASES:
        coeffs = transform_in_basis(psi, basis)
        mags2 = np.abs(coeffs) ** 2
        sorted_desc = np.sort(mags2)[::-1]
        cumulative = np.cumsum(sorted_desc)
        total = float(cumulative[-1])
        for p in thresholds:
            if total <= 0:
                out[(basis, p)] = len(psi)
            else:
                idx = int(np.searchsorted(cumulative, p * total, side="left"))
                out[(basis, p)] = idx + 1
    return out
