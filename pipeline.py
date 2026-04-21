"""Four-stage compression pipeline for complex statevectors in C^d (d a power of 2).

Stages:
    1. WHT (Walsh-Hadamard Transform), real/imag componentwise.
    2. Magnitude PolarQuant: Lloyd-Max quantization of magnitudes.
    3. Phase quantization: uniform 2^k bins on [0, 2pi).
    4. QJL-analog residual correction: 1-bit sign correction via complex JL.

All array I/O is numpy.complex128. Each stage exposes encode/decode and the
top-level `compress`/`decompress` helpers thread them according to a config
flag. `compressed_bits` reports the bit count for a given payload + config so
fidelity-vs-ratio plots use the same accounting everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Bit-budget bookkeeping
# ---------------------------------------------------------------------------

FLOAT_BITS = 32          # original real/imag stored as float32-equivalent
SCALE_BITS = 32          # any single scalar metadata (alpha, etc.)
SEED_BITS = 32           # PRNG seed for JL projection (shared encode/decode)


# ---------------------------------------------------------------------------
# Stage 1: Walsh-Hadamard Transform (vectorized, normalized, self-inverse)
# ---------------------------------------------------------------------------

def _fwht_real(a: np.ndarray) -> np.ndarray:
    """Vectorized normalized fast Walsh-Hadamard Transform on a real 1D array.

    Length must be a power of 2. The normalized WHT is orthogonal and
    symmetric, so it is its own inverse.
    """
    n = a.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"length must be a power of 2, got {n}")
    out = a.astype(np.float64, copy=True)
    h = 1
    while h < n:
        out = out.reshape(-1, 2, h)
        u = out[:, 0, :].copy()
        v = out[:, 1, :].copy()
        out[:, 0, :] = u + v
        out[:, 1, :] = u - v
        out = out.reshape(-1)
        h *= 2
    return out / np.sqrt(n)


def wht(x: np.ndarray) -> np.ndarray:
    """Apply the normalized WHT to a complex vector componentwise."""
    real = _fwht_real(np.real(x))
    imag = _fwht_real(np.imag(x))
    return (real + 1j * imag).astype(np.complex128)


def iwht(x: np.ndarray) -> np.ndarray:
    """Inverse WHT. Identical to `wht` because the normalized WHT is involutive."""
    return wht(x)


# ---------------------------------------------------------------------------
# Stage 2: Magnitude PolarQuant (Lloyd-Max-style)
# ---------------------------------------------------------------------------

def lloyd_max_codebook(values: np.ndarray, n_levels: int, max_iter: int = 30,
                       tol: float = 1e-9) -> np.ndarray:
    """Train a 1D Lloyd-Max quantizer on `values`.

    Returns sorted codebook of length `n_levels`. Initial centroids are placed
    at quantiles of `values` so the algorithm starts in a sensible regime.
    """
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_levels)
    qs = np.linspace(0.0, 1.0, n_levels + 2)[1:-1]
    centroids = np.quantile(values, qs)
    centroids = np.sort(np.unique(centroids))
    if centroids.size < n_levels:
        # Degenerate distribution (e.g., almost all magnitudes equal).
        pad = np.linspace(values.min(), values.max() + 1e-12, n_levels)
        centroids = np.unique(np.concatenate([centroids, pad]))[:n_levels]
        if centroids.size < n_levels:
            centroids = np.linspace(values.min(), values.max() + 1e-12, n_levels)
    for _ in range(max_iter):
        # Boundaries are midpoints between adjacent centroids.
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        idx = np.searchsorted(boundaries, values)
        new_centroids = centroids.copy()
        for i in range(n_levels):
            mask = idx == i
            if mask.any():
                new_centroids[i] = values[mask].mean()
        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def mag_quant_encode(magnitudes: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize positive magnitudes to `bits` bits using a per-vector Lloyd-Max codebook.

    Returns (indices, codebook). The codebook (size 2**bits) must be sent
    alongside the indices; the caller folds its bit cost into the budget.
    """
    n_levels = 1 << bits
    codebook = lloyd_max_codebook(magnitudes, n_levels)
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = np.searchsorted(boundaries, magnitudes).astype(np.int32)
    return indices, codebook


def mag_quant_decode(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return codebook[indices]


# ---------------------------------------------------------------------------
# Stage 3: Phase quantization (uniform on [0, 2*pi))
# ---------------------------------------------------------------------------

def phase_quant_encode(phases: np.ndarray, bits: int) -> np.ndarray:
    """Map phases in [0, 2*pi) to `bits`-bit uniform bin indices."""
    n_levels = 1 << bits
    step = 2.0 * np.pi / n_levels
    # Wrap into [0, 2*pi).
    p = np.mod(phases, 2.0 * np.pi)
    indices = np.floor(p / step).astype(np.int32)
    np.clip(indices, 0, n_levels - 1, out=indices)
    return indices


def phase_quant_decode(indices: np.ndarray, bits: int) -> np.ndarray:
    """Reconstruct phases at the midpoint of each bin."""
    n_levels = 1 << bits
    step = 2.0 * np.pi / n_levels
    return (indices.astype(np.float64) + 0.5) * step


# ---------------------------------------------------------------------------
# Stage 4: QJL-analog 1-bit residual correction
# ---------------------------------------------------------------------------

def _jl_matrix(m: int, d: int, seed: int) -> np.ndarray:
    """Random complex Gaussian JL matrix, shape (m, d), entries CN(0, 1/m)."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((m, d))
    imag = rng.standard_normal((m, d))
    return ((real + 1j * imag) / np.sqrt(2.0 * m)).astype(np.complex128)


def qjl_encode(psi: np.ndarray, psi_hat_intermediate: np.ndarray, m: int,
               seed: int) -> tuple[np.ndarray, float]:
    """Encode 1-bit sign correction; choose alpha to maximize *fidelity*.

    Optimizing for fidelity (a gauge-invariant quantity) rather than L2
    residual energy is important: when the intermediate reconstruction differs
    from psi only by a global phase, the L2 residual is non-zero but no scalar
    correction can reduce it without hurting fidelity. In that regime the
    fidelity-optimal alpha is 0 (skip the correction); in regimes with real
    state-vector mismatch the optimum is non-zero. The decoder only needs
    `(sign_bits, alpha)`.
    """
    from scipy.optimize import minimize_scalar  # local import keeps top-level light

    d = psi.shape[0]
    residual = psi - psi_hat_intermediate
    P = _jl_matrix(m, d, seed)
    y = P @ residual
    s = np.where(np.real(y) >= 0, 1.0, -1.0).astype(np.float64)
    r_hat = P.conj().T @ s.astype(np.complex128)

    def neg_fid(alpha: float) -> float:
        candidate = psi_hat_intermediate + alpha * r_hat
        norm = float(np.linalg.norm(candidate))
        if norm < 1e-15:
            return 0.0
        overlap = abs(np.vdot(psi, candidate)) ** 2
        return -float(overlap / (norm * norm))

    # Energy-optimal alpha as a sensible starting bracket.
    denom = float(np.real(np.vdot(r_hat, r_hat)))
    energy_alpha = (float(np.real(np.vdot(r_hat, residual))) / denom) if denom > 0 else 0.0
    bracket = max(2.0 * abs(energy_alpha), 1.0)
    result = minimize_scalar(neg_fid, bounds=(-bracket, bracket), method="bounded",
                             options={"xatol": 1e-6})
    best_alpha = float(result.x)
    if neg_fid(best_alpha) >= neg_fid(0.0):
        # QJL would not improve fidelity -- record alpha = 0 so the decoder
        # leaves the intermediate reconstruction alone.
        best_alpha = 0.0
    return s, best_alpha


def qjl_decode(sign_bits: np.ndarray, alpha: float, m: int, d: int,
               seed: int) -> np.ndarray:
    P = _jl_matrix(m, d, seed)
    r_hat = P.conj().T @ sign_bits.astype(np.complex128)
    return (alpha * r_hat).astype(np.complex128)


# ---------------------------------------------------------------------------
# Configurable pipeline
# ---------------------------------------------------------------------------

@dataclass
class Payload:
    """Container for a compressed statevector at any pipeline depth."""
    config: str
    d: int
    bits_mag: int
    bits_phase: int
    # WHT-only fallback (uncompressed complex post-WHT vector).
    wht_only: Optional[np.ndarray] = None
    # Magnitude stream + codebook.
    mag_indices: Optional[np.ndarray] = None
    mag_codebook: Optional[np.ndarray] = None
    # Untouched phases when phase stage is skipped.
    raw_phases: Optional[np.ndarray] = None
    # Phase indices when phase stage is on.
    phase_indices: Optional[np.ndarray] = None
    # QJL correction.
    qjl_signs: Optional[np.ndarray] = None
    qjl_alpha: Optional[float] = None
    qjl_m: Optional[int] = None
    qjl_seed: Optional[int] = None


CONFIGS = ("wht", "wht_mag", "wht_mag_phase", "full")


def compress(psi: np.ndarray, config: str, bits_mag: int = 3, bits_phase: int = 3,
             qjl_m: Optional[int] = None, qjl_seed: int = 0) -> Payload:
    """Run `psi` (unit-norm C^d) through the pipeline up to `config`.

    Configs (cumulative):
        "wht"            -> WHT only, store complex result.
        "wht_mag"        -> + Lloyd-Max magnitude quantization, raw phases kept.
        "wht_mag_phase"  -> + uniform phase quantization.
        "full"           -> + QJL 1-bit residual correction.
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")
    d = psi.shape[0]
    payload = Payload(config=config, d=d, bits_mag=bits_mag, bits_phase=bits_phase)

    transformed = wht(psi)

    if config == "wht":
        payload.wht_only = transformed.copy()
        return payload

    magnitudes = np.abs(transformed)
    phases = np.angle(transformed)
    mag_indices, mag_codebook = mag_quant_encode(magnitudes, bits_mag)
    payload.mag_indices = mag_indices
    payload.mag_codebook = mag_codebook

    if config == "wht_mag":
        payload.raw_phases = phases.copy()
        return payload

    phase_indices = phase_quant_encode(phases, bits_phase)
    payload.phase_indices = phase_indices

    if config == "wht_mag_phase":
        return payload

    # Full: compute intermediate reconstruction; encode QJL with fidelity-optimal alpha.
    intermediate = _decompress_wht_mag_phase(payload)
    m = qjl_m if qjl_m is not None else d
    signs, alpha = qjl_encode(psi, intermediate, m, qjl_seed)
    payload.qjl_signs = signs
    payload.qjl_alpha = alpha
    payload.qjl_m = m
    payload.qjl_seed = qjl_seed
    return payload


def _decompress_wht_only(payload: Payload) -> np.ndarray:
    return iwht(payload.wht_only)


def _decompress_wht_mag(payload: Payload) -> np.ndarray:
    mags = mag_quant_decode(payload.mag_indices, payload.mag_codebook)
    transformed = mags * np.exp(1j * payload.raw_phases)
    psi_hat = iwht(transformed.astype(np.complex128))
    norm = np.linalg.norm(psi_hat)
    if norm > 0:
        psi_hat = psi_hat / norm
    return psi_hat


def _decompress_wht_mag_phase(payload: Payload) -> np.ndarray:
    mags = mag_quant_decode(payload.mag_indices, payload.mag_codebook)
    phases = phase_quant_decode(payload.phase_indices, payload.bits_phase)
    transformed = mags * np.exp(1j * phases)
    psi_hat = iwht(transformed.astype(np.complex128))
    norm = np.linalg.norm(psi_hat)
    if norm > 0:
        psi_hat = psi_hat / norm
    return psi_hat


def _decompress_full(payload: Payload) -> np.ndarray:
    base = _decompress_wht_mag_phase(payload)
    correction = qjl_decode(payload.qjl_signs, payload.qjl_alpha, payload.qjl_m,
                            payload.d, payload.qjl_seed)
    psi_hat = base + correction
    norm = np.linalg.norm(psi_hat)
    if norm > 0:
        psi_hat = psi_hat / norm
    return psi_hat


def decompress(payload: Payload) -> np.ndarray:
    if payload.config == "wht":
        return _decompress_wht_only(payload)
    if payload.config == "wht_mag":
        return _decompress_wht_mag(payload)
    if payload.config == "wht_mag_phase":
        return _decompress_wht_mag_phase(payload)
    if payload.config == "full":
        return _decompress_full(payload)
    raise ValueError(payload.config)


# ---------------------------------------------------------------------------
# Bit accounting
# ---------------------------------------------------------------------------

def original_bits(d: int) -> int:
    """2*d floats at FLOAT_BITS each (real and imag)."""
    return 2 * d * FLOAT_BITS


def compressed_bits(payload: Payload) -> int:
    """Total bits used to represent the payload, including any sent metadata."""
    d = payload.d
    if payload.config == "wht":
        # We deliberately count "WHT only" as the same cost as the original,
        # since it's a lossless linear transform with no quantization. This
        # keeps the WHT-only point honest as the upper-fidelity baseline.
        return 2 * d * FLOAT_BITS

    bits = 0
    # Magnitudes: codebook (2^bits floats) + d indices.
    n_mag_levels = 1 << payload.bits_mag
    bits += n_mag_levels * FLOAT_BITS          # codebook
    bits += d * payload.bits_mag               # indices

    if payload.config == "wht_mag":
        # Raw phases: d floats (unquantized).
        bits += d * FLOAT_BITS
        return bits

    # Phase indices.
    bits += d * payload.bits_phase

    if payload.config == "wht_mag_phase":
        return bits

    # QJL: m sign bits + scale alpha + seed handle.
    bits += payload.qjl_m * 1
    bits += SCALE_BITS                          # alpha
    bits += SEED_BITS                           # seed (so decoder regenerates P)
    return bits


def compression_ratio(payload: Payload) -> float:
    return original_bits(payload.d) / compressed_bits(payload)


# ---------------------------------------------------------------------------
# Convenience metric
# ---------------------------------------------------------------------------

def fidelity(psi: np.ndarray, psi_hat: np.ndarray) -> float:
    """F = |<psi|psi_hat>|^2."""
    return float(np.abs(np.vdot(psi, psi_hat)) ** 2)
