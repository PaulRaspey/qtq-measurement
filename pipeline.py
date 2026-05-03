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
# Stage 2 alternates: outlier-aware magnitude quantizers
# ---------------------------------------------------------------------------
#
# Each alternate is a drop-in replacement for `mag_quant_encode` /
# `mag_quant_decode`: it consumes a magnitude vector (length d), produces some
# encoded representation that the decoder maps back to a length-d magnitude
# vector. The decoder output goes through the same `mags * exp(1j*phases)`
# recombination + inverse WHT + renormalize as before.
#
# v2 (this follow-up) adds three: top-k exact + Lloyd-Max residual,
# log-domain Lloyd-Max, and percentile-clipped Lloyd-Max. Each is designed
# to handle heavy-tailed magnitude distributions better than plain Lloyd-Max.

LOG_EPS = 1e-12
DEFAULT_PERCENTILE = 99.0
DEFAULT_TOPK = 8
COUNT_BITS = 16  # bits to store outlier-count in percentile-clip header


def topk_lloyd_encode(magnitudes: np.ndarray, bits: int, k: int = DEFAULT_TOPK
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-k exact + Lloyd-Max on the rest.

    Identifies the k indices with largest |c|, stores those magnitudes
    losslessly (float32), and Lloyd-Max-quantizes the remaining d-k entries
    at the requested bit budget. Returns (topk_idx, topk_vals, rest_indices,
    rest_codebook). `topk_idx` is sorted ascending for deterministic
    serialization.
    """
    d = magnitudes.shape[0]
    if k >= d:
        raise ValueError(f"k={k} must be strictly less than d={d}")
    n_levels = 1 << bits
    topk_idx = np.argpartition(magnitudes, -k)[-k:]
    topk_idx = np.sort(topk_idx).astype(np.int32)
    topk_vals = magnitudes[topk_idx].astype(np.float32)
    rest_mask = np.ones(d, dtype=bool)
    rest_mask[topk_idx] = False
    rest_mags = magnitudes[rest_mask]
    rest_codebook = lloyd_max_codebook(rest_mags, n_levels)
    boundaries = (rest_codebook[:-1] + rest_codebook[1:]) / 2.0
    rest_indices = np.searchsorted(boundaries, rest_mags).astype(np.int32)
    return topk_idx, topk_vals, rest_indices, rest_codebook


def topk_lloyd_decode(topk_idx: np.ndarray, topk_vals: np.ndarray,
                      rest_indices: np.ndarray, rest_codebook: np.ndarray,
                      d: int) -> np.ndarray:
    out = np.zeros(d, dtype=np.float64)
    rest_mask = np.ones(d, dtype=bool)
    rest_mask[topk_idx] = False
    out[rest_mask] = rest_codebook[rest_indices]
    out[topk_idx] = topk_vals.astype(np.float64)
    return out


def log_lloyd_encode(magnitudes: np.ndarray, bits: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max in log space.

    Quantizes log(|c| + LOG_EPS) instead of |c|; better-conditioned for
    distributions spanning multiple orders of magnitude. The codebook stored
    holds log-space centroids; decode applies exp() and clips negatives that
    can arise from numerical underflow near zero.
    """
    n_levels = 1 << bits
    log_mags = np.log(magnitudes + LOG_EPS)
    codebook = lloyd_max_codebook(log_mags, n_levels)
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0
    indices = np.searchsorted(boundaries, log_mags).astype(np.int32)
    return indices, codebook


def log_lloyd_decode(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    log_mags = codebook[indices]
    out = np.exp(log_mags) - LOG_EPS
    return np.clip(out, 0.0, None)


def percentile_clip_encode(magnitudes: np.ndarray, bits: int,
                            percentile: float = DEFAULT_PERCENTILE
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Outliers above `percentile` stored exact; Lloyd-Max trained on the *remaining* bulk only.

    For percentile=99 on d=1024, ~10 entries land in the outlier tail. The
    bulk distribution (with outliers fully removed, not clipped in place) is
    much less skewed, so Lloyd-Max allocates centroids only across the actual
    bulk and doesn't waste any on the threshold spike. Returns
    (outlier_idx, outlier_vals, bulk_indices, bulk_codebook); `bulk_indices`
    has length d - n_outliers, indexed in original-position order over the
    bulk.
    """
    d = magnitudes.shape[0]
    n_levels = 1 << bits
    threshold = np.percentile(magnitudes, percentile)
    outlier_mask = magnitudes > threshold
    outlier_idx = np.flatnonzero(outlier_mask).astype(np.int32)
    outlier_vals = magnitudes[outlier_mask].astype(np.float32)
    bulk_mags = magnitudes[~outlier_mask]
    bulk_codebook = lloyd_max_codebook(bulk_mags, n_levels)
    boundaries = (bulk_codebook[:-1] + bulk_codebook[1:]) / 2.0
    bulk_indices = np.searchsorted(boundaries, bulk_mags).astype(np.int32)
    return outlier_idx, outlier_vals, bulk_indices, bulk_codebook


def percentile_clip_decode(outlier_idx: np.ndarray, outlier_vals: np.ndarray,
                            bulk_indices: np.ndarray, bulk_codebook: np.ndarray,
                            d: int) -> np.ndarray:
    out = np.zeros(d, dtype=np.float64)
    bulk_mask = np.ones(d, dtype=bool)
    bulk_mask[outlier_idx] = False
    out[bulk_mask] = bulk_codebook[bulk_indices]
    out[outlier_idx] = outlier_vals.astype(np.float64)
    return out


MAG_QUANTIZERS = ("lloyd", "topk", "log", "percentile")


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
    mag_quantizer: str = "lloyd"   # one of MAG_QUANTIZERS
    # Orthogonal basis applied before quantization (v6+; default "wht" keeps
    # v1/v2/v3 byte accounting unchanged). Decoder needs the same basis.
    basis: str = "wht"
    # WHT-only (or chosen-basis-only) fallback: uncompressed complex
    # post-transform vector. Field name kept for v1/v2/v3 backward compat.
    wht_only: Optional[np.ndarray] = None
    # Magnitude stream + codebook (always populated for non-WHT configs;
    # interpretation depends on `mag_quantizer`):
    #   lloyd      -> direct Lloyd-Max indices/codebook over |c|
    #   topk       -> Lloyd-Max indices/codebook over the d-k non-topk entries
    #   log        -> Lloyd-Max indices/codebook over log(|c|+LOG_EPS)
    #   percentile -> Lloyd-Max indices/codebook over the clipped bulk (length d)
    mag_indices: Optional[np.ndarray] = None
    mag_codebook: Optional[np.ndarray] = None
    # Top-k extras (mag_quantizer == "topk").
    topk_idx: Optional[np.ndarray] = None
    topk_vals: Optional[np.ndarray] = None
    # Percentile-clip extras (mag_quantizer == "percentile").
    outlier_idx: Optional[np.ndarray] = None
    outlier_vals: Optional[np.ndarray] = None
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


# ---------------------------------------------------------------------------
# Basis pluggability (v6+)
# ---------------------------------------------------------------------------
#
# By default the pipeline applies a normalized WHT before quantization.
# v6 allows the encoder to pick among orthogonal bases per vector. The
# decoder reads `payload.basis` and inverts accordingly. Two-bit basis tag
# is charged in `compressed_bits` only when basis != "wht", so the v1/v2/v3
# byte accounting is preserved.

SUPPORTED_BASES = ("wht", "dct", "identity")
BASIS_TAG_BITS = 2  # log2(3) ~= 1.58; round up. WHT-only path pays 0.


def _apply_forward_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    if basis == "wht":
        return wht(psi)
    # Defer the import; concentration_basis imports pipeline (for wht/iwht),
    # so a top-level import here would create a circular dependency.
    from concentration_basis import transform_in_basis
    return transform_in_basis(psi, basis)


def _apply_inverse_basis(coeffs: np.ndarray, basis: str) -> np.ndarray:
    if basis == "wht":
        return iwht(coeffs)
    from concentration_basis import inverse_transform_in_basis
    return inverse_transform_in_basis(coeffs, basis)


def _encode_magnitudes(payload: Payload, magnitudes: np.ndarray, bits: int,
                        topk: int, percentile: float) -> None:
    """Run the chosen magnitude quantizer and stash its state on `payload`."""
    if payload.mag_quantizer == "lloyd":
        idx, cb = mag_quant_encode(magnitudes, bits)
        payload.mag_indices = idx
        payload.mag_codebook = cb
    elif payload.mag_quantizer == "topk":
        topk_idx, topk_vals, rest_indices, rest_cb = topk_lloyd_encode(magnitudes, bits, k=topk)
        payload.topk_idx = topk_idx
        payload.topk_vals = topk_vals
        payload.mag_indices = rest_indices
        payload.mag_codebook = rest_cb
    elif payload.mag_quantizer == "log":
        idx, cb = log_lloyd_encode(magnitudes, bits)
        payload.mag_indices = idx
        payload.mag_codebook = cb
    elif payload.mag_quantizer == "percentile":
        out_idx, out_vals, bulk_idx, bulk_cb = percentile_clip_encode(magnitudes, bits,
                                                                       percentile=percentile)
        payload.outlier_idx = out_idx
        payload.outlier_vals = out_vals
        payload.mag_indices = bulk_idx
        payload.mag_codebook = bulk_cb
    else:
        raise ValueError(f"unknown mag_quantizer {payload.mag_quantizer!r}")


def _decode_magnitudes(payload: Payload) -> np.ndarray:
    if payload.mag_quantizer == "lloyd":
        return mag_quant_decode(payload.mag_indices, payload.mag_codebook)
    if payload.mag_quantizer == "topk":
        return topk_lloyd_decode(payload.topk_idx, payload.topk_vals,
                                 payload.mag_indices, payload.mag_codebook, payload.d)
    if payload.mag_quantizer == "log":
        return log_lloyd_decode(payload.mag_indices, payload.mag_codebook)
    if payload.mag_quantizer == "percentile":
        return percentile_clip_decode(payload.outlier_idx, payload.outlier_vals,
                                      payload.mag_indices, payload.mag_codebook, payload.d)
    raise ValueError(f"unknown mag_quantizer {payload.mag_quantizer!r}")


def compress(psi: np.ndarray, config: str, bits_mag: int = 3, bits_phase: int = 3,
             qjl_m: Optional[int] = None, qjl_seed: int = 0,
             mag_quantizer: str = "lloyd", topk: int = DEFAULT_TOPK,
             percentile: float = DEFAULT_PERCENTILE,
             basis: str = "wht") -> Payload:
    """Run `psi` (unit-norm C^d) through the pipeline up to `config`.

    Configs (cumulative):
        "wht"            -> chosen-basis transform only, store complex result.
        "wht_mag"        -> + magnitude quantization, raw phases kept.
        "wht_mag_phase"  -> + uniform phase quantization.
        "full"           -> + QJL 1-bit residual correction.

    `mag_quantizer` selects the stage-2 algorithm; see MAG_QUANTIZERS. `topk`
    only applies to mag_quantizer="topk"; `percentile` only to "percentile".
    `basis` selects the orthogonal transform applied before quantization;
    default "wht" preserves v1/v2/v3 behavior. v6+ allows "dct" or "identity";
    decoder reads payload.basis and inverts accordingly.
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")
    if mag_quantizer not in MAG_QUANTIZERS:
        raise ValueError(f"mag_quantizer must be one of {MAG_QUANTIZERS}, got {mag_quantizer!r}")
    d = psi.shape[0]
    payload = Payload(config=config, d=d, bits_mag=bits_mag, bits_phase=bits_phase,
                      mag_quantizer=mag_quantizer, basis=basis)

    transformed = _apply_forward_basis(psi, basis)

    if config == "wht":
        payload.wht_only = transformed.copy()
        return payload

    magnitudes = np.abs(transformed)
    phases = np.angle(transformed)
    _encode_magnitudes(payload, magnitudes, bits_mag, topk=topk, percentile=percentile)

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
    return _apply_inverse_basis(payload.wht_only, payload.basis)


def _decompress_wht_mag(payload: Payload) -> np.ndarray:
    mags = _decode_magnitudes(payload)
    transformed = mags * np.exp(1j * payload.raw_phases)
    psi_hat = _apply_inverse_basis(transformed.astype(np.complex128), payload.basis)
    norm = np.linalg.norm(psi_hat)
    if norm > 0:
        psi_hat = psi_hat / norm
    return psi_hat


def _decompress_wht_mag_phase(payload: Payload) -> np.ndarray:
    mags = _decode_magnitudes(payload)
    phases = phase_quant_decode(payload.phase_indices, payload.bits_phase)
    transformed = mags * np.exp(1j * phases)
    psi_hat = _apply_inverse_basis(transformed.astype(np.complex128), payload.basis)
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


def _magnitude_stage_bits(payload: Payload) -> int:
    """Bits for whichever magnitude quantizer is in use."""
    d = payload.d
    bits = payload.bits_mag
    n_levels = 1 << bits
    if payload.mag_quantizer == "lloyd":
        return n_levels * FLOAT_BITS + d * bits
    if payload.mag_quantizer == "topk":
        k = int(payload.topk_idx.shape[0])
        idx_bits = max(1, int(np.ceil(np.log2(d))))
        return (k * idx_bits             # top-k positions
                + k * FLOAT_BITS         # top-k values (full precision)
                + (d - k) * bits         # rest indices
                + n_levels * FLOAT_BITS) # rest codebook
    if payload.mag_quantizer == "log":
        # Same byte-shape as lloyd; LOG_EPS is a constant by convention.
        return n_levels * FLOAT_BITS + d * bits
    if payload.mag_quantizer == "percentile":
        n_out = int(payload.outlier_idx.shape[0])
        idx_bits = max(1, int(np.ceil(np.log2(d))))
        return (COUNT_BITS                  # outlier count header (variable)
                + n_out * idx_bits          # outlier positions
                + n_out * FLOAT_BITS        # outlier values (full precision)
                + (d - n_out) * bits        # bulk indices (length d - n_out)
                + n_levels * FLOAT_BITS)    # bulk codebook
    raise ValueError(f"unknown mag_quantizer {payload.mag_quantizer!r}")


def compressed_bits(payload: Payload) -> int:
    """Total bits used to represent the payload, including any sent metadata."""
    d = payload.d
    if payload.config == "wht":
        # We deliberately count "WHT only" as the same cost as the original,
        # since it's a lossless linear transform with no quantization. This
        # keeps the WHT-only point honest as the upper-fidelity baseline.
        return 2 * d * FLOAT_BITS

    bits = _magnitude_stage_bits(payload)

    if payload.config == "wht_mag":
        # Raw phases: d floats (unquantized).
        bits += d * FLOAT_BITS
    else:
        # Phase indices.
        bits += d * payload.bits_phase
        if payload.config != "wht_mag_phase":
            # QJL: m sign bits + scale alpha + seed handle.
            bits += payload.qjl_m * 1
            bits += SCALE_BITS                  # alpha
            bits += SEED_BITS                   # seed (so decoder regenerates P)

    # v6+: charge basis-tag bits only when basis != "wht" so v1/v2/v3 byte
    # accounting (and the recorded results_v*.csv ratios) are unchanged.
    if payload.basis != "wht":
        bits += BASIS_TAG_BITS
    return bits


def compression_ratio(payload: Payload) -> float:
    return original_bits(payload.d) / compressed_bits(payload)


# ---------------------------------------------------------------------------
# Convenience metric
# ---------------------------------------------------------------------------

def fidelity(psi: np.ndarray, psi_hat: np.ndarray) -> float:
    """F = |<psi|psi_hat>|^2."""
    return float(np.abs(np.vdot(psi, psi_hat)) ** 2)


# ---------------------------------------------------------------------------
# v6: basis-adaptive selectors
# ---------------------------------------------------------------------------
#
# Two selection strategies. `compress_adaptive_kstar` is the realistic
# encoder: it picks the basis with the smallest k*(0.9) -- a single scalar
# the encoder can compute cheaply -- without ever running the full pipeline
# in alternative bases. `compress_adaptive_oracle` is the upper bound: try
# all bases at full cost, pick the one with the highest reconstructed
# fidelity. The gap between the two tells us how good a predictor k* is
# for the actual quantity we care about (fidelity).

def compress_adaptive_kstar(psi: np.ndarray, config: str = "full",
                             bits_mag: int = 3, bits_phase: int = 3,
                             mag_quantizer: str = "topk", topk: int = DEFAULT_TOPK,
                             qjl_seed: int = 0,
                             bases: tuple = SUPPORTED_BASES,
                             threshold: float = 0.9) -> Payload:
    """Pick the basis with smallest k*(threshold), then run the pipeline there."""
    from concentration_basis import k_star_in_basis
    best_basis = None
    best_k = None
    for b in bases:
        k = k_star_in_basis(psi, threshold, b)
        if best_k is None or k < best_k:
            best_k = k
            best_basis = b
    return compress(psi, config=config, bits_mag=bits_mag, bits_phase=bits_phase,
                    mag_quantizer=mag_quantizer, topk=topk, qjl_seed=qjl_seed,
                    basis=best_basis)


def compress_adaptive_oracle(psi: np.ndarray, config: str = "full",
                              bits_mag: int = 3, bits_phase: int = 3,
                              mag_quantizer: str = "topk", topk: int = DEFAULT_TOPK,
                              qjl_seed: int = 0,
                              bases: tuple = SUPPORTED_BASES) -> Payload:
    """Try every basis, return the payload that decodes to the highest fidelity."""
    best_payload = None
    best_fid = -1.0
    for b in bases:
        p = compress(psi, config=config, bits_mag=bits_mag, bits_phase=bits_phase,
                     mag_quantizer=mag_quantizer, topk=topk, qjl_seed=qjl_seed, basis=b)
        psi_hat = decompress(p)
        f = fidelity(psi, psi_hat)
        if f > best_fid:
            best_fid = f
            best_payload = p
    return best_payload


def compress_adaptive_with_selector(psi: np.ndarray, selector,
                                     config: str = "full",
                                     bits_mag: int = 3, bits_phase: int = 3,
                                     mag_quantizer: str = "topk",
                                     topk: int = DEFAULT_TOPK,
                                     qjl_seed: int = 0,
                                     bases: tuple = SUPPORTED_BASES) -> Payload:
    """Run any (psi, bases)->basis selector then compress in the chosen basis.

    `selector` must be a callable that accepts (psi, bases) and returns a
    basis name string. See `basis_selectors.py` for v7 selector functions.
    """
    chosen = selector(psi, bases)
    return compress(psi, config=config, bits_mag=bits_mag, bits_phase=bits_phase,
                    mag_quantizer=mag_quantizer, topk=topk, qjl_seed=qjl_seed,
                    basis=chosen)
