"""Lightweight sanity checks for the four pipeline stages.

These are spot checks, not a comprehensive test suite. Run with:
    python test_pipeline.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

import pipeline as pl
import states


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return ok


def test_wht_self_inverse() -> bool:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    y = pl.iwht(pl.wht(x))
    err = float(np.max(np.abs(x - y)))
    return _check("WHT is self-inverse to ~1e-12", err < 1e-10, f"max abs err {err:.2e}")


def test_wht_orthogonal_norm() -> bool:
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    x /= np.linalg.norm(x)
    y = pl.wht(x)
    err = abs(np.linalg.norm(y) - 1.0)
    return _check("WHT preserves L2 norm", err < 1e-10, f"|||y||-1| = {err:.2e}")


def test_polarquant_norm_preserved() -> bool:
    rng = np.random.default_rng(2)
    psi = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    payload = pl.compress(psi, "wht_mag", bits_mag=3)
    psi_hat = pl.decompress(payload)
    err = abs(np.linalg.norm(psi_hat) - 1.0)
    return _check("PolarQuant decode preserves unit norm", err < 1e-10,
                  f"|||psi_hat||-1| = {err:.2e}")


def test_phase_quant_levels() -> bool:
    """k-bit phase quantization should map every input to one of 2^k midpoint values."""
    bits = 3
    n_levels = 1 << bits
    step = 2.0 * np.pi / n_levels
    expected = (np.arange(n_levels) + 0.5) * step
    rng = np.random.default_rng(3)
    phases = rng.uniform(0, 2 * np.pi, 5000)
    indices = pl.phase_quant_encode(phases, bits)
    decoded = pl.phase_quant_decode(indices, bits)
    unique_decoded = np.unique(decoded)
    ok_count = unique_decoded.size == n_levels
    ok_values = np.allclose(np.sort(unique_decoded), expected)
    ok_range = (indices.min() >= 0) and (indices.max() < n_levels)
    return _check(
        f"Phase quant maps to {n_levels} midpoint levels",
        ok_count and ok_values and ok_range,
        f"levels={unique_decoded.size}, idx range [{indices.min()},{indices.max()}]",
    )


def test_pipeline_on_basis_state() -> bool:
    """A basis state has zero entropy; full pipeline should recover it almost exactly."""
    psi = np.zeros(1024, dtype=np.complex128)
    psi[42] = 1.0
    fids = []
    for bits in (2, 3, 4):
        payload = pl.compress(psi, "full", bits_mag=bits, bits_phase=bits, qjl_seed=0)
        psi_hat = pl.decompress(payload)
        fids.append(pl.fidelity(psi, psi_hat))
    ok = all(f > 0.999 for f in fids)
    return _check(
        "Full pipeline on basis state F > 0.999 at all bit budgets",
        ok,
        f"fidelities {[round(f, 6) for f in fids]}",
    )


def test_states_unit_norm() -> bool:
    fail = []
    for name, gen in states.STATE_GENERATORS.items():
        psi = gen(seed=0)
        if abs(np.linalg.norm(psi) - 1.0) > 1e-10:
            fail.append(name)
    return _check("All state generators return unit-norm vectors",
                  not fail, f"failed: {fail}" if fail else "")


def test_states_dim() -> bool:
    fail = []
    for name, gen in states.STATE_GENERATORS.items():
        psi = gen(seed=0)
        if psi.shape != (1024,) or psi.dtype != np.complex128:
            fail.append(f"{name}({psi.shape},{psi.dtype})")
    return _check("All state generators return complex128[1024]", not fail,
                  f"bad: {fail}" if fail else "")


def test_full_pipeline_runs_on_real_states() -> bool:
    """Smoke test: pipeline runs end-to-end on each state class and returns finite fidelity."""
    fail = []
    for name, gen in states.STATE_GENERATORS.items():
        try:
            psi = gen(seed=7)
            payload = pl.compress(psi, "full", bits_mag=3, bits_phase=3, qjl_seed=7)
            psi_hat = pl.decompress(payload)
            f = pl.fidelity(psi, psi_hat)
            if not np.isfinite(f):
                fail.append(f"{name}: non-finite fidelity {f}")
        except Exception as exc:  # noqa: BLE001
            fail.append(f"{name}: {type(exc).__name__}: {exc}")
    return _check("Full pipeline runs end-to-end on every state class", not fail,
                  "; ".join(fail) if fail else "")


# ---------------------------------------------------------------------------
# v2: outlier-aware magnitude quantizers
# ---------------------------------------------------------------------------

V2_QUANTIZERS = ("topk", "log", "percentile")


def test_v2_quantizers_preserve_unit_norm() -> bool:
    """Each v2 quantizer must yield a unit-norm reconstruction on a Haar state."""
    rng = np.random.default_rng(11)
    psi = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    fail = []
    for q in V2_QUANTIZERS:
        payload = pl.compress(psi, "wht_mag", bits_mag=3, mag_quantizer=q)
        psi_hat = pl.decompress(payload)
        err = abs(np.linalg.norm(psi_hat) - 1.0)
        if err >= 1e-10:
            fail.append(f"{q}: |||psi_hat||-1| = {err:.2e}")
    return _check("v2 quantizers preserve unit norm", not fail,
                  "; ".join(fail) if fail else "")


def test_v2_quantizers_basis_state_fidelity() -> bool:
    """Basis state should round-trip to F >= 0.999 through each v2 quantizer (full pipeline)."""
    psi = np.zeros(1024, dtype=np.complex128)
    psi[100] = 1.0
    fail = []
    for q in V2_QUANTIZERS:
        for bits in (2, 3, 4):
            payload = pl.compress(psi, "full", bits_mag=bits, bits_phase=bits,
                                  mag_quantizer=q, qjl_seed=0)
            psi_hat = pl.decompress(payload)
            f = pl.fidelity(psi, psi_hat)
            if f < 0.999:
                fail.append(f"{q}@{bits}b: F={f:.4f}")
    return _check("v2 quantizers F>=0.999 on basis state at all bit budgets", not fail,
                  "; ".join(fail) if fail else "")


def test_v2_quantizers_high_bits_near_lossless() -> bool:
    """At a generous bit budget, each v2 quantizer should give wht_mag F > 0.999.

    "Lossless at full precision" interpreted operationally: 10 bits per magnitude
    (1024 codebook entries) should be enough that the dominant remaining error
    is float32 truncation of the codebook, not the quantization itself. We use
    wht_mag (raw phases) so phase quant doesn't pollute the test.
    """
    rng = np.random.default_rng(22)
    psi = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    fail = []
    for q in V2_QUANTIZERS:
        payload = pl.compress(psi, "wht_mag", bits_mag=10, mag_quantizer=q)
        psi_hat = pl.decompress(payload)
        f = pl.fidelity(psi, psi_hat)
        if f < 0.999:
            fail.append(f"{q}@10b: F={f:.6f}")
    return _check("v2 quantizers near-lossless at 10-bit budget", not fail,
                  "; ".join(fail) if fail else "")


def test_v2_quantizers_bit_accounting_finite() -> bool:
    """compressed_bits must produce a positive finite count for each v2 variant."""
    rng = np.random.default_rng(33)
    psi = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    fail = []
    for q in V2_QUANTIZERS:
        payload = pl.compress(psi, "full", bits_mag=3, bits_phase=3,
                              mag_quantizer=q, qjl_seed=0)
        bits = pl.compressed_bits(payload)
        ratio = pl.compression_ratio(payload)
        if not (0 < bits < pl.original_bits(1024) * 10) or not np.isfinite(ratio):
            fail.append(f"{q}: bits={bits}, ratio={ratio}")
    return _check("v2 bit accounting yields finite positive ratios", not fail,
                  "; ".join(fail) if fail else "")


def main() -> int:
    tests = [
        test_wht_self_inverse,
        test_wht_orthogonal_norm,
        test_polarquant_norm_preserved,
        test_phase_quant_levels,
        test_pipeline_on_basis_state,
        test_states_unit_norm,
        test_states_dim,
        test_full_pipeline_runs_on_real_states,
        test_v2_quantizers_preserve_unit_norm,
        test_v2_quantizers_basis_state_fidelity,
        test_v2_quantizers_high_bits_near_lossless,
        test_v2_quantizers_bit_accounting_finite,
    ]
    failures = 0
    for t in tests:
        try:
            if not t():
                failures += 1
        except Exception:  # noqa: BLE001
            print(f"[FAIL] {t.__name__} -- exception")
            traceback.print_exc()
            failures += 1
    print()
    print(f"{len(tests) - failures}/{len(tests)} sanity checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
