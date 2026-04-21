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
