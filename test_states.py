"""Sanity checks for the v3 state generators.

Spot checks; not a comprehensive test suite. Run with:
    python test_states.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

import states


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return ok


# ---------------------------------------------------------------------------
# Heisenberg
# ---------------------------------------------------------------------------

def test_heisenberg_unit_norm() -> bool:
    psi = states.heisenberg_ground_state(seed=0)
    err = abs(np.linalg.norm(psi) - 1.0)
    return _check("Heisenberg unit norm", err < 1e-10, f"|||psi||-1|={err:.2e}")


def test_heisenberg_shape_dtype() -> bool:
    psi = states.heisenberg_ground_state(seed=0)
    ok = psi.shape == (1024,) and psi.dtype == np.complex128
    return _check("Heisenberg complex128[1024]", ok, f"shape={psi.shape} dtype={psi.dtype}")


def test_heisenberg_energy_matches_dense_eigh() -> bool:
    """eigsh ground state energy matches dense eigh to <1e-9 (theoretical = numerical)."""
    H = states._build_heisenberg_hamiltonian(10, 1.0, periodic=False).toarray()
    e_dense = float(np.linalg.eigvalsh(H)[0])
    psi = states.heisenberg_ground_state(seed=0)
    e_psi = float(np.real(np.vdot(psi, H @ psi)))
    err = abs(e_psi - e_dense)
    # Known value for N=10 OBC AFM Pauli Heisenberg: E_0 ~= -17.0321
    in_range = -17.04 < e_dense < -17.02
    return _check("Heisenberg E_0 matches dense eigh & known value (-17.03)",
                  err < 1e-9 and in_range,
                  f"E_psi={e_psi:.6f} E_dense={e_dense:.6f} err={err:.2e}")


def test_heisenberg_deterministic_up_to_phase() -> bool:
    """Two seeds give identical states up to a global phase (fidelity ~ 1)."""
    a = states.heisenberg_ground_state(seed=0)
    b = states.heisenberg_ground_state(seed=999)
    fid = float(abs(np.vdot(a, b)) ** 2)
    return _check("Heisenberg deterministic up to global phase",
                  abs(fid - 1.0) < 1e-10, f"F={fid:.10f}")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_heisenberg_unit_norm,
        test_heisenberg_shape_dtype,
        test_heisenberg_energy_matches_dense_eigh,
        test_heisenberg_deterministic_up_to_phase,
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
    print(f"{len(tests) - failures}/{len(tests)} v3 sanity checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
