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
# Clifford
# ---------------------------------------------------------------------------

def test_clifford_unit_norm_shape_dtype() -> bool:
    psi = states.random_clifford_state(seed=0)
    ok = (psi.shape == (1024,) and psi.dtype == np.complex128
          and abs(np.linalg.norm(psi) - 1.0) < 1e-10)
    return _check("Clifford unit norm + complex128[1024]", ok,
                  f"shape={psi.shape} dtype={psi.dtype} "
                  f"|||psi||-1|={abs(np.linalg.norm(psi)-1.0):.2e}")


def test_clifford_stabilizer_amplitudes() -> bool:
    """Every nonzero amplitude has magnitude 1/sqrt(2^k) for some integer k.

    Stabilizer-state property: the support of psi has size 2^k for some k <= N,
    and on the support all amplitudes have equal magnitude 1/sqrt(2^k).
    """
    fail = []
    for seed in range(5):
        psi = states.random_clifford_state(seed=seed)
        nonzero = np.abs(psi)[np.abs(psi) > 1e-10]
        if nonzero.size == 0:
            fail.append(f"seed={seed}: empty support")
            continue
        # log2(1/|a|^2) should be a non-negative integer.
        k_float = -np.log2(nonzero ** 2)
        k_int = np.round(k_float)
        max_dev = float(np.max(np.abs(k_float - k_int)))
        if max_dev > 1e-8 or np.any(k_int < 0) or np.any(k_int > 10):
            fail.append(f"seed={seed}: max_dev={max_dev:.2e}, k range "
                        f"[{int(k_int.min())}, {int(k_int.max())}]")
    return _check("Clifford amplitudes are 1/sqrt(2^k)",
                  not fail, "; ".join(fail) if fail else "")


def test_clifford_stochastic() -> bool:
    a = states.random_clifford_state(seed=0)
    b = states.random_clifford_state(seed=0)
    c = states.random_clifford_state(seed=1)
    ok = np.array_equal(a, b) and not np.array_equal(a, c)
    return _check("Clifford stochastic w/ deterministic seeds", ok)


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_heisenberg_unit_norm,
        test_heisenberg_shape_dtype,
        test_heisenberg_energy_matches_dense_eigh,
        test_heisenberg_deterministic_up_to_phase,
        test_clifford_unit_norm_shape_dtype,
        test_clifford_stabilizer_amplitudes,
        test_clifford_stochastic,
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
