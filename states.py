"""Three statevector classes for the compression measurement.

All generators return unit-norm `numpy.complex128` arrays of length 1024 (10 qubits).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


N_QUBITS = 10
DIM = 1 << N_QUBITS  # 1024


# ---------------------------------------------------------------------------
# Haar-random
# ---------------------------------------------------------------------------

def haar_random(seed: int | None = None) -> np.ndarray:
    """Uniformly random unit vector in C^DIM via Gaussian + normalize."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(DIM)
    imag = rng.standard_normal(DIM)
    psi = (real + 1j * imag).astype(np.complex128)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Transverse-field Ising ground state
# ---------------------------------------------------------------------------

# Cache the Hamiltonian/ground state since they don't depend on the seed.
# (Seed only affects starting vector for eigsh, which we keep deterministic.)
_TFIM_CACHE: dict[tuple[int, float, float, bool], np.ndarray] = {}


def _build_tfim_hamiltonian(n: int, J: float, h: float, periodic: bool = False) -> sp.csr_matrix:
    """H = -J sum_i sigma^z_i sigma^z_{i+1} - h sum_i sigma^x_i.

    Open boundary conditions by default (periodic=False).
    """
    sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float64))
    sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float64))
    I2 = sp.identity(2, format="csr", dtype=np.float64)

    def kron_op(local_ops: list[sp.csr_matrix]) -> sp.csr_matrix:
        """Tensor product of single-site ops, site 0 leftmost."""
        op = local_ops[0]
        for next_op in local_ops[1:]:
            op = sp.kron(op, next_op, format="csr")
        return op

    H = sp.csr_matrix((1 << n, 1 << n), dtype=np.float64)

    # ZZ couplings.
    pairs = list(range(n - 1))
    if periodic and n > 2:
        pairs.append(n - 1)  # wraps to site 0
    for i in pairs:
        ops = [I2] * n
        ops[i] = sz
        ops[(i + 1) % n] = sz
        H = H - J * kron_op(ops)

    # Transverse field.
    for i in range(n):
        ops = [I2] * n
        ops[i] = sx
        H = H - h * kron_op(ops)

    return H.tocsr()


def tfim_ground_state(seed: int | None = 0, J: float = 1.0, h: float = 1.0,
                      periodic: bool = False) -> np.ndarray:
    """Ground state of the 10-qubit transverse-field Ising model at the critical point.

    Cached because the Hamiltonian is deterministic; the `seed` arg is kept for
    interface uniformity with the other state classes (eigsh starting vector
    is set deterministically below).
    """
    key = (N_QUBITS, J, h, periodic)
    if key not in _TFIM_CACHE:
        H = _build_tfim_hamiltonian(N_QUBITS, J, h, periodic)
        # Deterministic starting vector for reproducibility.
        v0 = np.ones(DIM, dtype=np.float64) / np.sqrt(DIM)
        eigvals, eigvecs = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-12)
        gs = eigvecs[:, 0].astype(np.complex128)
        # Normalize and fix gauge (positive first nonzero amplitude) for determinism.
        gs = gs / np.linalg.norm(gs)
        nonzero = np.flatnonzero(np.abs(gs) > 1e-12)
        if nonzero.size:
            phase = gs[nonzero[0]] / np.abs(gs[nonzero[0]])
            gs = gs / phase
        _TFIM_CACHE[key] = gs
    gs = _TFIM_CACHE[key].copy()
    # Inject a tiny seed-dependent global phase so 20-sample runs aren't
    # identical bytewise (irrelevant to fidelity, which is gauge-invariant,
    # but keeps the QJL projection seeds independent if we use sample_idx).
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        gs = np.exp(1j * phi) * gs
    return gs


# ---------------------------------------------------------------------------
# Heisenberg antiferromagnet ground state (10-site, OBC)
# ---------------------------------------------------------------------------

# Like TFIM, the Hamiltonian is deterministic; cache the ground state.
_HEISENBERG_CACHE: dict[tuple[int, float, bool], np.ndarray] = {}


def _build_heisenberg_hamiltonian(n: int, J: float, periodic: bool = False) -> sp.csr_matrix:
    """H = J sum_i (sigma^x_i sigma^x_{i+1} + sigma^y_i sigma^y_{i+1} + sigma^z_i sigma^z_{i+1}).

    Pauli convention; J > 0 is antiferromagnetic. OBC by default.
    """
    sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    sy = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    I2 = sp.identity(2, format="csr", dtype=np.complex128)

    def kron_op(local_ops: list[sp.csr_matrix]) -> sp.csr_matrix:
        op = local_ops[0]
        for next_op in local_ops[1:]:
            op = sp.kron(op, next_op, format="csr")
        return op

    H = sp.csr_matrix((1 << n, 1 << n), dtype=np.complex128)
    pairs = list(range(n - 1))
    if periodic and n > 2:
        pairs.append(n - 1)
    for i in pairs:
        for s_op in (sx, sy, sz):
            ops = [I2] * n
            ops[i] = s_op
            ops[(i + 1) % n] = s_op
            H = H + J * kron_op(ops)
    return H.tocsr()


def heisenberg_ground_state(seed: int | None = 0, J: float = 1.0,
                             periodic: bool = False) -> np.ndarray:
    """Ground state of the 10-qubit antiferromagnetic Heisenberg chain (OBC).

    Deterministic up to a seed-dependent global phase (matches the TFIM
    interface so per-sample QJL projection seeds remain independent).
    """
    key = (N_QUBITS, J, periodic)
    if key not in _HEISENBERG_CACHE:
        H = _build_heisenberg_hamiltonian(N_QUBITS, J, periodic)
        v0 = np.ones(DIM, dtype=np.complex128) / np.sqrt(DIM)
        eigvals, eigvecs = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-12)
        gs = eigvecs[:, 0].astype(np.complex128)
        gs = gs / np.linalg.norm(gs)
        nonzero = np.flatnonzero(np.abs(gs) > 1e-12)
        if nonzero.size:
            phase = gs[nonzero[0]] / np.abs(gs[nonzero[0]])
            gs = gs / phase
        _HEISENBERG_CACHE[key] = gs
    gs = _HEISENBERG_CACHE[key].copy()
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        phi = rng.uniform(0.0, 2.0 * np.pi)
        gs = np.exp(1j * phi) * gs
    return gs


# ---------------------------------------------------------------------------
# Random matrix-product state with bond dimension chi=16
# ---------------------------------------------------------------------------

def random_mps(seed: int | None = None, chi: int = 16) -> np.ndarray:
    """Random MPS on N_QUBITS sites with bond dim `chi`, contracted to a full vector.

    Each site tensor A^{(i)} has shape (left_bond, 2, right_bond) with
    left_bond = right_bond = chi for interior sites and = 1 at the open
    boundaries. Entries are complex Gaussian; the resulting statevector is
    explicitly normalized.
    """
    rng = np.random.default_rng(seed)

    def random_tensor(left: int, right: int) -> np.ndarray:
        real = rng.standard_normal((left, 2, right))
        imag = rng.standard_normal((left, 2, right))
        return (real + 1j * imag).astype(np.complex128)

    # Build site tensors.
    tensors = []
    for i in range(N_QUBITS):
        left = 1 if i == 0 else chi
        right = 1 if i == N_QUBITS - 1 else chi
        tensors.append(random_tensor(left, right))

    # Contract left to right. State has shape (1, 2, 2, ..., 2, right_bond).
    # We accumulate a tensor `acc` of shape (1, 2^k, right_bond_k).
    acc = tensors[0]  # shape (1, 2, chi)
    for i in range(1, N_QUBITS):
        # acc: (1, 2^i, B), next: (B, 2, R)
        next_t = tensors[i]
        # einsum: (1, 2^i, B) * (B, 2, R) -> (1, 2^i, 2, R) -> (1, 2^(i+1), R)
        acc = np.einsum("lab,bcr->lacr", acc, next_t)
        l, a, c, r = acc.shape
        acc = acc.reshape(l, a * c, r)
    # Final shape: (1, DIM, 1).
    psi = acc.reshape(DIM)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Random Clifford-circuit state
# ---------------------------------------------------------------------------

CLIFFORD_DEFAULT_DEPTH = 200


def _apply_single_qubit_gate(psi: np.ndarray, gate: np.ndarray, qubit: int,
                              n: int) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` (0 = leftmost / most significant) of an n-qubit state."""
    left_dim = 1 << qubit
    right_dim = 1 << (n - 1 - qubit)
    psi_r = psi.reshape(left_dim, 2, right_dim)
    return np.einsum("ab,lbr->lar", gate, psi_r).reshape(-1)


def _apply_cnot(psi: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """CNOT with `control` and `target` qubits (0 = leftmost). Vectorized over basis."""
    d = psi.shape[0]
    indices = np.arange(d)
    c_bits = (indices >> (n - 1 - control)) & 1
    target_mask = 1 << (n - 1 - target)
    flipped = indices ^ target_mask
    return np.where(c_bits == 1, psi[flipped], psi)


_H_GATE = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
_S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)


def random_clifford_state(seed: int | None = None,
                           depth: int = CLIFFORD_DEFAULT_DEPTH) -> np.ndarray:
    """Apply `depth` uniformly random Clifford generators ({H, S, CNOT}) to |0...0>.

    Generator set per gate:
      - H on a uniformly random qubit
      - S on a uniformly random qubit
      - CNOT on a uniformly random ordered pair of distinct qubits
    The output is a stabilizer state: every nonzero amplitude has magnitude
    1/sqrt(2^k) for some k in {0, ..., N_QUBITS}.
    """
    rng = np.random.default_rng(seed)
    psi = np.zeros(DIM, dtype=np.complex128)
    psi[0] = 1.0
    n = N_QUBITS
    for _ in range(depth):
        kind = rng.integers(0, 3)
        if kind == 0:
            q = int(rng.integers(0, n))
            psi = _apply_single_qubit_gate(psi, _H_GATE, q, n)
        elif kind == 1:
            q = int(rng.integers(0, n))
            psi = _apply_single_qubit_gate(psi, _S_GATE, q, n)
        else:
            c = int(rng.integers(0, n))
            t = int(rng.integers(0, n - 1))
            if t >= c:
                t += 1
            psi = _apply_cnot(psi, c, t, n)
    # Renormalize against accumulated FP drift.
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STATE_GENERATORS = {
    "haar": haar_random,
    "tfim": tfim_ground_state,
    "mps": random_mps,
    "heisenberg": heisenberg_ground_state,
    "clifford": random_clifford_state,
}
