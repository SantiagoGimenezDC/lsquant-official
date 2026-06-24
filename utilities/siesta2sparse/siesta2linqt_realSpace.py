#!/usr/bin/env python3
"""
siesta2linqt.py

Reads a SIESTA .fdf file, builds a k-point grid of size kx × ky × kz,
performs Löwdin orthogonalisation at each k-point, assembles the full
supercell (block-diagonal) operators, and writes them as linqt .CSR files:

    <prefix>.HAM.CSR    – Löwdin-orthogonalised Hamiltonian  H_ortho  (k-space, block-diagonal)
    <prefix>.VX.CSR     – velocity operator V_x  (k-space, from dHk/dk, for Kubo-Bastin)
    <prefix>.VY.CSR     – velocity operator V_y  (k-space)
    <prefix>.BLOCH.CSR  – Bloch transform matrix B (atom gauge)
    <prefix>.HREAL.CSR  – real-space Hamiltonian  H_real = B† H_blk B  (for MSD)
    <prefix>.VXREAL.CSR – real-space velocity  [H_real, X]             (for MSD)
    <prefix>.VYREAL.CSR – real-space velocity  [H_real, Y]             (for MSD)

Löwdin velocity formula (from the non-orthogonal Kubo–Bastin literature):

    H_ortho  = S^{-1/2} H S^{-1/2}

    V^α = S^{-1/2} ∂_α H S^{-1/2}
          - ½ H_ortho (∂_α S) S^{-1}
          - ½ (∂_α S) S^{-1} H_ortho

where ∂_α ≡ ∂/∂k_α  (sisl's dHk / dSk, gauge='r').

Bloch transform matrix (atom gauge):

    B_{(ik,α),(R,β)} = δ_{αβ} / √N_k × exp(i k_ik · (R + τ_α))

where τ_α is the Cartesian position of the atom hosting orbital α, and R
is the real-space lattice vector of unit cell labelled by grid index iR.
This unitary matrix maps the k-space block-diagonal basis to the real-space
supercell basis, enabling the mixed-space disorder application:

    H_full = H_k  +  B† V_disorder B

where V_disorder is diagonal in real space.

S^{-1/2} is computed via eigendecomposition of S (Hermitian positive-definite):
    S = U Λ U†  →  S^{-1/2} = U Λ^{-1/2} U†

Usage
-----
    python siesta2linqt.py <fdf_file> <kx> <ky> [--prefix NAME]
                           [--kz KZ] [--gauge GAUGE]

    fdf_file   path to SIESTA RUN.fdf (or equivalent)
    kx, ky     k-point grid dimensions
    --prefix   output file prefix  (default: derived from fdf filename)
    --kz       k-points along z    (default: 1)
    --gauge    sisl gauge string   (default: 'r')

Dependencies
------------
    pip install sisl scipy numpy
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.sparse import csr_matrix, coo_matrix, block_diag


# ─────────────────────────────────────────────────────────────────────────────
# CSR writer  (linqt format)
# ─────────────────────────────────────────────────────────────────────────────
def prune_and_sort(mat, tol=1e-6):
    """
    Remove entries with |value| < tol and sort column indices within each row
    (required by Eigen's CSR format: indices must be strictly ascending per row).
    """
    mat = mat.tocsr()

    # Prune small entries
    mat.data[np.abs(mat.data) < tol] = 0.0
    mat.eliminate_zeros()

    # Sort column indices within each row (Eigen requirement)
    mat.sort_indices()

    # Collapse any duplicate indices that may have appeared
    mat.sum_duplicates()

    return mat
    
    
    
def write_linqt_csr(mat, path: str):
    """
    Write a scipy sparse matrix to a linqt .CSR text file.

    Format (4 lines):
        Ndim  nnz
        re0 im0 re1 im1 ...    (2*nnz floats, interleaved)
        col0 col1 ...          (nnz int column indices)
        ptr0 ptr1 ...          (Ndim+1 int row pointers)
    """
    mat = mat.tocsr()
    mat.sort_indices()

    Ndim = mat.shape[1]
    nnz  = mat.nnz
    data = mat.data  # complex128

    with open(path, 'w') as f:
        f.write(f"{Ndim} {nnz}\n")

        # Interleaved real / imag
        interleaved = np.empty(2 * nnz, dtype=np.float64)
        interleaved[0::2] = data.real
        interleaved[1::2] = data.imag
        f.write(' '.join(f'{v:.22f}' for v in interleaved))
        f.write('\n')

        f.write(' '.join(map(str, mat.indices)))
        f.write('\n')
        f.write(' '.join(map(str, mat.indptr)))
        f.write('\n')

    print(f"  Written: {path}  (Ndim={Ndim}, nnz={nnz})")


# ─────────────────────────────────────────────────────────────────────────────
# Stable S^{-1/2} via eigendecomposition
# ─────────────────────────────────────────────────────────────────────────────

def sqrt_inv(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute S^{-1/2} for a Hermitian positive-definite matrix S.

    Uses eigh (stable for Hermitian matrices) rather than sqrtm(inv(S)),
    which can accumulate errors when eigenvalues are small.

    S = U Λ U†  →  S^{-1/2} = U diag(λ^{-1/2}) U†
    """
    lam, U = np.linalg.eigh(S)
    # Guard against near-zero / negative eigenvalues (numerical noise)
    lam = np.maximum(lam, eps)
    return (U * (lam ** -0.5)[np.newaxis, :]) @ U.conj().T


# ─────────────────────────────────────────────────────────────────────────────
# Per-k Löwdin operators
# ─────────────────────────────────────────────────────────────────────────────

def lowdin_operators_k(H_source, k, gauge: str):
    """
    Compute Löwdin-orthogonalised H_ortho, V_x, V_y at a single k-point.

    Returns
    -------
    H_orth : (W,W) complex ndarray
    Vx     : (W,W) complex ndarray
    Vy     : (W,W) complex ndarray
    """
    # Dense matrices at this k
    Hk  = H_source.Hk (k, gauge=gauge).toarray().astype(complex)
    Sk  = H_source.Sk (k, gauge=gauge).toarray().astype(complex)
    dHk = H_source.dHk(k, gauge=gauge)     # list: [x, y, ...]
    dSk = H_source.dSk(k, gauge=gauge)

    dHk_x = dHk[0].toarray().astype(complex)
    dHk_y = dHk[1].toarray().astype(complex)
    dSk_x = dSk[0].toarray().astype(complex)
    dSk_y = dSk[1].toarray().astype(complex)

    # S^{-1/2}  and  S^{-1}
    Sinvh = sqrt_inv(Sk)           # S^{-1/2}
    Sinv  = Sinvh @ Sinvh          # S^{-1}  = (S^{-1/2})²

    # Löwdin Hamiltonian
    H_orth = Sinvh @ Hk @ Sinvh

    # Velocity:
    #   V^α = S^{-1/2} ∂H S^{-1/2}
    #         - ½ H_orth ∂S S^{-1}
    #         - ½ ∂S S^{-1} H_orth
    def velocity(dH, dS):
        return (Sinvh @ dH @ Sinvh
                - 0.5 * H_orth @ dS @ Sinv
                - 0.5 * dS @ Sinv @ H_orth)

    Vx = velocity(dHk_x, dSk_x)
    Vy = velocity(dHk_y, dSk_y)

    return H_orth, Vx, Vy


# ─────────────────────────────────────────────────────────────────────────────
# Build full block-diagonal supercell operators
# ─────────────────────────────────────────────────────────────────────────────

def build_operators(H_source, kx: int, ky: int, kz: int, gauge: str):
    """
    Loop over the kx×ky×kz grid, compute Löwdin operators at each k,
    and return block-diagonal sparse matrices.

    Returns
    -------
    H_blk  : scipy sparse CSR  (Nk*W, Nk*W)
    Vx_blk : scipy sparse CSR
    Vy_blk : scipy sparse CSR  (or None if compute_vy=False)
    """
    K1, K2, K3 = np.mgrid[0:kx, 0:ky, 0:kz]
    Ks = np.column_stack([
        K1.ravel() / kx,
        K2.ravel() / ky,
        K3.ravel() / kz,
    ])
    Nk = len(Ks)

    print(f"  k-grid: {kx}×{ky}×{kz} = {Nk} k-points")

    H_list  = []
    Vx_list = []
    Vy_list = []

    t0 = time.time()
    for ik, k in enumerate(Ks):
        H_orth, Vx, Vy = lowdin_operators_k(H_source, k, gauge)
        H_list .append(csr_matrix(H_orth))
        Vx_list.append(csr_matrix(Vx))
        Vy_list.append(csr_matrix(Vy))

        # Progress on same line
        elapsed = time.time() - t0
        rate    = (ik + 1) / elapsed if elapsed > 0 else 0
        eta     = (Nk - ik - 1) / rate if rate > 0 else 0
        print(f"  k-point {ik+1}/{Nk}  |  {rate:.1f} k/s  |  ETA {eta:.0f}s   ",
              end='\r', flush=True)

    print(f"\n  Done in {time.time()-t0:.1f}s")

    print("  Assembling block-diagonal matrices ...", end=' ', flush=True)
    H_blk  = block_diag(H_list,  format='csr')
    Vx_blk = block_diag(Vx_list, format='csr')
    Vy_blk = block_diag(Vy_list, format='csr')
    print("done")

    return H_blk, Vx_blk, Vy_blk, Ks


# ─────────────────────────────────────────────────────────────────────────────
# Bloch transform matrix  B  (atom gauge)
# ─────────────────────────────────────────────────────────────────────────────

def build_bloch_transform(H_source, Ks: np.ndarray,
                          kx: int, ky: int, kz: int) -> csr_matrix:
    """
    Build the Bloch transform matrix that maps the k-space block-diagonal
    basis to the real-space supercell basis (atom gauge).

    Matrix elements:
        B_{(ik,α), (iR,β)} = δ_{αβ} / √N_k × exp(i k_ik · (R_iR + τ_α))

    where:
        k_ik   – Cartesian k-vector of grid point ik  (1/Å, includes 2π)
        R_iR   – Cartesian position of unit cell iR within the supercell (Å)
        τ_α    – Cartesian position of the atom hosting orbital α (Å)

    The δ_{αβ} means the matrix is orbital-diagonal: only same-orbital
    entries are nonzero, so each row has exactly N_k non-zero elements.

    Parameters
    ----------
    H_source : sisl.Hamiltonian
    Ks       : (N_k, 3) array of fractional k-coordinates (from build_operators)
    kx,ky,kz : k-grid dimensions

    Returns
    -------
    B : (N_k*W, N_k*W) unitary sparse CSR matrix
    """
    rcell = H_source.geometry.rcell  # (3,3) reciprocal lattice, rows, with 2π factor
    cell  = H_source.geometry.cell   # (3,3) real-space lattice, rows (Å)
    W     = H_source.no              # orbitals per unit cell
    Nk    = len(Ks)
    N     = Nk * W

    # Map every orbital to the Cartesian position of its host atom
    # H_source.geometry.a2o(ia, all=True) gives orbital indices for atom ia
    xyz     = H_source.geometry.xyz          # (natoms, 3) in Å
    tau_orb = np.zeros((W, 3), dtype=float)
    for ia in range(H_source.na):
        orbs = H_source.geometry.a2o(ia, all=True)
        tau_orb[orbs] = xyz[ia]

    # Cartesian k-vectors:  k_cart[ik] = Ks[ik] @ rcell  (1/Å, with 2π)
    k_carts = Ks @ rcell                     # (Nk, 3)

    # Real-space positions of supercell unit cells
    # Ks stores fractional coords [n1/kx, n2/ky, n3/kz] → integer grid indices
    n_grid = np.round(Ks * np.array([kx, ky, kz])).astype(int)  # (Nk, 3)
    R_vecs = n_grid @ cell                   # (Nk, 3) in Å

    # Lattice part of the phase: phase_lat[ik, iR] = k_ik · R_iR
    phase_lat = k_carts @ R_vecs.T           # (Nk, Nk)

    inv_sqrt_Nk = 1.0 / np.sqrt(Nk)

    # Build in COO format.  For each orbital α, the subblock is a dense Nk×Nk
    # matrix placed at row-stride W (rows ik*W+α) and col-stride W (cols iR*W+α).
    nnz      = Nk * Nk * W
    row_arr  = np.empty(nnz, dtype=np.int64)
    col_arr  = np.empty(nnz, dtype=np.int64)
    data_arr = np.empty(nnz, dtype=np.complex128)

    ik_idx = np.arange(Nk, dtype=np.int64)
    iR_idx = np.arange(Nk, dtype=np.int64)

    for alpha in range(W):
        phase_atom  = k_carts @ tau_orb[alpha]
        phase_total = phase_lat + phase_atom[:, np.newaxis]
        B_alpha     = inv_sqrt_Nk * np.exp(1j * phase_total)   # (Nk, Nk)

        rows_flat = np.repeat(ik_idx * W + alpha, Nk)   # (Nk*Nk,)
        cols_flat = np.tile  (iR_idx * W + alpha, Nk)   # (Nk*Nk,)

        sl = slice(alpha * Nk * Nk, (alpha + 1) * Nk * Nk)
        row_arr [sl] = rows_flat
        col_arr [sl] = cols_flat
        data_arr[sl] = B_alpha.ravel()
        
        
        
        
    B = coo_matrix((data_arr, (row_arr, col_arr)), shape=(N, N)).tocsr()

    # Sanity check: B should be unitary  →  max|B†B - I| should be ~1e-12
    err = abs((B.conj().T @ B) - np.eye(N)).max()
    print(f"  Unitarity check  max|B†B - I| = {err:.2e}")

    return B


# ─────────────────────────────────────────────────────────────────────────────
# Supercell position operator and real-space velocity  [H_real, X]
# ─────────────────────────────────────────────────────────────────────────────

def build_supercell_positions(H_source, Ks: np.ndarray,
                              kx: int, ky: int, kz: int):
    """
    Build diagonal position operators X and Y for the full Löwdin real-space
    supercell.  The diagonal entry for site (iR, α) is the full Cartesian
    coordinate of atom α inside unit cell iR:

        x(iR, α) = R_iR[0] + τ_α[0]       (runs 0 → L_x across the supercell)
        y(iR, α) = R_iR[1] + τ_α[1]

    This is the macroscopic position — not the periodic unit-cell position —
    which is required for a correctly growing MSD.

    Parameters
    ----------
    H_source : sisl.Hamiltonian
    Ks       : (Nk, 3) fractional k-coordinates (same ordering as build_operators)
    kx,ky,kz : k-grid dimensions

    Returns
    -------
    x_coords, y_coords : (Nk*W,) float arrays of supercell coordinates (Å)
    X_op, Y_op         : diagonal CSR sparse matrices of shape (Nk*W, Nk*W)
    """
    from scipy.sparse import diags as sp_diags

    cell     = H_source.geometry.cell   # (3,3) real lattice, rows (Å)
    W        = H_source.no
    Nk       = len(Ks)
    N        = Nk * W

    # Map each orbital to the position of its host atom
    xyz      = H_source.geometry.xyz    # (natoms, 3) in Å
    tau_orb  = np.zeros((W, 3))
    for ia in range(H_source.na):
        orbs = H_source.geometry.a2o(ia, all=True)
        tau_orb[orbs] = xyz[ia]

    # Real-space unit cell positions R for each k-grid point
    n_grid  = np.round(Ks * np.array([kx, ky, kz])).astype(int)  # (Nk, 3)
    R_vecs  = n_grid @ cell                                         # (Nk, 3) Å

    # Full supercell coordinates: x(iR,α) = R_iR[0] + τ_α[0]
    iR_idx  = np.repeat(np.arange(Nk), W)   # [0,0,...,1,1,...,Nk-1,...]
    al_idx  = np.tile  (np.arange(W),  Nk)  # [0,1,...,W-1,0,1,...,W-1,...]

    x_coords = R_vecs[iR_idx, 0] + tau_orb[al_idx, 0]   # (N,)
    y_coords = R_vecs[iR_idx, 1] + tau_orb[al_idx, 1]   # (N,)

    X_op = sp_diags(x_coords, format='csr')
    Y_op = sp_diags(y_coords, format='csr')

    print(f"  Supercell x range: [{x_coords.min():.2f}, {x_coords.max():.2f}] Å")
    print(f"  Supercell y range: [{y_coords.min():.2f}, {y_coords.max():.2f}] Å")


    return x_coords, y_coords, X_op, Y_op


def build_real_space_velocity(H_real, x_coords: np.ndarray,
                              y_coords: np.ndarray, tol: float = 1e-6):
    """
    Compute real-space velocity operators as commutators with the macroscopic
    position:

        VX_real = [H_real, X_real]
        VY_real = [H_real, Y_real]

    For a diagonal position operator X, the commutator is element-wise:

        [H, X]_{ij} = H_{ij} * (x_j - x_i)

    so VX_real has exactly the same sparsity pattern as H_real — only the
    values change.  This is mathematically equivalent to B† (dH/dk_x) B in
    the atom gauge, but expressed in the real-space basis where x_j - x_i
    includes the full inter-unit-cell displacement R_j - R_i, not just the
    intra-cell part.  This is what enables the MSD to grow without bound.

    Parameters
    ----------
    H_real    : (N, N) sparse CSR — real-space Löwdin Hamiltonian
    x_coords  : (N,) supercell x-coordinates from build_supercell_positions
    y_coords  : (N,) supercell y-coordinates
    tol       : pruning threshold

    Returns
    -------
    VX_real, VY_real : sparse CSR matrices, same sparsity as H_real
    """
    from scipy.sparse import coo_matrix

    """
    H_coo = H_real.tocoo()
    N     = H_real.shape[0]

    # Bond vectors for each nonzero element H_{ij}
    dx = x_coords[H_coo.col] - x_coords[H_coo.row]   # x_j - x_i
    dy = y_coords[H_coo.col] - y_coords[H_coo.row]

    VX = coo_matrix((H_coo.data * dx, (H_coo.row, H_coo.col)),
                    shape=(N, N)).tocsr()
    VY = coo_matrix((H_coo.data * dy, (H_coo.row, H_coo.col)),
                    shape=(N, N)).tocsr()

    VX = prune_and_sort(VX, tol)
    VY = prune_and_sort(VY, tol)
    """
    x_op = x_coords.diag()
    y_op = y_coords.diag()
    
    VX = H_real @ x_op - x_op @ H_real
    VY = H_real @ y_op - y_op @ H_real
    
    return VX, VY


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Löwdin-orthogonalise a SIESTA Hamiltonian and write "
                    "H, Vx, Vy, and the Bloch transform matrix as linqt .CSR files."
    )
    parser.add_argument("fdf_file",        help="Path to SIESTA .fdf file")
    parser.add_argument("kx",  type=int,   help="k-points along a1")
    parser.add_argument("ky",  type=int,   help="k-points along a2")
    parser.add_argument("--kz",    type=int, default=1,
                        help="k-points along a3 (default: 1)")
    parser.add_argument("--prefix", default=None,
                        help="Output file prefix (default: fdf stem)")
    parser.add_argument("--gauge", default="r",
                        help="sisl gauge string (default: 'r')")
    args = parser.parse_args()

    fdf_path = Path(args.fdf_file)
    if not fdf_path.exists():
        sys.exit(f"Error: file not found: {fdf_path}")

    prefix = args.prefix if args.prefix else fdf_path.stem

    # ── Read Hamiltonian ──────────────────────────────────────────────────────
    print(f"Reading {fdf_path} ...")
    try:
        import sisl
        fdf   = sisl.get_sile(fdf_path)
        H_src = fdf.read_hamiltonian()
    except Exception as e:
        sys.exit(f"Error reading SIESTA file: {e}")

    print(f"  orbitals per cell (W): {H_src.no}")
    print(f"  spin class: {H_src.spin}")

    # ── Build Löwdin operators ────────────────────────────────────────────────
    print(f"\nBuilding Löwdin operators on {args.kx}×{args.ky}×{args.kz} grid ...")
    H_blk, Vx_blk, Vy_blk, Ks = build_operators(
        H_src, args.kx, args.ky, args.kz, gauge=args.gauge
    )

    # ── Build Bloch transform matrix ──────────────────────────────────────────
    print("\nBuilding Bloch transform matrix (atom gauge) ...")
    B = build_bloch_transform(H_src, Ks, args.kx, args.ky, args.kz)

    # ── Real-space Hamiltonian  H_real = B† H_blk B ───────────────────────────
    # This is the sparse Löwdin-orthogonalised Hamiltonian in the Wannier-like
    # real-space basis.  Its sparsity mirrors the original SIESTA hopping range.
    print("\nBuilding real-space Hamiltonian H_real = B† H_blk B ...")
    H_real = prune_and_sort(B.conj().T @ H_blk @ B)
    print(f"  H_real nnz = {H_real.nnz}")

    # ── Supercell position operators ──────────────────────────────────────────
    print("\nBuilding supercell position operators (full macroscopic coordinates) ...")
    x_coords, y_coords, X_op, Y_op = build_supercell_positions(
        H_src, Ks, args.kx, args.ky, args.kz)

    # ── Real-space velocity  [H_real, X]  ─────────────────────────────────────
    # [H, X]_{ij} = H_{ij} * (x_j - x_i)  — same sparsity as H_real.
    # The bond vector x_j - x_i includes the full inter-unit-cell displacement,
    # so MSD grows correctly without finite-size recurrence.
    print("\nBuilding real-space velocity operators [H_real, X] and [H_real, Y] ...")
    VX_real, VY_real = build_real_space_velocity(H_real, x_coords, y_coords)

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting CSR files ...")
    write_linqt_csr(H_blk,   f"{prefix}.HAM.CSR")
    write_linqt_csr(Vx_blk,  f"{prefix}.VX.CSR")
    write_linqt_csr(Vy_blk,  f"{prefix}.VY.CSR")
    write_linqt_csr(B,       f"{prefix}.BLOCH.CSR")
    write_linqt_csr(H_real,  f"{prefix}.HREAL.CSR")
    write_linqt_csr(VX_real, f"{prefix}.VXREAL.CSR")
    write_linqt_csr(VY_real, f"{prefix}.VYREAL.CSR")

    print("\nAll done.")


if __name__ == "__main__":
    main()
