#!/usr/bin/env python3
"""
siesta2linqt.py

Reads a SIESTA .fdf file, builds a k-point grid of size kx × ky × kz,
performs Löwdin orthogonalisation at each k-point, assembles the full
supercell (block-diagonal) operators, and writes them as linqt .CSR files:

    <prefix>.HAM.CSR    – Löwdin-orthogonalised Hamiltonian  H_ortho
    <prefix>.VX.CSR     – velocity operator V_x
    <prefix>.VY.CSR     – velocity operator V_y
    <prefix>.BLOCH.CSR  – Bloch transform matrix B (atom gauge)

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
    
    
# ─────────────────────────────────────────────────────────────────────────────
# CSR writer  (linqt format)
# ─────────────────────────────────────────────────────────────────────────────

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

        # Explicit (Nk, Nk) index arrays via meshgrid
        rows_2d, cols_2d = np.meshgrid(
            ik_idx * W + alpha,   # row indices: ik*W + alpha
            iR_idx * W + alpha,   # col indices: iR*W + alpha
            indexing='ij'         # rows_2d[ik, iR], cols_2d[ik, iR]
        )

        sl = slice(alpha * Nk * Nk, (alpha + 1) * Nk * Nk)
        row_arr [sl] = rows_2d.ravel()
        col_arr [sl] = cols_2d.ravel()
        data_arr[sl] = B_alpha.ravel()
        

    B = coo_matrix((data_arr, (row_arr, col_arr)), shape=(N, N)).tocsr()

    # Sanity check: B should be unitary  →  max|B†B - I| should be ~1e-12
    err = abs((B.conj().T @ B) - np.eye(N)).max()
    print(f"  Unitarity check  max|B†B - I| = {err:.2e}")

    return B


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
                        
    parser.add_argument("--Bloch", default=False,
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

    geom = H_src.geometry

    # Number of atoms
    na = geom.na

    # Lattice vectors (3x3 array)
    cell = geom.lattice.cell
    

    print(f"  orbitals per cell (W): {H_src.no}")
    print(f"  spin class: {H_src.spin}")
    
    print("   The area per atom is: ", (cell[0,0]*cell[1,1] - cell[0,1]*cell[1,0])/na )

    # ── Build Löwdin operators ────────────────────────────────────────────────
    print(f"\nBuilding Löwdin operators on {args.kx}×{args.ky}×{args.kz} grid ...")
    H_blk, Vx_blk, Vy_blk, Ks = build_operators(
        H_src, args.kx, args.ky, args.kz, gauge=args.gauge
    )




    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting CSR files ...")
    write_linqt_csr(H_blk,  f"{prefix}.HAM.CSR")
    write_linqt_csr(Vx_blk, f"{prefix}.VX.CSR")
    write_linqt_csr(Vy_blk, f"{prefix}.VY.CSR")
    
    if(bool(args.Bloch) == True):    
    	# ── Build Bloch transform matrix ──────────────────────────────────────────
    	print("\nBuilding Bloch transform matrix (atom gauge) ...")
    	B = build_bloch_transform(H_src, Ks, args.kx, args.ky, args.kz)

    	write_linqt_csr(B,      f"{prefix}.BLOCH.CSR")

    	# ── Write test outputs ─────────────────────────────────────────────────────────
    	Hr_blk  = prune_and_sort(B.conj().T @ H_blk  @ B)
    	Vxr_blk = prune_and_sort(B.conj().T @ Vx_blk @ B)
    	Vyr_blk = prune_and_sort(B.conj().T @ Vy_blk @ B)
        
    	print("\nWriting CSR files ...")
    	write_linqt_csr(Hr_blk,  f"{prefix}_r.HAM.CSR")
    	write_linqt_csr(Vxr_blk, f"{prefix}_r.VX.CSR")
    	write_linqt_csr(Vyr_blk, f"{prefix}_r.VY.CSR")
     
     
     
    
    
    print("\nAll done.")


if __name__ == "__main__":
    main()
