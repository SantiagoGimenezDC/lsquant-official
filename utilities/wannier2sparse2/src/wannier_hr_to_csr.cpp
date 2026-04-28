/**
 * wannier_hr_to_csr.cpp
 *
 * Converts a Wannier90 *_hr.dat Hamiltonian into a supercell CSR sparse
 * matrix written in the linqt .CSR text format.
 *
 * Uses Eigen's SparseMatrix + setFromTriplets with a last-value-wins
 * duplicate policy to cleanly handle any repeated (row,col) entries.
 *
 * Compile (Eigen header-only, just point to it):
 *   g++ -O3 -std=c++17 -I/path/to/eigen -o wannier_hr_to_csr wannier_hr_to_csr.cpp
 *
 * Usage:
 *   ./wannier_hr_to_csr <hr_file> <N> <M> <P> [output_prefix]
 *
 * Output:
 *   <prefix>.CSR       – linqt text format
 *   <prefix>_info.txt  – summary
 *
 * .CSR format (4 lines):
 *   Ndim  nnz
 *   re0 im0 re1 im1 ...     (2*nnz floats, interleaved)
 *   col0 col1 ...           (nnz integers)
 *   ptr0 ptr1 ...           (Ndim+1 integers)
 */

#include <eigen3/Eigen/Sparse>

#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using Cplx     = std::complex<double>;
using SpMat    = Eigen::SparseMatrix<Cplx, Eigen::RowMajor>;
using Triplet  = Eigen::Triplet<Cplx>;

// ---------------------------------------------------------------------------
// Hamiltonian data read from _hr.dat
// ---------------------------------------------------------------------------
struct HoppingEntry {
    int Rx, Ry, Rz;   // lattice vector
    int m, n;          // orbital indices (0-based)
    Cplx val;
};

struct HrData {
    int num_wann = 0;
    int nrpts    = 0;
    std::vector<HoppingEntry> hoppings;
};

// ---------------------------------------------------------------------------
// Parse _hr.dat
// ---------------------------------------------------------------------------
HrData parse_hr(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open: " + filename);

    HrData hr;
    std::string line;

    // Line 1: comment
    std::getline(f, line);

    // Line 2: num_wann
    std::getline(f, line);
    hr.num_wann = std::stoi(line);

    // Line 3: nrpts
    std::getline(f, line);
    hr.nrpts = std::stoi(line);

    // Degeneracy weights – read and discard (nrpts integers, 15 per line)
    {
        int read = 0;
        while (read < hr.nrpts) {
            std::getline(f, line);
            std::istringstream ss(line);
            int w;
            while (ss >> w && read < hr.nrpts) ++read;
        }
    }

    // Hopping elements: Rx Ry Rz m n Re Im
    hr.hoppings.reserve(hr.nrpts * hr.num_wann * hr.num_wann);
    int Rx, Ry, Rz, m, n;
    double re, im;
    while (f >> Rx >> Ry >> Rz >> m >> n >> re >> im) {
        hr.hoppings.push_back({Rx, Ry, Rz, m - 1, n - 1, Cplx(re, im)});
    }

    return hr;
}

// ---------------------------------------------------------------------------
// Supercell flat indices
// ---------------------------------------------------------------------------
inline int cell_flat(int ix, int iy, int iz, int M, int P) {
    return ix * (M * P) + iy * P + iz;
}
inline int global_idx(int cell, int orb, int W) {
    return cell * W + orb;
}

// ---------------------------------------------------------------------------
// Build Eigen sparse matrix via triplet list
// Duplicates (same row,col from different R-vectors wrapping to the same
// supercell image) are resolved with last-entry-wins via a custom functor.
// ---------------------------------------------------------------------------
SpMat build_sparse(const HrData& hr, int N, int M, int P)
{
    const int W      = hr.num_wann;
    const int Ncells = N * M * P;
    const int Ndim   = Ncells * W;

    // Reserve: each hopping entry replicates once per unit cell
    std::vector<Triplet> triplets;
    triplets.reserve(hr.hoppings.size() * Ncells);

    for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < M; ++iy)
    for (int iz = 0; iz < P; ++iz) {
        const int src_cell = cell_flat(ix, iy, iz, M, P);

        for (const auto& hop : hr.hoppings) {
            if (hop.val == Cplx(0.0)) continue;

            // Destination cell with PBC
            const int jx = ((ix + hop.Rx) % N + N) % N;
            const int jy = ((iy + hop.Ry) % M + M) % M;
            const int jz = ((iz + hop.Rz) % P + P) % P;
            const int dst_cell = cell_flat(jx, jy, jz, M, P);

            const int row = global_idx(src_cell, hop.m, W);
            const int col = global_idx(dst_cell, hop.n, W);

            triplets.emplace_back(row, col, hop.val);
        }
    }

    SpMat mat(Ndim, Ndim);

    // setFromTriplets with a custom duplicate handler: last value wins.
    // The functor receives (existing, incoming) and returns the one to keep.
    mat.setFromTriplets(
        triplets.begin(), triplets.end(),
        [](const Cplx& /*existing*/, const Cplx& incoming) {
            return incoming;   // last-added entry wins
        }
    );

    return mat;
}

// ---------------------------------------------------------------------------
// Write linqt .CSR text format
//   Line 1: Ndim  nnz
//   Line 2: re0 im0 re1 im1 ...    (interleaved, 2*nnz values)
//   Line 3: col0 col1 ...          (nnz column indices)
//   Line 4: ptr0 ptr1 ...          (Ndim+1 row pointers)
// ---------------------------------------------------------------------------
void write_linqt_csr(const std::string& path, const SpMat& mat)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << std::setprecision(22) << std::fixed;

    const int Ndim = mat.rows();
    const int nnz  = mat.nonZeros();

    // Line 1
    f << Ndim << " " << nnz << "\n";

    // Line 2: interleaved real/imag over non-zeros (row-major order)
    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.value().real() << " " << it.value().imag() << " ";
    f << "\n";

    // Line 3: column indices
    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.col() << " ";
    f << "\n";

    // Line 4: row pointers  (outerIndexPtr has Ndim+1 entries for RowMajor)
    const int* outerPtr = mat.outerIndexPtr();
    for (int i = 0; i <= Ndim; ++i)
        f << outerPtr[i] << " ";
    f << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <hr_file> <N> <M> <P> [output_prefix]\n";
        return 1;
    }

    const std::string hr_file = argv[1];
    const int N = std::stoi(argv[2]);
    const int M = std::stoi(argv[3]);
    const int P = std::stoi(argv[4]);
    const std::string prefix = (argc >= 6) ? argv[5] : "supercell";

    if (N <= 0 || M <= 0 || P <= 0)
        throw std::invalid_argument("N, M, P must be positive");

    std::cout << "Reading " << hr_file << " ...\n";
    HrData hr = parse_hr(hr_file);
    std::cout << "  num_wann = " << hr.num_wann
              << "  nrpts = "    << hr.nrpts
              << "  entries = "  << hr.hoppings.size() << "\n";

    std::cout << "Building " << N << "x" << M << "x" << P
              << " supercell ...\n";
    SpMat mat = build_sparse(hr, N, M, P);

    const long Ndim = mat.rows();
    const long nnz  = mat.nonZeros();
    std::cout << "  Ndim = " << Ndim << "\n"
              << "  nnz  = " << nnz  << "\n"
              << "  fill = " << (double)nnz / ((double)Ndim * Ndim) << "\n";

    const std::string csr_path = prefix + ".CSR";
    std::cout << "Writing " << csr_path << " ...\n";
    write_linqt_csr(csr_path, mat);

    {
        std::ofstream info(prefix + "_info.txt");
        info << "supercell: " << N << " x " << M << " x " << P << "\n"
             << "num_wann:  " << hr.num_wann << "\n"
             << "nrpts:     " << hr.nrpts    << "\n"
             << "Ndim:      " << Ndim        << "\n"
             << "nnz:       " << nnz         << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
