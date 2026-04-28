/**
 * wannier_velocity_to_csr.cpp
 *
 * Builds the supercell velocity operators  v_x = (i/hbar)[H,X]
 * and v_y = (i/hbar)[H,Y]  in the Wannier basis and writes them
 * as linqt .CSR text files.
 *
 * Matrix element formula (Peierls / diagonal-position approximation):
 *
 *   (v_alpha)_{mn}(R) = (i/hbar) * (x_n^alpha + R_alpha - x_m^alpha) * H_{mn}(R)
 *
 * where x_m^alpha is the alpha-Cartesian component of the m-th Wannier
 * centre, and R_alpha = R . a_alpha is the Cartesian projection of the
 * lattice vector R along direction alpha.
 *
 * Compile:
 *   g++ -O3 -std=c++17 -I/usr/include/eigen3 -o wannier_velocity_to_csr wannier_velocity_to_csr.cpp
 *
 * Usage:
 *   ./wannier_velocity_to_csr <hr_file> <cell_file> <wann_centres_xyz> <N> <M> <P> [prefix]
 *
 * Inputs:
 *   hr_file           – Wannier90 *_hr.dat
 *   cell_file         – 3 lines, each with 3 floats: the lattice vectors a1,a2,a3
 *   wann_centres_xyz  – Wannier90 *_centres.xyz  (first line = num_wann,
 *                       second line = comment, then  "X  x  y  z"  per centre)
 *
 * Outputs:
 *   <prefix>_vx.CSR
 *   <prefix>_vy.CSR
 *   <prefix>_info.txt
 *
 * .CSR format (4 lines):
 *   Ndim  nnz
 *   re0 im0 re1 im1 ...    (2*nnz values, interleaved)
 *   col0 col1 ...          (nnz column indices)
 *   ptr0 ptr1 ...          (Ndim+1 row pointers)
 *
 * NOTE: hbar is set to 1 (atomic / Wannier units).  If you need SI,
 *       divide each matrix element by hbar = 1.054571817e-34 J·s.
 */

#include <eigen3/Eigen/Sparse>

#include <array>
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
using Cplx    = std::complex<double>;
using SpMat   = Eigen::SparseMatrix<Cplx, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<Cplx>;
using Vec3    = std::array<double, 3>;

// ---------------------------------------------------------------------------
// Parse lattice vectors  (3 lines, 3 floats each)
// Returns cell[i][j] = j-th Cartesian component of i-th lattice vector
// ---------------------------------------------------------------------------
std::array<Vec3, 3> parse_cell(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open cell file: " + filename);

    std::array<Vec3, 3> cell{};
    for (int i = 0; i < 3; ++i)
        if (!(f >> cell[i][0] >> cell[i][1] >> cell[i][2]))
            throw std::runtime_error("Cell file too short");
    return cell;
}

// ---------------------------------------------------------------------------
// Parse Wannier centres xyz file
// Line 1: num_wann
// Line 2: comment
// Lines 3..: "X  x  y  z"
// Returns centres[m][alpha] in Cartesian Angstrom
// ---------------------------------------------------------------------------
std::vector<Vec3> parse_centres(const std::string& filename, int expected_wann)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open centres file: " + filename);

    std::string line;
    // Line 1: num_wann
    std::getline(f, line);
    const int nw = std::stoi(line);
    if (nw != expected_wann)
        throw std::runtime_error("centres file has " + std::to_string(nw) +
                                 " centres but _hr.dat has num_wann = " +
                                 std::to_string(expected_wann));
    // Line 2: comment
    std::getline(f, line);

    std::vector<Vec3> centres(nw);
    std::string label;
    for (int m = 0; m < nw; ++m) {
        if (!(f >> label >> centres[m][0] >> centres[m][1] >> centres[m][2]))
            throw std::runtime_error("centres file too short at entry " +
                                     std::to_string(m));
    }
    return centres;
}

// ---------------------------------------------------------------------------
// Parse _hr.dat
// ---------------------------------------------------------------------------
struct HoppingEntry {
    int Rx, Ry, Rz;
    int m, n;        // 0-based
    Cplx val;
};

struct HrData {
    int num_wann = 0;
    int nrpts    = 0;
    std::vector<HoppingEntry> hoppings;
};

HrData parse_hr(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open: " + filename);

    HrData hr;
    std::string line;

    std::getline(f, line);          // comment
    std::getline(f, line); hr.num_wann = std::stoi(line);
    std::getline(f, line); hr.nrpts    = std::stoi(line);

    // Degeneracy weights – discard
    {
        int read = 0;
        while (read < hr.nrpts) {
            std::getline(f, line);
            std::istringstream ss(line);
            int w;
            while (ss >> w && read < hr.nrpts) ++read;
        }
    }

    hr.hoppings.reserve(static_cast<size_t>(hr.nrpts) * hr.num_wann * hr.num_wann);
    int Rx, Ry, Rz, m, n;
    double re, im;
    while (f >> Rx >> Ry >> Rz >> m >> n >> re >> im)
        hr.hoppings.push_back({Rx, Ry, Rz, m - 1, n - 1, Cplx(re, im)});

    return hr;
}

// ---------------------------------------------------------------------------
// Supercell index helpers
// ---------------------------------------------------------------------------
inline int cell_flat(int ix, int iy, int iz, int M, int P) {
    return ix * (M * P) + iy * P + iz;
}
inline int global_idx(int cell, int orb, int W) {
    return cell * W + orb;
}

// ---------------------------------------------------------------------------
// Build velocity operator for one Cartesian direction alpha (0=x, 1=y, 2=z)
//
//   (v_alpha)_{mn}(R, supercell) = i * (x_n^alpha + R_alpha - x_m^alpha) * H_{mn}(R)
//
// where R_alpha = Rx*a1[alpha] + Ry*a2[alpha] + Rz*a3[alpha]
// ---------------------------------------------------------------------------
SpMat build_velocity(const HrData&            hr,
                     const std::array<Vec3,3>& cell,
                     const std::vector<Vec3>&  centres,
                     int N, int M, int P,
                     int alpha)
{
    const int W      = hr.num_wann;
    const int Ndim   = N * M * P * W;

    std::vector<Triplet> triplets;
    triplets.reserve(hr.hoppings.size() * (size_t)(N * M * P));

    // Precompute Cartesian component of each lattice vector along alpha
    // a_alpha[i] = cell[i][alpha]
    const double a1a = cell[0][alpha];
    const double a2a = cell[1][alpha];
    const double a3a = cell[2][alpha];

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

            // Cartesian displacement along alpha:
            //   delta = (x_n + R_alpha) - x_m
            // where R_alpha is the *bare* lattice vector (before PBC folding)
            // because H_{mn}(R) encodes the coupling across exactly R unit cells.
            const double R_alpha = hop.Rx * a1a + hop.Ry * a2a + hop.Rz * a3a;
            const double delta   = centres[hop.n][alpha] + R_alpha
                                 - centres[hop.m][alpha];

            // v_alpha = i * delta * H
            const Cplx v_val = Cplx(0.0, 1.0) * delta * hop.val;

            if (v_val == Cplx(0.0)) continue;

            const int row = global_idx(src_cell, hop.m, W);
            const int col = global_idx(dst_cell, hop.n, W);
            triplets.emplace_back(row, col, v_val);
        }
    }

    SpMat mat(Ndim, Ndim);
    mat.setFromTriplets(
        triplets.begin(), triplets.end(),
        [](const Cplx& /*existing*/, const Cplx& incoming) {
            return incoming;   // last-added wins (consistent with H builder)
        }
    );
    return mat;
}

// ---------------------------------------------------------------------------
// Write linqt .CSR text format
// ---------------------------------------------------------------------------
void write_linqt_csr(const std::string& path, const SpMat& mat)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << std::setprecision(22) << std::fixed;

    const int  Ndim     = mat.rows();
    const int  nnz      = mat.nonZeros();
    const int* outerPtr = mat.outerIndexPtr();

    // Line 1
    f << Ndim << " " << nnz << "\n";

    // Line 2: interleaved re/im
    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.value().real() << " " << it.value().imag() << " ";
    f << "\n";

    // Line 3: column indices
    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.col() << " ";
    f << "\n";

    // Line 4: row pointers
    for (int i = 0; i <= Ndim; ++i)
        f << outerPtr[i] << " ";
    f << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <hr_file> <cell_file> <centres_xyz>"
                  << " <N> <M> <P> [prefix]\n";
        return 1;
    }

    const std::string hr_file      = argv[1];
    const std::string cell_file    = argv[2];
    const std::string centres_file = argv[3];
    const int N = std::stoi(argv[4]);
    const int M = std::stoi(argv[5]);
    const int P = std::stoi(argv[6]);
    const std::string prefix = (argc >= 8) ? argv[7] : "supercell";

    if (N <= 0 || M <= 0 || P <= 0)
        throw std::invalid_argument("N, M, P must be positive");

    std::cout << "Reading " << hr_file << " ...\n";
    HrData hr = parse_hr(hr_file);
    std::cout << "  num_wann = " << hr.num_wann
              << "  nrpts = "    << hr.nrpts
              << "  entries = "  << hr.hoppings.size() << "\n";

    std::cout << "Reading " << cell_file << " ...\n";
    auto cell = parse_cell(cell_file);
    std::cout << "  a1 = [" << cell[0][0] << ", " << cell[0][1] << ", " << cell[0][2] << "]\n"
              << "  a2 = [" << cell[1][0] << ", " << cell[1][1] << ", " << cell[1][2] << "]\n"
              << "  a3 = [" << cell[2][0] << ", " << cell[2][1] << ", " << cell[2][2] << "]\n";

    std::cout << "Reading " << centres_file << " ...\n";
    auto centres = parse_centres(centres_file, hr.num_wann);
    std::cout << "  " << centres.size() << " Wannier centres loaded\n";

    const long Ndim = (long)N * M * P * hr.num_wann;
    std::cout << "Supercell " << N << "x" << M << "x" << P
              << "  Ndim = " << Ndim << "\n";

    // Build and write v_x
    std::cout << "Building v_x ...\n";
    {
        SpMat vx = build_velocity(hr, cell, centres, N, M, P, 0);
        std::cout << "  nnz = " << vx.nonZeros() << "\n";
        write_linqt_csr(prefix + "_vx.CSR", vx);
        std::cout << "  Written: " << prefix << "_vx.CSR\n";
    }

    // Build and write v_y
    std::cout << "Building v_y ...\n";
    {
        SpMat vy = build_velocity(hr, cell, centres, N, M, P, 1);
        std::cout << "  nnz = " << vy.nonZeros() << "\n";
        write_linqt_csr(prefix + "_vy.CSR", vy);
        std::cout << "  Written: " << prefix << "_vy.CSR\n";
    }

    {
        std::ofstream info(prefix + "_info.txt");
        info << "supercell: " << N << " x " << M << " x " << P << "\n"
             << "num_wann:  " << hr.num_wann << "\n"
             << "nrpts:     " << hr.nrpts    << "\n"
             << "Ndim:      " << Ndim        << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
