/**
 * wannier_to_csr.cpp
 *
 * Builds supercell CSR sparse matrices for the Hamiltonian H, velocity
 * operators Vx/Vy/Vz, spin operators Sx/Sy/Sz, and spin-current operators
 * {Sα, Vβ}/2  for all nine (α,β) combinations, from Wannier90 output files.
 *
 * All matrices are written in the linqt .CSR text format.
 *
 * Physics
 * -------
 * Hamiltonian (atom gauge, PBC supercell):
 *   H_{mn}(R_supercell) = H_{mn}(R_unit_cell)
 *   with periodic images folded by  j = (i + R) mod (N,M,P)
 *
 * Velocity (atom gauge / Peierls):
 *   (Vα)_{mn}(R) = i * (τ_n^α + R_α - τ_m^α) * H_{mn}(R)
 *   where R_α = Rx*a1_α + Ry*a2_α + Rz*a3_α
 *
 * Spin operators (hbar=1, ordering: |↑,0⟩|↓,0⟩|↑,1⟩|↓,1⟩...):
 *   Sx = ½ σx ⊗ I_norb ,  Sy = ½ σy ⊗ I_norb ,  Sz = ½ σz ⊗ I_norb
 *   These are block-diagonal and k-independent → same in every unit cell.
 *
 * Spin-current (symmetrised anticommutator = standard spin-Hall operator):
 *   J^α_β = ½ {Sα, Vβ} = ½ (Sα Vβ + Vβ Sα)
 *
 * Compile:
 *   g++ -O3 -std=c++17 -I/usr/include/eigen3 -o wannier_to_csr wannier_to_csr.cpp
 *
 * Usage:
 *   ./wannier_to_csr <hr_file> <cell_file> <centres_xyz> <N> <M> <P> \
 *                   [--H] [--Vx] [--Vy] [--Vz]                       \
 *                   [--Sx] [--Sy] [--Sz]                              \
 *                   [--SxVx] [--SxVy] [--SxVz]                       \
 *                   [--SyVx] [--SyVy] [--SyVz]                       \
 *                   [--SzVx] [--SzVy] [--SzVz]                       \
 *                   [--prefix <name>]
 *
 *   If no operator flags are given, only H is written.
 *
 * Output files:  <prefix>_H.CSR,  <prefix>_Vx.CSR,  <prefix>_SxVy.CSR, ...
 *
 * .CSR format (4 lines):
 *   Ndim  nnz
 *   re0 im0 re1 im1 ...   (2*nnz values, interleaved)
 *   col0 col1 ...         (nnz column indices)
 *   ptr0 ptr1 ...         (Ndim+1 row pointers)
 */

#include <eigen3/Eigen/Sparse>

#include <array>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using Cplx    = std::complex<double>;
using SpMat   = Eigen::SparseMatrix<Cplx, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<Cplx>;
using Vec3    = std::array<double, 3>;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct HoppingEntry {
    int  Rx, Ry, Rz;   // lattice vector (unit cells)
    int  m, n;          // orbital indices, 0-based
    Cplx val;
};

struct HrData {
    int num_wann = 0;
    int nrpts    = 0;
    std::vector<HoppingEntry> hoppings;
};

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------
HrData parse_hr(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open: " + filename);

    HrData hr;
    std::string line;

    std::getline(f, line);                          // comment
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

std::vector<Vec3> parse_centres(const std::string& filename, int expected)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open centres file: " + filename);
    std::string line;
    std::getline(f, line);
    const int nw = std::stoi(line);
    if (nw != expected)
        throw std::runtime_error("centres/hr num_wann mismatch");
    std::getline(f, line);   // comment

    std::vector<Vec3> c(nw);
    std::string label;
    for (int m = 0; m < nw; ++m)
        if (!(f >> label >> c[m][0] >> c[m][1] >> c[m][2]))
            throw std::runtime_error("centres file too short");
    return c;
}

// ---------------------------------------------------------------------------
// Supercell index helpers
// ---------------------------------------------------------------------------
inline int cell_flat(int ix, int iy, int iz, int M, int P)
{ return ix*(M*P) + iy*P + iz; }

inline int global_idx(int cell, int orb, int W)
{ return cell*W + orb; }

// ---------------------------------------------------------------------------
// Sparse matrix assembly from triplets (last-value-wins for duplicates)
// ---------------------------------------------------------------------------
SpMat from_triplets(int Ndim, std::vector<Triplet>& triplets)
{
    SpMat mat(Ndim, Ndim);
    mat.setFromTriplets(triplets.begin(), triplets.end(),
        [](const Cplx&, const Cplx& inc){ return inc; });
    return mat;
}

// ---------------------------------------------------------------------------
// Build H supercell
// ---------------------------------------------------------------------------
SpMat build_H(const HrData& hr, int N, int M, int P)
{
    const int W    = hr.num_wann;
    const int Ndim = N*M*P*W;

    std::vector<Triplet> tr;
    tr.reserve(hr.hoppings.size() * (size_t)(N*M*P));

    for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < M; ++iy)
    for (int iz = 0; iz < P; ++iz) {
        const int src = cell_flat(ix,iy,iz,M,P);
        for (const auto& h : hr.hoppings) {
            if (h.val == Cplx(0)) continue;
            const int jx = ((ix+h.Rx)%N+N)%N;
            const int jy = ((iy+h.Ry)%M+M)%M;
            const int jz = ((iz+h.Rz)%P+P)%P;
            const int dst = cell_flat(jx,jy,jz,M,P);
            tr.emplace_back(global_idx(src,h.m,W), global_idx(dst,h.n,W), h.val);
        }
    }
    return from_triplets(Ndim, tr);
}

// ---------------------------------------------------------------------------
// Build Vα supercell  (atom gauge)
//   (Vα)_mn(R) = i*(τ_n^α + R_α - τ_m^α) * H_mn(R)
// ---------------------------------------------------------------------------
SpMat build_V(const HrData& hr, const std::array<Vec3,3>& cell,
              const std::vector<Vec3>& centres, int N, int M, int P, int alpha)
{
    const int W    = hr.num_wann;
    const int Ndim = N*M*P*W;

    const double a1a = cell[0][alpha];
    const double a2a = cell[1][alpha];
    const double a3a = cell[2][alpha];

    std::vector<Triplet> tr;
    tr.reserve(hr.hoppings.size() * (size_t)(N*M*P));

    for (int ix = 0; ix < N; ++ix)
    for (int iy = 0; iy < M; ++iy)
    for (int iz = 0; iz < P; ++iz) {
        const int src = cell_flat(ix,iy,iz,M,P);
        for (const auto& h : hr.hoppings) {
            if (h.val == Cplx(0)) continue;
            const int jx = ((ix+h.Rx)%N+N)%N;
            const int jy = ((iy+h.Ry)%M+M)%M;
            const int jz = ((iz+h.Rz)%P+P)%P;
            const int dst = cell_flat(jx,jy,jz,M,P);
            const double R_a = h.Rx*a1a + h.Ry*a2a + h.Rz*a3a;
            const double delta = centres[h.n][alpha] + R_a - centres[h.m][alpha];
            const Cplx v = Cplx(0,1)*delta*h.val;
            if (v == Cplx(0)) continue;
            tr.emplace_back(global_idx(src,h.m,W), global_idx(dst,h.n,W), v);
        }
    }
    return from_triplets(Ndim, tr);
}

// ---------------------------------------------------------------------------
// Build Sα supercell  (block-diagonal, same 2×2 block per unit cell)
//
// Spin ordering: |↑,0⟩|↓,0⟩|↑,1⟩|↓,1⟩...  →  even=up, odd=down
//
//   Sx: (2k,2k+1) = (2k+1,2k) = ½   for each orbital k
//   Sy: (2k,2k+1) = -i/2,  (2k+1,2k) = +i/2
//   Sz: (2k,2k)   = +½,    (2k+1,2k+1) = -½
// ---------------------------------------------------------------------------
SpMat build_S(int W_uc, int N, int M, int P, int alpha)
{
    if (W_uc % 2 != 0)
        throw std::runtime_error("num_wann is odd: cannot build spin operators");

    const int Ncells = N*M*P;
    const int W      = W_uc;          // Wannier functions per unit cell
    const int Ndim   = Ncells*W;
    const int norb   = W/2;           // spatial orbitals per cell

    std::vector<Triplet> tr;
    tr.reserve(Ncells * norb * 2);    // each orbital contributes ≤2 entries per Sα

    for (int ic = 0; ic < Ncells; ++ic) {
        for (int k = 0; k < norb; ++k) {
            const int up  = global_idx(ic, 2*k,   W);   // even = spin up
            const int dn  = global_idx(ic, 2*k+1, W);   // odd  = spin down

            if (alpha == 0) {        // Sx
                tr.emplace_back(up, dn, Cplx( 0.5, 0));
                tr.emplace_back(dn, up, Cplx( 0.5, 0));
            } else if (alpha == 1) { // Sy
                tr.emplace_back(up, dn, Cplx(0, -0.5));
                tr.emplace_back(dn, up, Cplx(0,  0.5));
            } else {                 // Sz
                tr.emplace_back(up, up, Cplx( 0.5, 0));
                tr.emplace_back(dn, dn, Cplx(-0.5, 0));
            }
        }
    }
    return from_triplets(Ndim, tr);
}

// ---------------------------------------------------------------------------
// Build spin-current  J^α_β = ½ {Sα, Vβ}  =  ½(Sα·Vβ + Vβ·Sα)
// ---------------------------------------------------------------------------
SpMat build_SV(const SpMat& Sa, const SpMat& Vb)
{
    // Eigen sparse arithmetic: this allocates but stays sparse
    SpMat result = 0.5 * (Sa * Vb + Vb * Sa);
    result.makeCompressed();
    return result;
}

// ---------------------------------------------------------------------------
// Write linqt .CSR text format
// ---------------------------------------------------------------------------
void write_csr(const std::string& path, const SpMat& mat)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << std::setprecision(22) << std::fixed;

    const int  Ndim     = mat.rows();
    const int  nnz      = mat.nonZeros();
    const int* outerPtr = mat.outerIndexPtr();

    f << Ndim << " " << nnz << "\n";

    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.value().real() << " " << it.value().imag() << " ";
    f << "\n";

    for (int r = 0; r < Ndim; ++r)
        for (SpMat::InnerIterator it(mat, r); it; ++it)
            f << it.col() << " ";
    f << "\n";

    for (int i = 0; i <= Ndim; ++i)
        f << outerPtr[i] << " ";
    f << "\n";
}

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
bool has_flag(int argc, char* argv[], const std::string& flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

std::string get_option(int argc, char* argv[], const std::string& flag,
                       const std::string& def)
{
    for (int i = 1; i < argc-1; ++i)
        if (std::string(argv[i]) == flag) return argv[i+1];
    return def;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 7) {
        std::cerr <<
            "Usage: " << argv[0] << "\n"
            "  <hr_file> <cell_file> <centres_xyz> <N> <M> <P>\n"
            "  [--H]                          Hamiltonian\n"
            "  [--Vx] [--Vy] [--Vz]          Velocity components\n"
            "  [--Sx] [--Sy] [--Sz]          Spin components\n"
            "  [--SxVx] ... [--SzVz]         Spin-current {Sα,Vβ}/2\n"
            "  [--prefix <name>]              Output prefix (default: supercell)\n"
            "\n"
            "  If no operator flags are given, --H is assumed.\n";
        return 1;
    }

    const std::string hr_file      = argv[1];
    const std::string cell_file    = argv[2];
    const std::string centres_file = argv[3];
    const int N = std::stoi(argv[4]);
    const int M = std::stoi(argv[5]);
    const int P = std::stoi(argv[6]);
    const std::string prefix = get_option(argc, argv, "--prefix", "supercell");

    // Which operators to build
    const bool do_H   = has_flag(argc, argv, "--H");
    const bool do_Vx  = has_flag(argc, argv, "--Vx");
    const bool do_Vy  = has_flag(argc, argv, "--Vy");
    const bool do_Vz  = has_flag(argc, argv, "--Vz");
    const bool do_Sx  = has_flag(argc, argv, "--Sx");
    const bool do_Sy  = has_flag(argc, argv, "--Sy");
    const bool do_Sz  = has_flag(argc, argv, "--Sz");
    const bool do_SxVx = has_flag(argc, argv, "--SxVx");
    const bool do_SxVy = has_flag(argc, argv, "--SxVy");
    const bool do_SxVz = has_flag(argc, argv, "--SxVz");
    const bool do_SyVx = has_flag(argc, argv, "--SyVx");
    const bool do_SyVy = has_flag(argc, argv, "--SyVy");
    const bool do_SyVz = has_flag(argc, argv, "--SyVz");
    const bool do_SzVx = has_flag(argc, argv, "--SzVx");
    const bool do_SzVy = has_flag(argc, argv, "--SzVy");
    const bool do_SzVz = has_flag(argc, argv, "--SzVz");

    // If nothing requested, default to H only
    const bool any_explicit = do_H||do_Vx||do_Vy||do_Vz||
                              do_Sx||do_Sy||do_Sz||
                              do_SxVx||do_SxVy||do_SxVz||
                              do_SyVx||do_SyVy||do_SyVz||
                              do_SzVx||do_SzVy||do_SzVz;
    const bool write_H = do_H || !any_explicit;

    // Determine what actually needs to be computed
    const bool need_V[3] = {
        do_Vx || do_SxVx || do_SyVx || do_SzVx,
        do_Vy || do_SxVy || do_SyVy || do_SzVy,
        do_Vz || do_SxVz || do_SyVz || do_SzVz
    };
    const bool need_S[3] = {
        do_Sx || do_SxVx || do_SxVy || do_SxVz,
        do_Sy || do_SyVx || do_SyVy || do_SyVz,
        do_Sz || do_SzVx || do_SzVy || do_SzVz
    };
    const bool need_any_V = need_V[0] || need_V[1] || need_V[2];
    const bool need_any_S = need_S[0] || need_S[1] || need_S[2];

    // ---- Read inputs -------------------------------------------------------
    std::cout << "Reading " << hr_file << " ...\n";
    HrData hr = parse_hr(hr_file);
    std::cout << "  num_wann=" << hr.num_wann
              << "  nrpts=" << hr.nrpts
              << "  entries=" << hr.hoppings.size() << "\n";

    std::array<Vec3,3> cell{};
    std::vector<Vec3>  centres;
    if (need_any_V) {
        std::cout << "Reading " << cell_file << " ...\n";
        cell = parse_cell(cell_file);
        std::cout << "Reading " << centres_file << " ...\n";
        centres = parse_centres(centres_file, hr.num_wann);
        std::cout << "  " << centres.size() << " Wannier centres loaded\n";
    }

    const long Ndim = (long)N*M*P*hr.num_wann;
    std::cout << "Supercell " << N << "x" << M << "x" << P
              << "  Ndim=" << Ndim << "\n\n";

    // ---- Build H -----------------------------------------------------------
    SpMat H_mat;
    if (write_H) {
        std::cout << "Building H ... " << std::flush;
        H_mat = build_H(hr, N, M, P);
        std::cout << "nnz=" << H_mat.nonZeros() << "\n";
        write_csr(prefix + "_H.CSR", H_mat);
        std::cout << "  Written: " << prefix << "_H.CSR\n";
    }

    // ---- Build velocity components ----------------------------------------
    SpMat V[3];
    const char* vname[3] = {"Vx","Vy","Vz"};
    for (int a = 0; a < 3; ++a) {
        if (!need_V[a]) continue;
        std::cout << "Building " << vname[a] << " ... " << std::flush;
        V[a] = build_V(hr, cell, centres, N, M, P, a);
        std::cout << "nnz=" << V[a].nonZeros() << "\n";
        if (a==0 && do_Vx) { write_csr(prefix+"_Vx.CSR", V[a]); std::cout << "  Written: " << prefix << "_Vx.CSR\n"; }
        if (a==1 && do_Vy) { write_csr(prefix+"_Vy.CSR", V[a]); std::cout << "  Written: " << prefix << "_Vy.CSR\n"; }
        if (a==2 && do_Vz) { write_csr(prefix+"_Vz.CSR", V[a]); std::cout << "  Written: " << prefix << "_Vz.CSR\n"; }
    }

    // ---- Build spin components --------------------------------------------
    SpMat S[3];
    const char* sname[3] = {"Sx","Sy","Sz"};
    for (int a = 0; a < 3; ++a) {
        if (!need_S[a]) continue;
        std::cout << "Building " << sname[a] << " ... " << std::flush;
        S[a] = build_S(hr.num_wann, N, M, P, a);
        std::cout << "nnz=" << S[a].nonZeros() << "\n";
        if (a==0 && do_Sx) { write_csr(prefix+"_Sx.CSR", S[a]); std::cout << "  Written: " << prefix << "_Sx.CSR\n"; }
        if (a==1 && do_Sy) { write_csr(prefix+"_Sy.CSR", S[a]); std::cout << "  Written: " << prefix << "_Sy.CSR\n"; }
        if (a==2 && do_Sz) { write_csr(prefix+"_Sz.CSR", S[a]); std::cout << "  Written: " << prefix << "_Sz.CSR\n"; }
    }

    // ---- Build spin-current operators -------------------------------------
    // J^α_β = ½ {Sα, Vβ}
    struct SVReq { bool do_it; int sa, vb; const char* name; };
    std::vector<SVReq> sv_reqs = {
        {do_SxVx, 0,0,"SxVx"}, {do_SxVy, 0,1,"SxVy"}, {do_SxVz, 0,2,"SxVz"},
        {do_SyVx, 1,0,"SyVx"}, {do_SyVy, 1,1,"SyVy"}, {do_SyVz, 1,2,"SyVz"},
        {do_SzVx, 2,0,"SzVx"}, {do_SzVy, 2,1,"SzVy"}, {do_SzVz, 2,2,"SzVz"},
    };
    for (const auto& req : sv_reqs) {
        if (!req.do_it) continue;
        std::cout << "Building " << req.name << " = ½{S"
                  << (char)('x'+req.sa) << ",V" << (char)('x'+req.vb)
                  << "} ... " << std::flush;
        SpMat JJ = build_SV(S[req.sa], V[req.vb]);
        std::cout << "nnz=" << JJ.nonZeros() << "\n";
        write_csr(prefix + "_" + req.name + ".CSR", JJ);
        std::cout << "  Written: " << prefix << "_" << req.name << ".CSR\n";
    }

    // ---- Info file --------------------------------------------------------
    {
        std::ofstream info(prefix + "_info.txt");
        info << "supercell: " << N << " x " << M << " x " << P << "\n"
             << "num_wann:  " << hr.num_wann << "\n"
             << "nrpts:     " << hr.nrpts    << "\n"
             << "Ndim:      " << Ndim        << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
