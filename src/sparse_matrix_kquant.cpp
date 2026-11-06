#include "sparse_matrix.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>

namespace {
// Plain real-arithmetic complex products. Without -ffast-math /
// -fcx-limited-range, GCC compiles std::complex * std::complex into a
// call to __muldc3 (NaN/Inf checks), which is slow in hot loops.
inline std::complex<double> kq_cmul(const std::complex<double> a, const std::complex<double> b)
{
    return { a.real() * b.real() - a.imag() * b.imag(),
             a.real() * b.imag() + a.imag() * b.real() };
}
// conj(a) * b
inline std::complex<double> kq_cmul_conj(const std::complex<double> a, const std::complex<double> b)
{
    return { a.real() * b.real() + a.imag() * b.imag(),
             a.real() * b.imag() - a.imag() * b.real() };
}
}

// ─────────────────────────────────────────────────────────────────────────────
// SparseMatrixType_kQuant::ReadPhasesFromFile
//
// Reads the binary .BLOCH_PHASES file written by siesta2linqt.py.
//
// File layout (little-endian):
//   ASCII line 1 : "BLOCH_PHASES\n"
//   ASCII line 2 : "{Nk} {W} {kx} {ky} {kz}\n"
//   int32  block  : n_grid[Nk*3]          – grid integer indices
//   cdouble block : atom_phases[Nk*W]     – exp(i k·τ_α)/√Nk
//   cdouble block : phi_x[kx*kx]          – exp(i·2π·n·m/kx)
//   cdouble block : phi_y[ky*ky]
//   cdouble block : phi_z[kz*kz]
// ─────────────────────────────────────────────────────────────────────────────
bool SparseMatrixType_kQuant::ReadPhasesFromFile(const std::string& filename)
{
    std::cout << "\nReading Bloch phase file: " << filename << std::endl;

    FILE* f = std::fopen(filename.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        return false;
    }

    // ── ASCII header ──────────────────────────────────────────────────────────
    char line[256];
    if (!std::fgets(line, sizeof(line), f) ||
        std::strncmp(line, "BLOCH_PHASES", 12) != 0) {
        std::cerr << "ERROR: Not a BLOCH_PHASES file." << std::endl;
        std::fclose(f);
        return false;
    }
    if (!std::fgets(line, sizeof(line), f)) {
        std::cerr << "ERROR: Truncated header." << std::endl;
        std::fclose(f);
        return false;
    }
    if (std::sscanf(line, "%d %d %d %d %d", &Nk, &W, &kx, &ky, &kz) != 5) {
        std::cerr << "ERROR: Bad header dimensions." << std::endl;
        std::fclose(f);
        return false;
    }
    std::cout << "  Nk=" << Nk << "  W=" << W
              << "  grid=" << kx << "×" << ky << "×" << kz << std::endl;

    // ── n_grid [Nk × 3] int32 ────────────────────────────────────────────────
    std::vector<int32_t> tmp_grid(Nk * 3);
    if (std::fread(tmp_grid.data(), sizeof(int32_t), Nk*3, f) != (size_t)(Nk*3)) {
        std::cerr << "ERROR: Failed reading n_grid." << std::endl;
        std::fclose(f); return false;
    }
    n_grid.resize(Nk);
    for (int i = 0; i < Nk; ++i) {
        n_grid[i][0] = tmp_grid[i*3 + 0];
        n_grid[i][1] = tmp_grid[i*3 + 1];
        n_grid[i][2] = tmp_grid[i*3 + 2];
    }

    // ── atom_phases [Nk × W] complex128 ──────────────────────────────────────
    atom_phases.resize(Nk * W);
    if (std::fread(atom_phases.data(), sizeof(value_t), Nk*W, f) != (size_t)(Nk*W)) {
        std::cerr << "ERROR: Failed reading atom_phases." << std::endl;
        std::fclose(f); return false;
    }

    // ── phi_x [kx × kx], phi_y [ky × ky], phi_z [kz × kz] ──────────────────
    phi_x.resize(kx * kx);
    if (std::fread(phi_x.data(), sizeof(value_t), kx*kx, f) != (size_t)(kx*kx)) {
        std::cerr << "ERROR: Failed reading phi_x." << std::endl;
        std::fclose(f); return false;
    }
    phi_y.resize(ky * ky);
    if (std::fread(phi_y.data(), sizeof(value_t), ky*ky, f) != (size_t)(ky*ky)) {
        std::cerr << "ERROR: Failed reading phi_y." << std::endl;
        std::fclose(f); return false;
    }
    phi_z.resize(kz * kz);
    if (std::fread(phi_z.data(), sizeof(value_t), kz*kz, f) != (size_t)(kz*kz)) {
        std::cerr << "ERROR: Failed reading phi_z." << std::endl;
        std::fclose(f); return false;
    }

    std::fclose(f);
    std::cout << "  Bloch phases loaded successfully." << std::endl;


for (int ik = 0; ik < Nk; ++ik) {
    int expected = n_grid[ik][0]*ky*kz + n_grid[ik][1]*kz + n_grid[ik][2];
    if (expected != ik) {
        std::cerr << "k-point ordering mismatch at ik=" << ik
                  << "  expected flat=" << expected << std::endl;
        break;
    }
}


    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// apply_B :  real-space → k-space
//
//   out[ik·W + α] = atom_phase[ik][α] × Σ_{iR} lat_phase(ik,iR) × in[iR·W+α]
//
// Inner loop order: iterate over α (orbitals) in the outer loop so that
// the iR summation accesses in[] with stride W — kept in cache by the
// compiler's prefetcher.  OpenMP parallelises over ik.
// ─────────────────────────────────────────────────────────────────────────────
void SparseMatrixType_kQuant::apply_B(value_t* out, const value_t* in) const
{
    const int N = Nk * W;
    std::fill(out, out + N, value_t(0.0, 0.0));

    #pragma omp parallel for schedule(dynamic)
    for (int ik = 0; ik < Nk; ++ik) {
        for (int alpha = 0; alpha < W; ++alpha) {
            value_t acc(0.0, 0.0);
            for (int iR = 0; iR < Nk; ++iR) {
                acc += lattice_phase(ik, iR) * in[iR*W + alpha];
            }
            out[ik*W + alpha] = atom_phases[ik*W + alpha] * acc;
        }
    }
}
/*
void SparseMatrixType_kQuant::apply_B_FFT( value_t* out, const value_t* in)
{
    const int N = Nk * W;

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("PrepareFFT() must be called before Multiply.");

    // Step 1: premultiply by conj(atom_phases)  →  ready for B†
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        fft_buf[i] = std::conj(atom_phases[i]) * in[i];

    // Step 2: FFTW_FORWARD  →  B†x in real space  (indexed by iR)
    fftw_execute(plan_fwd);



    // Step 5: postmultiply by atom_phases + normalise + accumulate into y
    const value_t scale = 1.0 / sqrt( static_cast<double>(Nk) );
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        out[i] += scale * atom_phases[i] * fft_buf[i];
	}*/
void SparseMatrixType_kQuant::apply_B_FFT(value_t* out, const value_t* in)
{
    const int N = Nk * Bs;
    value_t* buf = fft_buf.data();
    const value_t* ph = atom_phases_full.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        buf[i] = in[i];

    fftw_execute(plan_bwd);   // computes exact lattice sum, no normalization

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        out[i] = kq_cmul(ph[i], buf[i]);   // atom_phases already has 1/√Nk
}



// ─────────────────────────────────────────────────────────────────────────────
// apply_Bdagger :  k-space → real-space
//
//   out[iR·W + α] = Σ_{ik} conj(lat_phase(ik,iR))
//                        × conj(atom_phase[ik][α])
//                        × in[ik·W + α]
// ─────────────────────────────────────────────────────────────────────────────
void SparseMatrixType_kQuant::apply_Bdagger(value_t* out, const value_t* in) const
{
    const int N = Nk * W;
    std::fill(out, out + N, value_t(0.0, 0.0));

    #pragma omp parallel for schedule(dynamic)
    for (int iR = 0; iR < Nk; ++iR) {
        for (int alpha = 0; alpha < W; ++alpha) {
            value_t acc(0.0, 0.0);
            for (int ik = 0; ik < Nk; ++ik) {
                acc += std::conj(lattice_phase(ik, iR))
                     * std::conj(atom_phases[ik*W + alpha])
                     * in[ik*W + alpha];
            }
            out[iR*W + alpha] = acc;
        }
    }
}

void SparseMatrixType_kQuant::apply_Bdagger_FFT(value_t* out, const value_t* in)
{
    const int N = Nk * Bs;
    value_t* buf = fft_buf.data();
    const value_t* ph = atom_phases_full.data();

    // premultiply by conj(atom_phases)  (first part of B†)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        buf[i] = kq_cmul_conj(ph[i], in[i]);

    // FORWARD FFT: exp(-i·2π·k·n/N)  →  completes B†, no normalisation needed
    fftw_execute(plan_fwd);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        out[i] = buf[i];
}

/*
void SparseMatrixType_kQuant::apply_Bdagger_FFT( value_t* out, const value_t* in)
{
    const int N = Nk * W;

    // Step 1: premultiply by conj(atom_phases)  →  ready for B†
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        fft_buf[i] =  in[i];

    // Step 4: FFTW_BACKWARD  →  unnormalised real→k transform
    fftw_execute(plan_bwd);


    // Step 5: postmultiply by atom_phases + normalise + accumulate into y
    const value_t scale = 1.0 / sqrt( static_cast<double>(Nk) );
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        out[i] = scale * atom_phases[i] * fft_buf[i];
}
*/



// ─────────────────────────────────────────────────────────────────────────────
// PrepareFFT
//
// Creates two in-place FFTW plans for a batch of W transforms of size
// kx×ky×kz each, with the data interleaved as [ik*W + α] (stride = W).
//
// B†  uses FFTW_FORWARD  (exp(-i·2π·k·n/N)):  k-space → real-space
// B   uses FFTW_BACKWARD (exp(+i·2π·k·n/N)):  real-space → k-space
// ─────────────────────────────────────────────────────────────────────────────
void SparseMatrixType_kQuant::PrepareFFT()
{
    // Full per-k block: W (spinless) or 2W (spinful, layout ik*2W + s*W + α).
    if (this->rank() % Nk != 0)
        throw std::runtime_error("SparseMatrixType_kQuant: rank() is not a multiple of Nk");
    Bs = this->rank() / Nk;
    if (Bs % W != 0)
        throw std::runtime_error("SparseMatrixType_kQuant: per-k block size is not a multiple of W");
    if (Bs != W)
        std::cout << "  Per-k block = " << Bs << " = " << Bs / W << " x W"
                  << " (assuming layout ik*Bs + spin*W + alpha)" << std::endl;

    const int N = Nk * Bs;
    fft_buf.assign(N, value_t(0.0, 0.0));

    // Expand the atom phases once so the hot loops index them linearly.
    atom_phases_full.resize(N);
    for (int ik = 0; ik < Nk; ++ik)
        for (int j = 0; j < Bs; ++j)
            atom_phases_full[ik * Bs + j] = atom_phases[ik * W + j % W];

    // Multi-threaded FFTW for the 3D batch transforms (we link -lfftw3_omp).
    static bool fftw_threads_ready = false;
    if (!fftw_threads_ready) {
        fftw_init_threads();
        fftw_threads_ready = true;
    }
    fftw_plan_with_nthreads(omp_get_max_threads());

    int n[3]       = {kx, ky, kz};
    fftw_complex* buf = reinterpret_cast<fftw_complex*>(fft_buf.data());

    // Both plans operate in-place on buf with stride=Bs between elements of
    // the same transform and distance=1 between consecutive transforms.
    // Plans are built once, so FFTW_MEASURE pays off (it overwrites buf,
    // which is why buf is (re)zeroed afterwards).
    plan_fwd = fftw_plan_many_dft(
        3, n, Bs,
        buf, nullptr, Bs, 1,
        buf, nullptr, Bs, 1,
        FFTW_FORWARD,  FFTW_MEASURE);

    plan_bwd = fftw_plan_many_dft(
        3, n, Bs,
        buf, nullptr, Bs, 1,
        buf, nullptr, Bs, 1,
        FFTW_BACKWARD, FFTW_MEASURE);

    // Later plans (e.g. the per-thread 1D FFTs of the Kubo solver) must be
    // single-threaded: they are executed from inside OpenMP regions.
    fftw_plan_with_nthreads(1);

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("SparseMatrixType_kQuant: FFTW plan creation failed");

    std::fill(fft_buf.begin(), fft_buf.end(), value_t(0.0, 0.0));

    std::cout << "  FFTW plans ready  (3D batch, howmany=" << Bs
              << ", grid=" << kx << "×" << ky << "×" << kz
              << ", threads=" << omp_get_max_threads() << ")" << std::endl;
}


// ─────────────────────────────────────────────────────────────────────────────
// GenerateAndersonDisorder
//
// Fills disorder[iR*W + α] with a uniform random value in
// [-amplitude/2, +amplitude/2].  The same random draw is used for all W
// orbitals within a given unit cell (standard on-site Anderson model).
// ─────────────────────────────────────────────────────────────────────────────
void SparseMatrixType_kQuant::GenerateAndersonDisorder(double amplitude,
                                                        unsigned int seed)
{
    disorder.resize(Nk * W);
    std::srand(seed);
    for (int iR = 0; iR < Nk; ++iR) {
        //NO! --> One random value per unit cell, shared across all orbitals
        
      for (int alpha = 0; alpha < W; ++alpha){
	  double v = amplitude * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
	  disorder[iR * W + alpha] = value_t(v, 0.0);
      }
    }
    std::cout << "  Anderson disorder generated: amplitude=" << amplitude
              << ", seed=" << seed << std::endl;
}



void SparseMatrixType_kQuant::Generate_zMagneticDisorder(
    double strength,
    double concentration,
    unsigned int seed)
{
    disorder.resize(this->numRows());

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int iR = 0; iR < Nk; ++iR) {

        // One impurity decision/value per orbital
        for (int alpha = 0; alpha < W; ++alpha) {

            double v = 0.0;

            if (dist(gen) < concentration) {
                v = strength;
            }

            // Spin-up
            disorder[iR * 2 * W + alpha] = value_t(+v, 0.0);

            // Spin-down
            disorder[iR * 2 * W + W + alpha] = value_t(-v, 0.0);
        }
    }

    std::cout << "  Magnetic disorder generated:"
              << " strength=" << strength
              << ", concentration=" << concentration
              << ", seed=" << seed
              << std::endl;
}


void SparseMatrixType_kQuant::Generate_RdirMagneticDisorder(
    double strength,
    double concentration,
    unsigned int seed,
    const std::string& orbital_mapping_file)
{
    disorder_x.resize(this->numRows());
    disorder_y.resize(this->numRows());
    disorder_z.resize(this->numRows());

    disorder_x.setZero();
    disorder_y.setZero();
    disorder_z.setZero();

    // ------------------------------------------------------------
    // Read orbital mapping for ONE unit cell
    // ------------------------------------------------------------

    struct AtomOrbitals {
        int atom_index;
        std::string species;
        std::vector<int> orbitals;
    };

    std::vector<AtomOrbitals> atoms;

    std::ifstream file(orbital_mapping_file);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open orbital mapping file: " +
            orbital_mapping_file);
    }

    std::string line;

    while (std::getline(file, line)) {

        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);

        AtomOrbitals atom;
        int norb;

        ss >> atom.atom_index;
        ss >> atom.species;
        ss >> norb;

        atom.orbitals.resize(norb);

        for (int i = 0; i < norb; ++i)
            ss >> atom.orbitals[i];

        atoms.push_back(atom);
    }

    file.close();

    // ------------------------------------------------------------
    // Check that mapping agrees with W
    // ------------------------------------------------------------

    int total_orbitals = 0;

    for (const auto& atom : atoms)
        for (int orbital : atom.orbitals)
            total_orbitals = std::max(total_orbitals, orbital + 1);

    if (total_orbitals != W) {
        throw std::runtime_error(
            "Orbital mapping is inconsistent with W: " +
            std::to_string(total_orbitals) +
            " != " +
            std::to_string(W));
    }

    // ------------------------------------------------------------
    // Random-number generators
    // ------------------------------------------------------------

    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> phi_dist(
        0.0, 2.0 * M_PI);

    int n_impurities = 0;

    // ------------------------------------------------------------
    // Generate disorder over the ENTIRE real-space supercell
    // ------------------------------------------------------------

    for (int iR = 0; iR < Nk; ++iR) {

        for (const auto& atom : atoms) {

            // Is this particular atom in this particular unit cell
            // magnetic?
            if (dist(gen) >= concentration)
                continue;

            ++n_impurities;

	    
            // ----------------------------------------------------
            // One random spin direction for this atom
            // ----------------------------------------------------

            double cos_theta = 2.0 * dist(gen) - 1.0;
            double sin_theta =
	    std::sqrt(1.0 - cos_theta * cos_theta);

            double phi = phi_dist(gen);

            double nx = sin_theta * std::cos(phi);
            double ny = sin_theta * std::sin(phi);
            double nz = cos_theta;

	    



	    /*
	    // ----------------------------------------------------
	    // One random spin direction for this atom, biased toward +z.
	    // Sampled from a von Mises-Fisher distribution on the sphere:
	    // kappa = 0 recovers your original isotropic sampling; larger
	    // kappa concentrates directions more tightly around +z, i.e.
	    // increasingly resembles a coherent exchange field rather than
	    // pure random-direction disorder.
	    // ----------------------------------------------------
	    const double kappa = 3.0; // tune: 0 = isotropic, larger = stronger +z bias

	    double xi = dist(gen); // uniform in [0,1)
	    double cos_theta;
	    if (kappa > 1e-8) {
	      cos_theta = 1.0 + (1.0 / kappa) *
                std::log(xi + (1.0 - xi) * std::exp(-2.0 * kappa));
	    } else {
	      cos_theta = 2.0 * xi - 1.0; // isotropic fallback, same as before
	    }

	    double sin_theta =
	      std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta)); // guard tiny FP negatives
	    double phi = phi_dist(gen);
	    double nx = sin_theta * std::cos(phi);
	    double ny = sin_theta * std::sin(phi);
	    double nz = cos_theta;
	    */

 
            // ----------------------------------------------------
            // Apply SAME moment to every orbital on this atom
            // ----------------------------------------------------

            for (int alpha : atom.orbitals) {

                int idx = iR * W + alpha;

                disorder_x[idx] =
                    value_t(strength * nx, 0.0);

                disorder_y[idx] =
                    value_t(strength * ny, 0.0);

                disorder_z[idx] =
                    value_t(strength * nz, 0.0);
            }
        }
    }

    std::cout
        << "  Random magnetic disorder generated:"
        << " strength=" << strength
        << " eV, concentration=" << concentration
        << ", impurities=" << n_impurities
        << "/" << Nk * atoms.size()
        << " atoms, seed=" << seed
        << std::endl;
}


void SparseMatrixType_kQuant::apply_disorder( const value_t* in, value_t* out)
{
    const value_t I(0.0, 1.0);

    #pragma omp parallel for schedule(static)
    for (int iR = 0; iR < Nk; ++iR) {

        const int spatial_offset = iR * W;
        const int spin_offset    = iR * 2 * W;

	  
        for (int alpha = 0; alpha < W; ++alpha) {


            const int spatial = spatial_offset + alpha;

            const int up   = spin_offset + alpha;
            const int down = spin_offset + W + alpha;

            const value_t Vx = disorder_x[spatial];
            const value_t Vy = disorder_y[spatial];
            const value_t Vz = disorder_z[spatial];

            // H_mag = Vx*sigma_x + Vy*sigma_y + Vz*sigma_z

	    
            out[up] =
                Vz * in[up]
                + (Vx - I * Vy) * in[down];

            out[down] =
                (Vx + I * Vy) * in[up]
                - Vz * in[down];
        }
    }

}









void SparseMatrixType_kQuant::Multiply_kQuant(const value_t  a,
                                               const value_t* x,
                                                     value_t  b,
                                                     value_t* y)
{
    SparseMatrixType::Multiply(a, x, b, y);

    if (disorder.empty()) {
        // The magnetic-disorder path (apply_disorder) is not wired in here yet;
        // fail loudly instead of silently dropping it.
        if (disorder_x.size() != 0)
            throw std::runtime_error("Multiply_kQuant: magnetic disorder set but only "
                                     "scalar `disorder` is applied; wire apply_disorder first.");
        return;
    }

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("PrepareFFT() must be called before Multiply.");

    // Disorder term  y += a · B† V B x, fused into three parallel passes over
    // fft_buf (no temporaries):
    //   1. fft_buf = conj(atom) · x          then FORWARD FFT  → real space
    //   2. fft_buf *= V                      then BACKWARD FFT → k space
    //   3. y += a · atom · fft_buf
    // atom_phases carry 1/√Nk each, so the unnormalised FFT pair is exact.
    const int N = Nk * Bs;
    value_t* buf = fft_buf.data();
    const value_t* ph = atom_phases_full.data();
    const value_t* V = disorder.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        buf[i] = kq_cmul_conj(ph[i], x[i]);

    fftw_execute(plan_fwd);

    if ((int)disorder.size() == N) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            buf[i] = kq_cmul(buf[i], V[i]);
    }
    else if ((int)disorder.size() == Nk * W) {
        // One potential per (cell, orbital), shared by all spin blocks.
        #pragma omp parallel for schedule(static)
        for (int iR = 0; iR < Nk; ++iR)
            for (int j = 0; j < Bs; ++j)
                buf[iR * Bs + j] = kq_cmul(buf[iR * Bs + j], V[iR * W + j % W]);
    }
    else
        throw std::runtime_error("Multiply_kQuant: disorder has size " +
                                 std::to_string(disorder.size()) + ", expected " +
                                 std::to_string(N) + " or " + std::to_string(Nk * W));

    fftw_execute(plan_bwd);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        y[i] += kq_cmul(a, kq_cmul(ph[i], buf[i]));
}


/*
void SparseMatrixType_kQuant::Multiply_kQuant(const value_t  a,
                                               const value_t* x,
                                                     value_t  b,
                                                     value_t* y)
{
    const int N = Nk * W;


    SparseMatrixType::Multiply(a, x, b, y);

    if (disorder_x.size()==0 && disorder.empty()) return;

    // Use two separate buffers — fft_buf is used internally by both FFT functions
    std::vector<value_t> real_buf(N), tmp(N);   // holds B|x⟩ in real space
    std::vector<value_t> k_buf(N);      // holds B†(V·B|x⟩) in k space

    apply_Bdagger_FFT(real_buf.data(), x);    // real_buf = B|x⟩

    
    apply_disorder(real_buf.data(),tmp.data());
    
    //for (int i = 0; i < N; ++i)
       //real_buf[i] *= disorder[i];     // real_buf = V·B|x⟩  (or -0.1 for test)
      
    
    apply_B_FFT(k_buf.data(), tmp.data());  // k_buf = B†V·B|x⟩

    
    for (int i = 0; i < N; ++i)
        y[i] += a * k_buf[i];

    
}
*/





void SparseMatrixType_kQuant::Multiply_kQuant(const value_t  a,
                                        const vector_t& x,
                                              value_t   b,
                                              vector_t& y)
{


  SparseMatrixType_kQuant::Multiply_kQuant(a, x.data(), b, y.data());

}








// ─────────────────────────────────────────────────────────────────────────────
// Multiply  (overrides SparseMatrixType::Multiply)
//
// Computes:  y = a * (H_k + B† V B) * x + b * y
//
// Step-by-step for the disorder term B† V B x:
//
//  1. fft_buf[i] = conj(atom_phases[i]) * x[i]         (premultiply for B†)
//  2. fftw_FORWARD on fft_buf                           (completes B†: k→real)
//  3. fft_buf[i] *= disorder[i]                         (apply V in real space)
//  4. fftw_BACKWARD on fft_buf                          (first part of B: real→k)
//  5. y[i] += (a/Nk) * atom_phases[i] * fft_buf[i]     (completes B + accumulate)
//
// The 1/Nk in step 5 normalises the unnormalised FFTW backward transform.
// ─────────────────────────────────────────────────────────────────────────────
void SparseMatrixType_kQuant::Multiply_kQuant_bak(const value_t  a,
                                        const value_t* x,
                                              value_t  b,
                                              value_t* y)
{
    const int N = Nk * W;

    // ── 1. H_k contribution (inherited sparse multiply) ──────────────────────
    SparseMatrixType::Multiply(a, x, b, y);

    // ── 2. Disorder contribution ──────────────────────────────────────────────
    if (disorder.empty()) return;   // no disorder → done

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("PrepareFFT() must be called before Multiply.");

    // Step 1: premultiply by conj(atom_phases)  →  ready for B†
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        fft_buf[i] = std::conj(atom_phases[i]) * x[i];

    // Step 2: FFTW_FORWARD  →  B†x in real space  (indexed by iR)
    fftw_execute(plan_fwd);

    // Step 3: apply Anderson disorder diagonally in real space
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        fft_buf[i] *= disorder[i] ;




    // Step 4: FFTW_BACKWARD  →  unnormalised real→ k transform
    fftw_execute(plan_bwd);







    // Step 5: postmultiply by atom_phases + normalise + accumulate into y
    const value_t scale = a / static_cast<double>(Nk);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        y[i] += scale * atom_phases[i] * fft_buf[i];
}

void SparseMatrixType_kQuant::Multiply_kQuant_bak(const value_t  a,
                                        const vector_t& x,
                                              value_t   b,
                                              vector_t& y)
{
    SparseMatrixType_kQuant::Multiply_kQuant(a, x.data(), b, y.data());
}
