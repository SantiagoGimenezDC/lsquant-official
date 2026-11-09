#include "sparse_matrix.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "linear_algebra.hpp"
#include <random>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <omp.h>

namespace {
// Plain real-arithmetic complex products (avoid __muldc3 in hot loops).
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
// SparseMatrixType_kQuant_nonOrth::ReadPhasesFromFile
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
bool SparseMatrixType_kQuant_nonOrth_ChrisVel::ReadPhasesFromFile(const std::string& filename)
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


void SparseMatrixType_kQuant_nonOrth_ChrisVel::Rescale_nonOrth(const complex<double> a, const complex<double> b)
{
	

        (*this->Matrix()) = a * (*this->Matrix()) + b*(*Sk_);

	set_scaleFactor(real(a));
	set_shiftFactor(real(b));

	return ;
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
void SparseMatrixType_kQuant_nonOrth_ChrisVel::apply_B(value_t* out, const value_t* in) 
{
    int N = numRows();//Nk * W;
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
void SparseMatrixType_kQuant_nonOrth_ChrisVel::apply_B_FFT(value_t* out, const value_t* in)
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
void SparseMatrixType_kQuant_nonOrth_ChrisVel::apply_Bdagger(value_t* out, const value_t* in) 
{
  int N = this->numRows();//Nk * W;
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

void SparseMatrixType_kQuant_nonOrth_ChrisVel::apply_Bdagger_FFT(value_t* out, const value_t* in)
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
void SparseMatrixType_kQuant_nonOrth_ChrisVel::PrepareFFT()
{
    // Full per-k block: W (spinless) or 2W (spinful, layout ik*2W + s*W + α).
    if (this->numRows() % Nk != 0)
        throw std::runtime_error("SparseMatrixType_kQuant_nonOrth_ChrisVel: matrix size is not a multiple of Nk");
    Bs = this->numRows() / Nk;
    if (Bs % W != 0)
        throw std::runtime_error("SparseMatrixType_kQuant_nonOrth_ChrisVel: per-k block size is not a multiple of W");
    if (Bs != W)
        std::cout << "  Per-k block = " << Bs << " = " << Bs / W << " x W"
                  << " (assuming layout ik*Bs + spin*W + alpha)" << std::endl;

    const int N = Nk * Bs;
    fft_buf.assign(N, value_t(0.0, 0.0));

    // Expand the atom phases once: both spin copies of orbital α share its phase.
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

    // In-place plans, stride Bs between elements of one transform and
    // distance 1 between transforms. Built once, so FFTW_MEASURE pays off
    // (it overwrites buf, which is re-zeroed afterwards).
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

    // Later plans (per-thread 1D FFTs of the Kubo solver) must be single-threaded.
    fftw_plan_with_nthreads(1);

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("SparseMatrixType_kQuant_nonOrth_ChrisVel: FFTW plan creation failed");

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
void SparseMatrixType_kQuant_nonOrth_ChrisVel::GenerateAndersonDisorder(double amplitude,
                                                        unsigned int seed)
{
  disorder.resize(this->numRows());//Nk * W);
    std::srand(seed);
    for (int iR = 0; iR < Nk; ++iR) {
        //NO! --> One random value per unit cell, shared across all orbitals
        
      for (int alpha = 0; alpha < W; ++alpha){
	  double v = amplitude * (static_cast<double>(std::rand()) / RAND_MAX - 0.5);
	  disorder[iR * 2 * W + alpha] = value_t(v, 0.0);
	  disorder[iR * 2 * W + W + alpha] = value_t(v, 0.0);
      }
    }
    std::cout << "  Anderson disorder generated: amplitude=" << amplitude
              << ", seed=" << seed << std::endl;
}



void SparseMatrixType_kQuant_nonOrth_ChrisVel::Generate_zMagneticDisorder(
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


// Random-direction magnetic disorder and its application (σ·V per orbital),
// identical to SparseMatrixType_kQuant so both solvers draw the same disorder.
void SparseMatrixType_kQuant_nonOrth_ChrisVel::Generate_RdirMagneticDisorder(
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


void SparseMatrixType_kQuant_nonOrth_ChrisVel::apply_disorder( const value_t* in, value_t* out)
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








void SparseMatrixType_kQuant_nonOrth_ChrisVel::Multiply_kQuant(const value_t  a,
                                        const vector_t& x,
                                              value_t   b,
                                              vector_t& y)
{
    SparseMatrixType_kQuant_nonOrth_ChrisVel::Multiply_kQuant(a, x.data(), b, y.data());
}




void SparseMatrixType_kQuant_nonOrth_ChrisVel::Multiply_kQuant(const value_t  a,
                                               const value_t* x,
                                                     value_t  b,
                                                     value_t* y)
{
    // y = a * ( H_Löwdin x + B† V B x ) + b * y
    const int N = numRows();

    if ((int)hk_buf.size() != N)
        hk_buf.assign(N, value_t(0.0, 0.0));
    value_t* hx = hk_buf.data();

    this->Hk_clean_nonOrth_Loewdin( x, hx );

    const bool magnetic = disorder_x.size() != 0;

    if (disorder.empty() && !magnetic) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i)
            y[i] = kq_cmul(a, hx[i]) + kq_cmul(b, y[i]);
        return;
    }

    if (!plan_fwd || !plan_bwd || Nk * Bs != N)
        throw std::runtime_error("PrepareFFT() must be called before Multiply with disorder.");

    if ((int)dis_real_buf.size() != N) {
        dis_real_buf.assign(N, value_t(0.0, 0.0));
        dis_tmp_buf.assign(N, value_t(0.0, 0.0));
    }
    value_t* real_buf = dis_real_buf.data();
    value_t* tmp      = dis_tmp_buf.data();

    apply_Bdagger_FFT(real_buf, x);              // real_buf = B|x⟩ (real space)

    if (magnetic) {
        if (Bs != 2 * W)
            throw std::runtime_error("Multiply_kQuant: magnetic disorder needs a spinful "
                                     "matrix (per-k block 2W)");
        apply_disorder(real_buf, tmp);           // tmp = (σ·V) B|x⟩
    }
    else
        std::fill(tmp, tmp + N, value_t(0.0, 0.0));

    if (!disorder.empty()) {                     // scalar part: tmp += V ∘ B|x⟩
        const value_t* V = disorder.data();
        if ((int)disorder.size() == N) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i)
                tmp[i] += kq_cmul(V[i], real_buf[i]);
        }
        else if ((int)disorder.size() == Nk * W) {
            #pragma omp parallel for schedule(static)
            for (int iR = 0; iR < Nk; ++iR)
                for (int j = 0; j < Bs; ++j)
                    tmp[iR * Bs + j] += kq_cmul(V[iR * W + j % W], real_buf[iR * Bs + j]);
        }
        else
            throw std::runtime_error("Multiply_kQuant: disorder has size " +
                                     std::to_string(disorder.size()) + ", expected " +
                                     std::to_string(N) + " or " + std::to_string(Nk * W));
    }

    apply_B_FFT(real_buf, tmp);                  // real_buf = B† V B|x⟩ (k space)

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        y[i] = kq_cmul(a, hx[i] + real_buf[i]) + kq_cmul(b, y[i]);
}






void SparseMatrixType_kQuant_nonOrth_ChrisVel::Hk_clean_nonOrth(const value_t * in, value_t * out) {

   Eigen::Map<const Eigen::Vector<std::complex<double>, -1>> eig_x(in, numRows());
   Eigen::Vector<std::complex<double>, -1> tmp(numRows());



   tmp =  (*Hk_) * eig_x;  



   linalg::orthogonalize(size_t(numRows()), Sk_, M_, tmp.data(), out);

   

};



void SparseMatrixType_kQuant_nonOrth_ChrisVel::vel_i_nonOrth_oneSided(const value_t * in, value_t * out, int dir) {

  Eigen::Map<const Eigen::Vector<std::complex<double>, -1>> eig_x(in, numRows());
  //Eigen::Map< Eigen::Vector<std::complex<double>, -1>>  out_eig(out, numRows());
  Eigen::Vector<std::complex<double>, -1> tmp_1( numRows()), tmp_2( numRows());


  double a =  this->scaleFactor(), b = this->shiftFactor();

  Eigen::SparseMatrix<std::complex<double>,  Eigen::RowMajor, indexType> Id(numRows(),numRows());


  if( dir == 1 )
    tmp_2 =  (*dHk_1_) * eig_x;
    
  else if( dir == 2 )
    tmp_2 =  (*dHk_2_) * eig_x;


  
  //if( dir == 1 )
  //  tmp_2 =  (*Sz_) * tmp_2;
    
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_, tmp_2.data(), out);
    




  
  
  
  if( dir == 1 )
    tmp_2 =  (*A_1_) * eig_x;
    
  
  else if( dir == 2 )
    tmp_2 =  (*A_2_) * eig_x;

  //tmp_2 /= a; //Why?? Is it because the velocity is not adimensionalized? so this is supposed to return de dimension to HK


  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_, tmp_2.data(), tmp_1.data());

  tmp_2  =   *Hk_* tmp_1;
  tmp_2 -= b * (*Sk_)* tmp_1  ;
  tmp_2 /= a;
  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_, tmp_2.data(), tmp_1.data());  


  
  //if( dir == 1 )
  //  tmp_1 =  (*Sz_) * tmp_1;
  
  linalg::axpy(numRows(), std::complex<double>(0,1.0), tmp_1.data(), out);

  


  
  tmp_2  =  *Hk_* eig_x;
  tmp_2 -= b * (*Sk_)  * eig_x;

  tmp_2 /= a; //Why?? Is it because the velocity is not adimensionalized? so this is supposed to return de dimension to HK

  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_, tmp_2.data(), tmp_1.data());


  if( dir == 1 )
    tmp_2 =  (*A_1d_) * tmp_1;
  else if( dir == 2 )
    tmp_2 =  (*A_2d_) * tmp_1;


  
  //if( dir == 1 )
  //  tmp_1 =  (*Sz_) * tmp_1;

  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_, tmp_2.data(), tmp_1.data());  


  linalg::axpy(numRows(), std::complex<double>(0,-1.0), tmp_1.data(), out);
  

  
};

void SparseMatrixType_kQuant_nonOrth_ChrisVel::sz(const Eigen::Vector<std::complex<double>, -1>& in, Eigen::Vector<std::complex<double>, -1>& out)
{
    const int block_size = 2 * norbs_;  

    out=in;

    #pragma omp parallel for schedule(static)
    for (int ik = 0; ik < nk_; ++ik)
        out.segment(ik * block_size + norbs_, norbs_) *= -1.0;
 
}













void SparseMatrixType_kQuant_nonOrth_ChrisVel::Hk_clean_nonOrth_Loewdin(const value_t * in, value_t * out) {

   Eigen::Map<const Eigen::Vector<std::complex<double>, -1>> eig_x(in, numRows());
   Eigen::Vector<std::complex<double>, -1> tmp(numRows());


   linalg::orthogonalize(size_t(numRows()), Sk_, M_Loew_, eig_x.data(), tmp.data());

   tmp =  (*Hk_) * tmp;  

   linalg::orthogonalize(size_t(numRows()), Sk_, M_Loew_, tmp.data(), out);

   

};



void SparseMatrixType_kQuant_nonOrth_ChrisVel::vel_i_nonOrth_Loewdin(const value_t * in, value_t * out, int dir) {

  Eigen::Map<const Eigen::Vector<std::complex<double>, -1>> eig_x(in, numRows());
  Eigen::Map< Eigen::Vector<std::complex<double>, -1>>  out_eig(out, numRows());
  Eigen::Vector<std::complex<double>, -1> tmp_1( numRows()), tmp_2( numRows());


  if( dir==3 ){
   
    //sz(eig_x, tmp_1);

    tmp_1 = (*Sz_) * eig_x;
    vel_i_nonOrth_Loewdin(tmp_1.data(), out_eig.data(), 2);

    
    vel_i_nonOrth_Loewdin(eig_x.data(), tmp_1.data(), 2);
    //sz(tmp_1, tmp_2);

    tmp_2 = (*Sz_) * tmp_1;
    

    out_eig = 0.25 * (out_eig + tmp_2);

    return;
  }



  double a =  this->scaleFactor(), b = this->shiftFactor();

  Eigen::SparseMatrix<std::complex<double>,  Eigen::RowMajor, indexType> Id(numRows(),numRows());



  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, eig_x.data(), tmp_2.data());

  if( dir == 1 )
    tmp_2 =  (*dHk_1_) * tmp_2;
    
  else if( dir == 2 )
    tmp_2 =  (*dHk_2_) * tmp_2;
  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_2.data(), out);
    




  

  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, eig_x.data(), tmp_2.data());
  
  if( dir == 1 )
    tmp_2 =  (*A_1_) * tmp_2;
    
  
  else if( dir == 2 )
    tmp_2 =  (*A_2_) * tmp_2;
  
  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_2.data(), tmp_1.data());
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_1.data(), tmp_2.data());

  tmp_1  =   *Hk_* tmp_2;
  tmp_1 -= b * (*Sk_)* tmp_2  ;
  tmp_1 /= a;
  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_1.data(), tmp_2.data());  


  linalg::axpy(numRows(), std::complex<double>(0,1.0), tmp_2.data(), out);

  






  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, eig_x.data(), tmp_1.data());    
  
  tmp_2  =  *Hk_* tmp_1;
  tmp_2 -= b * (*Sk_)  * tmp_1;

  tmp_2 /= a;
  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_2.data(), tmp_1.data());
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_1.data(), tmp_2.data());

  
  if( dir == 1 )
    tmp_1 =  (*A_1d_) * tmp_2;
  else if( dir == 2 )
    tmp_1 =  (*A_2d_) * tmp_2;



  
  linalg::orthogonalize(size_t(numRows()), Sk_,  M_Loew_, tmp_1.data(), tmp_2.data());  


  linalg::axpy(numRows(), std::complex<double>(0,-1.0), tmp_2.data(), out);
  

  
};
