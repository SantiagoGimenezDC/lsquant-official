#include "sparse_matrix.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>

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
    const int N = Nk * W;
    for (int i = 0; i < N; ++i)
        fft_buf[i] = in[i];

    fftw_execute(plan_bwd);   // computes exact lattice sum, no normalization

    for (int i = 0; i < N; ++i)
        out[i] = atom_phases[i] * fft_buf[i];   // atom_phases already has 1/√Nk
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
    const int N = Nk * W;

    // premultiply by conj(atom_phases)  (first part of B†)
    for (int i = 0; i < N; ++i)
        fft_buf[i] = std::conj(atom_phases[i]) * in[i];

    // FORWARD FFT: exp(-i·2π·k·n/N)  →  completes B†, no normalisation needed
    fftw_execute(plan_fwd);

    for (int i = 0; i < N; ++i)
        out[i] = fft_buf[i];
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
    const int N = Nk * W;
    // Allocate FFTW-aligned buffer so SIMD paths are used
    fft_buf.resize(N);

    int n[3]       = {kx, ky, kz};
    fftw_complex* buf = reinterpret_cast<fftw_complex*>(fft_buf.data());

    // Both plans operate in-place on buf with stride=W between elements of
    // the same transform and distance=1 between consecutive transforms.
    plan_fwd = fftw_plan_many_dft(
        3, n, W,
        buf, nullptr, W, 1,
        buf, nullptr, W, 1,
        FFTW_FORWARD,  FFTW_ESTIMATE);

    plan_bwd = fftw_plan_many_dft(
        3, n, W,
        buf, nullptr, W, 1,
        buf, nullptr, W, 1,
        FFTW_BACKWARD, FFTW_ESTIMATE);

    if (!plan_fwd || !plan_bwd)
        throw std::runtime_error("SparseMatrixType_kQuant: FFTW plan creation failed");

    std::cout << "  FFTW plans ready  (3D batch, howmany=" << W
              << ", grid=" << kx << "×" << ky << "×" << kz << ")" << std::endl;
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




    // Step 4: FFTW_BACKWARD  →  unnormalised real→k transform
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









void SparseMatrixType_kQuant::Multiply_kQuant(const value_t  a,
                                               const value_t* x,
                                                     value_t  b,
                                                     value_t* y)
{
    const int N = Nk * W;


    SparseMatrixType::Multiply(a, x, b, y);

    if (disorder.empty()) return;

    // Use two separate buffers — fft_buf is used internally by both FFT functions
    std::vector<value_t> real_buf(N);   // holds B|x⟩ in real space
    std::vector<value_t> k_buf(N);      // holds B†(V·B|x⟩) in k space

    apply_Bdagger_FFT(real_buf.data(), x);    // real_buf = B|x⟩

    for (int i = 0; i < N; ++i)
      real_buf[i] *= disorder[i];     // real_buf = V·B|x⟩  (or -0.1 for test)
      
    
    apply_B_FFT(k_buf.data(), real_buf.data());  // k_buf = B†V·B|x⟩

    
    for (int i = 0; i < N; ++i)
        y[i] += a * k_buf[i];

    
}


void SparseMatrixType_kQuant::Multiply_kQuant(const value_t  a,
                                        const vector_t& x,
                                              value_t   b,
                                              vector_t& y)
{
    SparseMatrixType_kQuant::Multiply_kQuant(a, x.data(), b, y.data());
}
