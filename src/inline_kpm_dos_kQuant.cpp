// C & C++ libraries
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <chrono>

#include "kpm_noneqop.hpp"
#include "chebyshev_moments.hpp"
#include "sparse_matrix.hpp"
#include "quantum_states.hpp"
#include "chebyshev_solver.hpp"

namespace spectral
{
    void printHelpMessage();
    void printWelcomeMessage();
};

void postProcess(chebyshev::Moments1D_kQuant&);


int main(int argc, char *argv[])
{
    // Usage: Label Op numMom [disorder_amplitude [state_file]]
    if (argc < 4 || argc > 6)
    {
        spectral::printHelpMessage();
        return 0;
    }
    else
        spectral::printWelcomeMessage();

    const std::string
        LABEL  = argv[1],
        S_OP   = argv[2],
        S_NMOM = argv[3];

    const int    numMoms         = atoi(S_NMOM.c_str());
    const double disorder_amplitude = (argc >= 5) ? std::stod(argv[4]) : 0.0;
    const bool   has_state_file  = (argc == 6);

    chebyshev::Moments1D_kQuant chebMoms(numMoms);

    // ── Operators ─────────────────────────────────────────────────────────────
    // OP[0] = Hamiltonian   → SparseMatrixType_kQuant (supports disorder + FFT)
    // OP[1] = spectral op   → plain SparseMatrixType
    SparseMatrixType_kQuant HAM;
    SparseMatrixType        OP1;

    HAM.SetID("HAM");
    OP1.SetID(S_OP);

    SparseMatrixBuilder builder;
    std::array<double, 2> spectral_bounds;

    // ── Load Hamiltonian ──────────────────────────────────────────────────────
    {
        std::string input = "operators/" + LABEL + ".HAM.CSR";
        builder.setSparseMatrix(&HAM);
        builder.BuildOPFromCSRFile(input);
        spectral_bounds = chebyshev::utility::SpectralBounds(HAM);
    }

    // ── Load Bloch phases and set up FFTW ─────────────────────────────────────
    {
        std::string phases_file = "operators/" + LABEL + ".BLOCH_PHASES";
        if (!HAM.ReadPhasesFromFile(phases_file))
        {
            std::cerr << "ERROR: Could not read Bloch phase file: "
                      << phases_file << std::endl;
            return 1;
        }
        HAM.PrepareFFT();
    }

    // ── Anderson disorder ─────────────────────────────────────────────────────
    if (disorder_amplitude > 0.0)
    {
        std::cout << "\nGenerating Anderson disorder  W = "
                  << disorder_amplitude << " eV ..." << std::endl;
        // Rescale amplitude to the adimensionalised units of H_bar:
        // disorder enters as V/a where a = HalfWidth (set below).
        // We pass the raw amplitude here; it will be rescaled after
        // BandWidth is set (see below).
        HAM.GenerateAndersonDisorder(disorder_amplitude);
    }
    else
        std::cout << "\nNo disorder (amplitude = 0)." << std::endl;

    // ── Load spectral operator ────────────────────────────────────────────────
    if (OP1.isIdentity())
        std::cout << "Operator " << OP1.ID()
                  << " treated as the identity." << std::endl;
    else
    {
        std::string input = "operators/" + LABEL + "." + OP1.ID() + ".CSR";
        builder.setSparseMatrix(&OP1);
        builder.BuildOPFromCSRFile(input);
    }

    // ── Configure Chebyshev moments ───────────────────────────────────────────
    const double half_width  = (spectral_bounds[1] - spectral_bounds[0]) * 1.0;
    const double band_center = (spectral_bounds[1] + spectral_bounds[0]) * 0.5;

    chebMoms.SystemLabel(LABEL);
    chebMoms.BandWidth (half_width);
    chebMoms.BandCenter(band_center);
    chebMoms.SetAndRescaleHamiltonian(HAM);
    chebMoms.Print();

    // Rescale disorder to adimensionalised units now that BandWidth is known.
    // H_bar = (H - b)/a  →  V_bar = V/a  →  disorder entries must be /a too.
    if (disorder_amplitude > 0.0 && !HAM.disorder.empty())
    {
        const double a = half_width;
        std::cout << "Rescaling disorder by 1/a = " << 1.0/a << std::endl;
        for (auto& v : HAM.disorder)
            v /= a;
    }

    // ── Random state generator ────────────────────────────────────────────────
    qstates::generator gen;
    if (has_state_file)
        gen = qstates::LoadStateFile(argv[5]);

    // ── Run KPM ───────────────────────────────────────────────────────────────
    chebyshev::SpectralMoments_kQuant( chebMoms, gen);

    postProcess(chebMoms);

    std::cout << "End of program" << std::endl;
    return 0;
}


void postProcess(chebyshev::Moments1D_kQuant& mu)
{
    const int numMoms   = mu.HighestMomentNumber();
    double    broadening = 1.0 / double(numMoms);

    mu.ApplyJacksonKernel(broadening);

    const int    num_div = 30 * numMoms;
    const double xbound  = chebyshev::CUTOFF;

    std::vector<double> energies(num_div);
    for (int i = 0; i < num_div; ++i)
        energies[i] = -xbound + i * (2*xbound) / (num_div - 1);

    std::cout << "Computing spectral function with " << numMoms << " moments\n"
              << "Grid: (" << -xbound << ", " << xbound << ")  step = "
              << energies[1] - energies[0] << "\n";

    std::string outputName = "mean" + mu.SystemLabel() + "JACKSON.dat";
    std::cout << "Saving to " << outputName << "\n";
    std::cout << "SystemSize = " << mu.SystemSize()
              << "  HalfWidth = " << mu.HalfWidth() << "\n";

    std::ofstream outputfile(outputName);
    std::vector<double> output(num_div, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_div; ++i)
    {
        double val = 0.0;
        const double energ = energies[i];
        for (int m = 0; m < numMoms; ++m)
            val += delta_chebF(energ, m) * mu(m).real();
        output[i] = val / mu.HalfWidth();
    }

    for (int i = 0; i < num_div; ++i)
        outputfile << energies[i] * mu.HalfWidth() + mu.BandCenter()
                   << " " << output[i] << "\n";

    outputfile.close();
    std::cout << "Done.\n";
}


void spectral::printHelpMessage()
{
    std::cout << "\nUsage:  kpm_dos  Label  Op  numMom  [disorder_W  [state_file]]\n\n"
              << "  Label          system label; loads operators/Label.HAM.CSR,\n"
              << "                 operators/Label.BLOCH_PHASES, operators/Label.Op.CSR\n"
              << "  Op             spectral operator name (use '1' for identity / DOS)\n"
              << "  numMom         number of Chebyshev moments\n"
              << "  disorder_W     Anderson disorder amplitude in eV (default: 0)\n"
              << "  state_file     optional path to an initial state file\n\n";
}

void spectral::printWelcomeMessage()
{
    std::cout << "──────────────────────────────────────────────────────\n"
              << "  KPM spectral function  (k-space H + real-space disorder)\n"
              << "──────────────────────────────────────────────────────\n";
}
