// C& C++ libraries
#include <iostream> /* for std::cout mostly */
#include <string>   /* for std::string class */
#include <fstream>  /* for std::ofstream and std::ifstream functions classes*/
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <cstdlib>

#include "kpm_noneqop.hpp" //Message functions
#include "chebyshev_moments.hpp"
#include "sparse_matrix.hpp"
#include "quantum_states.hpp"
#include "chebyshev_solver.hpp"

#include "Kubo_solver_FFT.hpp"

// Non-orthogonal counterpart of inline_compute-Kubo-kpm-FFT_kQuant: the Löwdin
// operators S^-1/2 (.) S^-1/2 are applied on the fly instead of being read
// pre-orthogonalized. Command line, disorder, seeds, energy grid and output
// names are kept identical to the kQuant driver, so both give the same
// result (up to the S^-1/2 Chebyshev truncation) for the same system.

namespace kpmKubo
{

void printHelpMessage();

void printWelcomeMessage();

}
int main(int argc, char *argv[])
{
	// Optional flags (any position):  --dis-seed N  --rv-seed N
	// Everything else is positional:  Label numMom R dis_amplitude [dis_type] [dis_conc]
	const unsigned int DEFAULT_DIS_SEED = 42, DEFAULT_RV_SEED = 3;
	unsigned int dis_seed = DEFAULT_DIS_SEED, rv_seed = DEFAULT_RV_SEED;
	std::vector<std::string> args;
	for ( int i = 1; i < argc; i++ )
	{
		const std::string arg = argv[i];
		if ( arg == "--dis-seed" || arg == "--rv-seed" )
		{
			if ( i + 1 >= argc )
			{
				std::cerr << "ERROR: " << arg << " needs a value" << std::endl;
				return 1;
			}
			const unsigned int value = std::stoul( argv[++i] );
			( arg == "--dis-seed" ? dis_seed : rv_seed ) = value;
		}
		else
			args.push_back( arg );
	}

	if ( args.size() < 4 || args.size() > 6 )
	{
		kpmKubo::printHelpMessage();
		return 0;
	}
	else
		kpmKubo::printWelcomeMessage();

	const std::string
		LABEL = args[0],
	  S_NUM_MOM = args[1],
	  S_NUM_R = args[2],
	  S_DIS_STR = args[3];

	// Optional: disorder type (default anderson) and concentration (magnetic types).
	const std::string DIS_TYPE   = ( args.size() > 4 ) ? args[4] : "anderson";
	const std::string S_DIS_CONC = ( args.size() > 5 ) ? args[5] : "";

	if ( DIS_TYPE != "anderson" && DIS_TYPE != "zmag" && DIS_TYPE != "rmag" )
	{
		std::cerr << "ERROR: unknown disorder type '" << DIS_TYPE
		          << "' (use anderson, zmag or rmag)" << std::endl;
		kpmKubo::printHelpMessage();
		return 1;
	}
	if ( DIS_TYPE != "anderson" && S_DIS_CONC.empty() )
	{
		std::cerr << "ERROR: disorder type '" << DIS_TYPE
		          << "' needs a concentration as 6th argument" << std::endl;
		return 1;
	}

	const int numMoms = atoi(S_NUM_MOM.c_str());
	int R = atoi(S_NUM_R.c_str());

	// Same energy grid as the kQuant driver (nump = 10*M).
        int num_sections = 1, nump = 10*numMoms;
        const double disorder_amplitude = std::stod(S_DIS_STR);
	const double disorder_concentration = S_DIS_CONC.empty() ? 0.0 : std::stod(S_DIS_CONC);

	chebyshev::formula sym_formula = chebyshev::KUBO_BASTIN;

	// Max Chebyshev order of the S^-1/2 expansion; nk and orbitals per spin are
	// taken from the BLOCH_PHASES header below (set_layout), not hard-coded.
	SparseMatrixType_kQuant_nonOrth_ChrisVel HAM(200, 46, 6400);
        SparseMatrixType                OP_S, OP_dHk_1, OP_dHk_2, OP_Ak_1, OP_Ak_2, OP_Sz;

	chebyshev::Vectors_sliced_kQuant_nonOrth_ChrisVel
	  chebVec( numMoms, num_sections ),
	  chebVec_2( numMoms, num_sections );

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
        HAM.set_layout(HAM.Nk, HAM.W);   // k-blocks and orbitals per spin for S^-1, S^-1/2
        HAM.PrepareFFT();
    }

    // ── Disorder ──────────────────────────────────────────────────────────────
    // The raw amplitude is used here; it is rescaled to the adimensionalised
    // units of H_bar (V/a) once BandWidth is known (see below).
    if (disorder_amplitude > 0.0)
    {
        const bool spinful = ( HAM.rank() == 2 * HAM.Nk * HAM.W );
        if ( DIS_TYPE != "anderson" && !spinful )
        {
            std::cerr << "ERROR: magnetic disorder needs a spinful matrix with blocked "
                      << "spin layout (rank = 2*Nk*W); got rank " << HAM.rank()
                      << ", Nk*W = " << HAM.Nk * HAM.W << std::endl;
            return 1;
        }

        if ( DIS_TYPE == "anderson" )
        {
            std::cout << "\nGenerating Anderson disorder  W = "
                      << disorder_amplitude << " eV ..." << std::endl;
            HAM.GenerateAndersonDisorder(disorder_amplitude, dis_seed);
        }
        else if ( DIS_TYPE == "zmag" )
        {
            std::cout << "\nGenerating z-magnetic disorder  strength = " << disorder_amplitude
                      << " eV, concentration = " << disorder_concentration << " ..." << std::endl;
            HAM.Generate_zMagneticDisorder(disorder_amplitude, disorder_concentration, dis_seed);
        }
        else // rmag
        {
            std::cout << "\nGenerating random-direction magnetic disorder  strength = "
                      << disorder_amplitude << " eV, concentration = "
                      << disorder_concentration << " ..." << std::endl;
            HAM.Generate_RdirMagneticDisorder(disorder_amplitude, disorder_concentration, dis_seed,
                                              "operators/" + LABEL + ".orbital_map.dat");
        }
    }
    else
        std::cout << "\nNo disorder (amplitude = 0)." << std::endl;


    // ── Non-orthogonal operators ──────────────────────────────────────────────
    {
        std::string input = "operators/" + LABEL + ".S.CSR";
        builder.setSparseMatrix(&OP_S);
        builder.BuildOPFromCSRFile(input);
    }

    {
        std::string input = "operators/" + LABEL + ".Sz.CSR";
        builder.setSparseMatrix(&OP_Sz);
        builder.BuildOPFromCSRFile(input);
    }

    {
        std::string input = "operators/" + LABEL + ".dHk_x.CSR";
        builder.setSparseMatrix(&OP_dHk_1);
        builder.BuildOPFromCSRFile(input);
    }

    {
        std::string input = "operators/" + LABEL + ".dHk_y.CSR";
        builder.setSparseMatrix(&OP_dHk_2);
        builder.BuildOPFromCSRFile(input);
    }

    {
        std::string input = "operators/" + LABEL + ".Ak_x.CSR";
        builder.setSparseMatrix(&OP_Ak_1);
        builder.BuildOPFromCSRFile(input);
    }

    {
        std::string input = "operators/" + LABEL + ".Ak_y.CSR";
        builder.setSparseMatrix(&OP_Ak_2);
        builder.BuildOPFromCSRFile(input);
    }

     HAM.set_Hk(HAM.Matrix());
     HAM.set_S(OP_S.Matrix());
     HAM.set_dHk_1(OP_dHk_1.Matrix());
     HAM.set_dHk_2(OP_dHk_2.Matrix());
     HAM.set_A_1(OP_Ak_1.Matrix());
     HAM.set_A_2(OP_Ak_2.Matrix());
     HAM.set_Sz(OP_Sz.Matrix());


    // ── Configure Chebyshev moments ───────────────────────────────────────────
    const double band_width  = (spectral_bounds[1] - spectral_bounds[0]) * 1.0;
    const double band_center = (spectral_bounds[1] + spectral_bounds[0]) * 0.5;

    // Rescale disorder to adimensionalised units now that BandWidth is known.
    // H_bar = (H - b)/a  →  V_bar = V/a  →  disorder entries must be /a too.
    // (Covers both the scalar `disorder` vector and the magnetic disorder_x/y/z.)
    if (disorder_amplitude > 0.0)
    {
        const double a = band_width;
        std::cout << "Rescaling disorder by 1/a = " << 1.0/a << std::endl;
        for (auto& v : HAM.disorder)
	  v /= a;
        for (auto& v : HAM.disorder_x)
	  v /= a;
        for (auto& v : HAM.disorder_y)
	  v /= a;
        for (auto& v : HAM.disorder_z)
	  v /= a;
    }

	//CONFIGURE THE CHEBYSHEV MOMENTS
	chebVec.SystemLabel(LABEL);
	chebVec.BandWidth ( band_width );
        chebVec.BandCenter( band_center );

	chebVec.SetAndRescaleHamiltonian_nonOrth(HAM);

	chebVec_2 = chebVec;

	std::cout << "/*----------------------------------------------------------------------------------------*/" ;
	std::cout << std::endl<< std::endl;
	std::cout << "          THIS POSTPROCESS is TWEAKED FOR THE BISMUTHENE CASE ONLY!!!!!!         " << std::endl << std::endl;
	std::cout << "/*----------------------------------------------------------------------------------------*/" ;
	std::cout << std::endl;


	//Compute the chebyshev expansion table
	qstates::generator gen;   // its constructor calls srand(3)
	// Seed the random vectors last, so disorder generation (Anderson uses
	// rand() too) cannot shift them. Default 3 = previous behaviour.
	std::srand( rv_seed );
	std::cout << "Disorder seed: " << dis_seed << "   Random-vector seed: " << rv_seed << std::endl;

	// Output name records the disorder type (and concentration for magnetic types).
	std::string dis_tag;
	if ( disorder_amplitude <= 0.0 )      dis_tag = "_noDisorder";
	else if ( DIS_TYPE == "anderson" )    dis_tag = "_Anderson";
	else if ( DIS_TYPE == "zmag" )        dis_tag = "_zMagnetic_disConc" + S_DIS_CONC;
	else                                  dis_tag = "_RdirMagnetic_disConc" + S_DIS_CONC;

	// Seeds are only tagged when not the default, so default names are unchanged.
	if ( disorder_amplitude > 0.0 && dis_seed != DEFAULT_DIS_SEED ) dis_tag += "_disSeed" + std::to_string(dis_seed);
	if ( rv_seed != DEFAULT_RV_SEED )                               dis_tag += "_rvSeed"  + std::to_string(rv_seed);

	std::string outputfilename="Bastin_FFT_V1-V2_"+LABEL+"KPM_M"+S_NUM_MOM+"x"+S_NUM_MOM+"_disAmp"+S_DIS_STR+dis_tag+".conductivity";

	chebyshev::Kubo_solver_FFT_kQuant_nonOrth_ChrisVel solver(R, numMoms,  num_sections, nump, sym_formula, chebVec, chebVec_2,  outputfilename);
	solver.compute(  gen );

	std::cout<<"End of program"<<std::endl;
	return 0;
}

void kpmKubo::printHelpMessage()
{
	std::cout << "The program should be called with the following options:" << std::endl
		  << "  Label numMom num_states dis_amplitude [dis_type] [dis_concentration]" << std::endl
		  << std::endl
		  << "  dis_type (default anderson):" << std::endl
		  << "    anderson  on-site Anderson disorder, amplitude in eV" << std::endl
		  << "    zmag      z-oriented magnetic impurities (+v up, -v down); needs dis_concentration" << std::endl
		  << "    rmag      random-direction magnetic impurities per atom; needs dis_concentration" << std::endl
		  << "              and operators/<Label>.orbital_map.dat" << std::endl
		  << "  Magnetic types need a spinful matrix with blocked spin layout (rank = 2*Nk*W)." << std::endl
		  << std::endl
		  << "  Optional flags (any position):" << std::endl
		  << "    --dis-seed N   disorder seed       (default 42)" << std::endl
		  << "    --rv-seed N    random-vector seed  (default 3)" << std::endl
		  << "  Non-default seeds are added to the output name (_disSeedN, _rvSeedN)." << std::endl
		  << std::endl
		  << "  Reads operators/<Label>.{HAM,S,Sz,dHk_x,dHk_y,Ak_x,Ak_y}.CSR and .BLOCH_PHASES." << std::endl;
};

void kpmKubo::printWelcomeMessage()
{
	std::cout << "WELCOME: This program will compute a table needed for expanding the correlation function in Chebyshev polynomialms" << std::endl;
};
