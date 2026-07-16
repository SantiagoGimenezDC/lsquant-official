// C& C++ libraries
#include <iostream> /* for std::cout mostly */
#include <string>   /* for std::string class */
#include <fstream>  /* for std::ofstream and std::ifstream functions classes*/
#include <stdlib.h>
#include <chrono>

#include "kpm_noneqop.hpp" //Message functions
#include "chebyshev_moments.hpp"
#include "sparse_matrix.hpp"
#include "quantum_states.hpp"
#include "chebyshev_solver.hpp"

#include "Kubo_solver_FFT.hpp"

namespace kpmKubo
{

void printHelpMessage();

void printWelcomeMessage();

}
int main(int argc, char *argv[])
{
	if ( !(argc == 5 || argc == 6) )
	{
		kpmKubo::printHelpMessage();
		return 0;
	}
	else
		kpmKubo::printWelcomeMessage();
	
	const std::string
		LABEL = argv[1],
	  S_NUM_MOM = argv[2],
	  S_NUM_R= argv[3];

	
	const int numMoms= atoi(argv[2]);

	int R = (argc >= 4) ? atoi(argv[3]) : 1;
	//R=1;
	
        int num_sections = 1, nump = 20*numMoms;
        const double disorder_amplitude = (argc >= 5) ? std::stod(argv[4]) : 0.0;

	chebyshev::formula sym_formula = chebyshev::KUBO_BASTIN;
	//chebyshev::Moments Hamiltonian_dummyMoms; //load number of moments


	SparseMatrixType_kQuant_nonOrth HAM;
        SparseMatrixType                OP_S, OP_dHk_1, OP_dHk_2, OP_dSk_1, OP_dSk_2;

    
	chebyshev::Vectors_sliced_kQuant_nonOrth 
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


    {
        std::string input = "operators/" + LABEL + ".S.CSR";
        builder.setSparseMatrix(&OP_S);
        builder.BuildOPFromCSRFile(input);
    }

    
    {
        std::string input = "operators/" + LABEL + ".dHk_x.CSR";
        builder.setSparseMatrix(&OP_dHk_1);
        builder.BuildOPFromCSRFile(input);
    }
    
    {
        std::string input = "operators/" + LABEL + ".dHk_x.CSR";
        builder.setSparseMatrix(&OP_dHk_2);
        builder.BuildOPFromCSRFile(input);
    }
    
    {
        std::string input = "operators/" + LABEL + ".dSk_x.CSR";
        builder.setSparseMatrix(&OP_dSk_1);
        builder.BuildOPFromCSRFile(input);
    }
    
    {
        std::string input = "operators/" + LABEL + ".dSk_x.CSR";
        builder.setSparseMatrix(&OP_dSk_2);
        builder.BuildOPFromCSRFile(input);
    }




    
     HAM.set_Hk(HAM.Matrix());
     HAM.set_S(OP_S.Matrix());
     HAM.set_dHk_1(OP_dHk_1.Matrix());
     HAM.set_dHk_2(OP_dHk_2.Matrix());
     HAM.set_dSk_1(OP_dSk_1.Matrix());
     HAM.set_dSk_2(OP_dSk_2.Matrix());

     
    // ── Configure Chebyshev moments ───────────────────────────────────────────
    const double half_width  = (spectral_bounds[1] - spectral_bounds[0]) * 1.0;
    const double band_center = (spectral_bounds[1] + spectral_bounds[0]) * 0.5;


    // Rescale disorder to adimensionalised units now that BandWidth is known.
    // H_bar = (H - b)/a  →  V_bar = V/a  →  disorder entries must be /a too.
    if (disorder_amplitude > 0.0 && !HAM.disorder.empty())
    {
        const double a = half_width;
        std::cout << "Rescaling disorder by 1/a = " << 1.0/a << std::endl;
        for (auto& v : HAM.disorder)
	  v /= a;  ///= a;
    }


    
	//CONFIGURE THE CHEBYSHEV MOMENTS
	chebVec.SystemLabel(LABEL);
	chebVec.BandWidth ( half_width );
        chebVec.BandCenter( band_center );
	chebVec.SetAndRescaleHamiltonian(HAM);

	chebVec_2 = chebVec;

	std::cout << "/*----------------------------------------------------------------------------------------*/" ;
	std::cout << std::endl<< std::endl;
	std::cout << "          THIS SOLVER & POSTPROCESS are TWEAKED FOR THE BISMUTHENE CASE ONLY!!!!!!         " << std::endl << std::endl;
	std::cout << "/*----------------------------------------------------------------------------------------*/" ;
	std::cout << std::endl;


	
	//Define thes states youll use
	//Factory state_factory ;

	//Compute the chebyshev expansion table
	qstates::generator gen;

	std::string outputfilename="Bastin_FFT_V1-V2"+LABEL+"KPM_M"+S_NUM_MOM+"x"+S_NUM_MOM+"_state"+gen.StateLabel()+".conductivity";

	chebyshev::Kubo_solver_FFT_kQuant_nonOrth solver(R, numMoms,  num_sections, nump, sym_formula, chebVec, chebVec_2,  outputfilename);
	solver.compute(  gen );

	//Save the table in a file



	std::cout<<"End of program"<<std::endl;
	return 0;
}

void kpmKubo::printHelpMessage()
{
	std::cout << "The program should be called with the following options: Label numMom num_states (default 1)" << std::endl
			  << std::endl;
	std::cout << "Label will be used to look for Label.Ham, Label.Op1 and Label.Op2" << std::endl;
	std::cout << "Op1 and Op2 will be used to located the sparse matrix file of two operators for the correlation" << std::endl;
	std::cout << "numMom will be used to set the number of moments in the chebyshev table" << std::endl;
};

void kpmKubo::printWelcomeMessage()
{
	std::cout << "WELCOME: This program will compute a table needed for expanding the correlation function in Chebyshev polynomialms" << std::endl;
};
