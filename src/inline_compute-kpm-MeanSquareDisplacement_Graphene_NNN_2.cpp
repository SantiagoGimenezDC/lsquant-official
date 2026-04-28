// C & C++ libraries
#include <iostream> /* for std::cout mostly */
#include <string>   /* for std::string class */
#include <fstream>  /* for std::ofstream and std::ifstream functions classes*/
#include <stdlib.h>
#include <chrono>


#include "kpm_noneqop.hpp" //Message functions
#include "chebyshev_moments.hpp"
#include "chebyshev_moments_Graphene_NNN.hpp"
#include "sparse_matrix.hpp"
#include "quantum_states.hpp"
#include "chebyshev_solver.hpp"
#include "chebyshev_solver_Graphene_NNN.hpp"
#include "Graphene_NNN.hpp"

namespace time_evolution
{
	void printHelpMessage();
	void printWelcomeMessage();
}


int main(int argc, char *argv[])
{
	if ( !(argc == 5 ) )
	{
		time_evolution::printHelpMessage();
		return 0;
	}
	else
		time_evolution::printWelcomeMessage();
	
	const std::string
	        S_SIZE = argv[1],
		S_NMOM = argv[2],
	        S_NTIME= argv[3],
		S_TMAX = argv[4];

	const int nMom_DOS = atoi(S_NMOM.c_str() );
	const int size = atoi(S_SIZE.c_str() );
	const int numTimes= atoi(S_NTIME.c_str() );
	const double tmax = stod(S_TMAX );


	
	Graphene_NNN graphene(size, false);


	
	SparseMatrixType OP[1];
	OP[0].SetID("DUMMY");

	std::array<double,2> spectral_bounds;
	spectral_bounds = chebyshev::utility::SpectralBounds(OP[0]);

	qstates::generator gen;
	
  
	graphene.Adimensionalize((spectral_bounds[1]-spectral_bounds[0]), (spectral_bounds[1]+spectral_bounds[0])*0.5);


	double dT= tmax / double(numTimes);

	std::cout<<"dT:  "<< dT<<std::endl;
	chebyshev::MeanSquareDisplacement_Graphene_NNN_2(graphene, nMom_DOS, numTimes, dT,  gen);




	//Post-processing data;
	const double a     = spectral_bounds[1] - spectral_bounds[0];
	const double b     = (spectral_bounds[1] + spectral_bounds[0]) * 0.5;
	const double tstep = tmax / double(numTimes);

	//Standard values for postprocessing
	const double Emin  = -1.0;
	const double Emax  =  1.0;
	const double dE    =  0.001;
	const double eta   =  0.00026;

	std::ofstream pp("postProcess.dat");
	if (!pp) throw std::runtime_error("Cannot open postProcess.dat");

	pp << "a         " << a         << "\n"
	   << "b         " << b         << "\n"
	   << "nMom_DOS  " << nMom_DOS  << "\n"
	   << "size      " << size      << "\n"
	   << "numTimes  " << numTimes  << "\n"

	   << "tmax      " << tmax      << "\n"
	   << "tstep     " << tstep     << "\n"

	   << "Emin      " << Emin      << "\n"
	   << "Emax      " << Emax      << "\n"
	   << "dE        " << dE        << "\n"
	   << "eta       " << eta       << "\n";

	pp.close();


	std::cout<<"End of program"<<std::endl;
	return 0;
}






void time_evolution::printHelpMessage()
	{
		std::cout << "The program should be called with the following options: Size numMom numTimeSteps MaxTime" << std::endl
				  << std::endl;
		std::cout << "Size is the the graphene lattice lattice side length in number of unit cells " << std::endl;
		std::cout << "numMom will be used to set the number of moments in the chebyshev table" << std::endl;
		std::cout << "numTimeSteps  will be used to set the number of timesteps in the chebyshev table" << std::endl;
		std::cout << "TimeMax  will be set the maximum time where the correlation will be evaluted " << std::endl;
	};

	inline
void time_evolution::printWelcomeMessage()
	{
		std::cout << "WELCOME: This program will compute a table needed for expanding the correlation function in Chebyshev polynomialms" << std::endl;
	};
