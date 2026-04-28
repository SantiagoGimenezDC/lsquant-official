// Used for OPENMP functions
#include "chebyshev_solver_Device.hpp"
#include "chebyshev_solver.hpp"


using namespace chebyshev;






int chebyshev::TimeEvolvedOperator_Device(SparseMatrixType &OP,  chebyshev::MomentsTD_Device &chebMoms, qstates::generator& gen  )
{
	const auto Dim = chebMoms.SystemSize();
	const auto NumMoms = chebMoms.HighestMomentNumber();
	const auto NumTimes= chebMoms.MaxTimeStep();


	if( chebMoms.Chebyshev2().size()!= Dim )
	     chebMoms.Chebyshev2() = Moments_Device::vector_t(Dim,Moments_Device::value_t(0)); 

		
	//Initialize the Random Phase vector used for the Trace approximation
	gen.SystemSize(Dim);	
	while( gen.getQuantumState() )
	{
		chebMoms.ResetTime();

		auto PhiR = gen.State();
		auto PhiL = PhiR;

		
		//Multiply right operator its operator
		//{
		auto PhiT = PhiR;
		chebMoms.device().spin_project( PhiR.data(), 2 );
		// OP.Multiply(PhiL,tempPhiL); //Defines <Phi| OP
		linalg::copy(PhiR, PhiL);
		//}

		int t=0;
		//Evolve state vector from t=0 to Tmax
		while ( chebMoms.CurrentTimeStep() !=  chebMoms.MaxTimeStep()  )
		{
			const auto n = chebMoms.CurrentTimeStep();

			//Set the evolved vector as initial vector of the chebyshev iterations
			chebMoms.SetInitVectors( PhiR );

			for(int m = 0 ; m < NumMoms ; m++ )
			{
				double scal=2.0/gen.NumberOfStates();
				if( m==0) scal*=0.5;
				//scal*=2;//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////REMOVE
				//OP.Multiply( chebMoms.Chebyshev0(), PhiT );

				chebMoms.device().J(  PhiT.data(), chebMoms.Chebyshev0().data(), 2 );
				chebMoms(m,n) += scal*linalg::vdot( PhiL, PhiT ) ;
				chebMoms.Iterate();
			}

			
			t++;
			cout<< t<<"/"<< chebMoms.MaxTimeStep()<<"  timestep:  " <<std::endl;
			chebMoms.IncreaseTimeStep();
			//evolve PhiL ---> PhiLt , PhiR ---> PhiRt 
			cout<<"  First time evolution:"<<std::endl;
			chebMoms.Evolve(PhiL) ;
			cout<<"  Second time evolution:"<<std::endl;
			chebMoms.Evolve(PhiR) ;

			
		        std::string outputfilename="TimeEvol_currentResult.chebmomTD";	
			chebMoms.saveIn(outputfilename);
			std::cout<<std::endl<<std::endl;

		}
	
	}
	
	return 0;
};



