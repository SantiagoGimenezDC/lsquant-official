#include "chebyshev_moments_Device.hpp"
#include "linear_algebra.hpp"








	//light functions
int chebyshev::Moments_Device::JacksonKernelMomCutOff( const double broad )
{
	assert( broad >0 );
	const double eta   =  2.0*broad/1000/this->BandWidth();
	return ceil(M_PI/eta);
};
	
//light functions
double chebyshev::Moments_Device::JacksonKernel(const double m,  const double Mom )
{
	const double
	phi_J = M_PI/(double)(Mom+1.0);
	return ( (Mom-m+1)*cos( phi_J*m )+ sin(phi_J*m)/tan(phi_J) )*phi_J/M_PI;
};


int chebyshev::Moments_Device::Iterate( )
{

  
  this->device().update_cheb(  Chebyshev2().data(), Chebyshev1().data(),   Chebyshev0().data() );

  //linalg::copy(Chebyshev1(),Chebyshev0());
  //linalg::copy(Chebyshev2(),Chebyshev1());
  
	

  //this->Hamiltonian().Multiply(2.0,this->Chebyshev1(),-1.0,this->Chebyshev0());
  //this->Chebyshev0().swap(this->Chebyshev1());
  
	return 0;
};
