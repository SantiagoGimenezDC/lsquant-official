#include "chebyshev_moments_Device.hpp"



void chebyshev::Moments_Device::SetInitVectors( const Moments_Device::vector_t& T0 )
{


	assert( T0.size() == this->SystemSize() );
	const auto dim = this->SystemSize();

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 
	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()

	linalg::copy ( T0, this->Chebyshev0() );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
        this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());
};




void chebyshev::Moments_Device::SetInitVectors( SparseMatrixType &OP , const Moments_Device::vector_t& T0 )
{

        const auto dim = this->SystemSize();
	assert( OP.rank() == this->SystemSize() && T0.size() == this->SystemSize() );

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 
	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()


	linalg::copy ( T0, this->Chebyshev1() );
	//OP.Multiply( 1.0, this->Chebyshev1(), 0.0, this->Chebyshev0() );

	this->device().J(this->Chebyshev0().data(), this->Chebyshev1().data(), 2 );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
	this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());

	
	return ;
};




void chebyshev::Vectors_sliced_Device::SetInitVectors_2( SparseMatrixType &OP , const Moments_Device::vector_t& T0 )
{

        const auto dim = this->SystemSize();
	assert( OP.rank() == this->SystemSize() && T0.size() == this->SystemSize() );

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 
	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()


	linalg::copy ( T0, this->Chebyshev1() );
	OP.Multiply( 1.0, this->Chebyshev1(), 0.0, this->Chebyshev0() );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
        this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());

	return ;
};




void chebyshev::Vectors_sliced_Device::SetInitVectors_2( const Moments_Device::vector_t& T0 )
{

	assert( T0.size() == this->SystemSize() );
	const auto dim = this->SystemSize();

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 
	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()

	linalg::copy ( T0, this->Chebyshev0() );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
	this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());
};





int chebyshev::Vectors_Device::IterateAll( )
{	
	//The vectorss Chebyshev0() and Chebyshev1() are assumed to have
	// been initialized
	linalg::copy( this->Chebyshev0() ,this->Vector(0) );
	for(int m=1; m < this->NumberOfVectors(); m++ )
	{
		linalg::copy( Chebyshev1() , this->Vector(m) );
		this->Hamiltonian().Multiply(2.0,Chebyshev1(),-1.0,Chebyshev0());
		Chebyshev0().swap(Chebyshev1());
	}
	return 0;
};


int chebyshev::Vectors_Device::Multiply( SparseMatrixType &OP )
{
	assert( OP.rank() == this->SystemSize() );
	if( this->OPV.size()!= OP.rank() )
		this->OPV = Moments_Device::vector_t ( OP.rank() );
	
	for(size_t m=0; m < this->NumberOfVectors(); m++ )
	{
		linalg::copy( this->Chebmu.ListElem(m), this->OPV ); 
		OP.Multiply(  this->OPV, this->Chebmu.ListElem(m) );
	}

	return 0;
};






int chebyshev::Vectors_sliced_Device::IterateAllSliced(int s )
{

  size_t segment_size = ( s == num_sections_-1 ? last_section_size_ : section_size_ ),
         segment_start = s * section_size_,
         DIM = this->SystemSize();

  
	//The vectorss Chebyshev0() and Chebyshev1() are assumed to have
	// been initialized

        linalg::extract_segment( Chebyshev0(),  segment_start, Vector(0));
	//linalg::copy( this->Chebyshev0() ,this->Vector(0) );

	for(int m=1; m < this->NumberOfVectors(); m++ )
	{
	  
	  linalg::extract_segment( Chebyshev1(),  segment_start, Vector(m));
	  this->Hamiltonian().Multiply(2.0,Chebyshev1(),-1.0,Chebyshev0());
	  Chebyshev0().swap(Chebyshev1());
	  	
}
	return 0;
};


int chebyshev::Vectors_sliced_Device::MultiplySliced( SparseMatrixType &OP, int s)
{
  size_t segment_size = ( s == num_sections_-1 ? last_section_size_ : section_size_ ),
    segment_start = s * section_size_,
    DIM = this->SystemSize();

  Moments_Device::vector_t tmp2(this->SystemSize());


	assert( OP.rank() == this->SystemSize() );
	if( OPV().size()!= OP.rank() )
	       OPV() = Moments_Device::vector_t ( OP.rank() );

#pragma omp parallel for
	for(int i=0; i<OP.rank(); i++){//not parallelized; with omp/ eigen this is straightforward;
	  OPV()[i] = 0.0;
	  tmp2[i] = 0.0;
	}

	
	for(int m=0; m < this->NumberOfVectors(); m++ )
	{
	  linalg::introduce_segment(Chebmu_.ListElem(m), OPV(), segment_start);
	  OP.Multiply(1.0,OPV(),0.0, tmp2); //Multiply(  OPV(), tmp2 );
	  linalg::extract_segment(tmp2, segment_start,  Chebmu_.ListElem(m));	  
	}

	return 0;
};


int chebyshev::Vectors_Device::EvolveAll(const double DeltaT, const double Omega0)
{
	const auto dim = this->SystemSize();
	const auto numVecs = this->NumberOfVectors();

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Device::vector_t(dim,Moments_Device::value_t(0)); 
	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()

	const auto I = Moments_Device::value_t(0, 1);
	const double x = Omega0*DeltaT;
	for(size_t m=0; m < this->NumberOfVectors(); m++ )
	{
		auto& myVec = this->Vector(m);
		
		int n = 0;
		double Jn = besselJ(n,x);
		linalg::copy(myVec , Chebyshev0());
		linalg::scal(0, myVec); //Set to zero
		linalg::axpy( Jn , Chebyshev0(), myVec);

		double Jn1 = besselJ(n+1,x);	
		this->Hamiltonian().Multiply(Chebyshev0(), Chebyshev1());
		linalg::axpy(-value_t(2) * I * Jn1, Chebyshev1(), myVec);
		
		auto nIp =-I;
		while( 0.5*(std::abs(Jn)+std::abs(Jn1) ) > 1e-15)
		{
			nIp*=-I ;
			Jn  = Jn1;
			Jn1 = besselJ(n, x);
			this->Hamiltonian().Multiply(2.0, Chebyshev1(), -1.0, Chebyshev0());
			linalg::axpy(value_t(2) * nIp * value_t(Jn1), Chebyshev0(), myVec);
			Chebyshev0().swap(Chebyshev1());
			n++;
		}
	}
  return 0;
};


double chebyshev::Vectors_Device::MemoryConsumptionInGB()
{
	return SizeInGB()+2.0*( (double)this->SystemSize() )*pow(2.0,-30.0) ;
}
	
	
