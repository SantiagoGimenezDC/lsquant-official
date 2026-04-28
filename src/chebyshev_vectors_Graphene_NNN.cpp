#include "chebyshev_moments_Graphene_NNN.hpp"



void chebyshev::Moments_Graphene_NNN::SetInitVectors( const Moments_Graphene_NNN::vector_t& T0 )
{


	assert( T0.size() == this->SystemSize() );
	const auto dim = this->SystemSize();

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 
	if( this->Chebyshev2().size()!= dim )
		this->Chebyshev2() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 

	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()

	linalg::copy ( T0, this->Chebyshev0() );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
        this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());
};




void chebyshev::Moments_Graphene_NNN::SetInitVectors( SparseMatrixType &OP , const Moments_Graphene_NNN::vector_t& T0 )
{

        const auto dim = this->SystemSize();
	assert( OP.rank() == this->SystemSize() && T0.size() == this->SystemSize() );

	if( this->Chebyshev0().size()!= dim )
		this->Chebyshev0() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 

	if( this->Chebyshev1().size()!= dim )
		this->Chebyshev1() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 
	if( this->Chebyshev2().size()!= dim )
		this->Chebyshev2() = Moments_Graphene_NNN::vector_t(dim,Moments_Graphene_NNN::value_t(0)); 

	//From now on this-> will be discarded in Chebyshev0() and Chebyshev1()


	linalg::copy ( T0, this->Chebyshev1() );
	//OP.Multiply( 1.0, this->Chebyshev1(), 0.0, this->Chebyshev0() );

	//this->device().J(this->Chebyshev0().data(), this->Chebyshev1().data(), 2 );
	//this->Hamiltonian().Multiply( 1.0, this->Chebyshev0(), 0.0, this->Chebyshev1() );
	this->device().H_ket(Chebyshev1().data(), Chebyshev0().data());

	
	return ;
};




	
	
