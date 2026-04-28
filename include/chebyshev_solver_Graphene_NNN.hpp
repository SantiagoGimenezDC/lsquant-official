#ifndef CHEBYSHEV_SOLVER_GRAPHENE_NNN
#define CHEBYSHEV_SOLVER_GRAPHENE_NNN

// C & C++ libraries
#include <cassert>   //for assert
#include <array>

#include <vector>    //for std::vector mostly class
#include <numeric>   //for std::accumulate *
#include <algorithm> //for std::max_elem
#include <complex>   ///for std::complex
#include <fstream>   //For ofstream
#include <limits>    //For getting machine precision limit
#include "sparse_matrix.hpp"
#include "chebyshev_moments.hpp"
#include "chebyshev_moments_Graphene_NNN.hpp"
#include "chebyshev_coefficients.hpp"
#include "linear_algebra.hpp"
#include <omp.h>
#include <chrono>
#include "quantum_states.hpp"
#include "kpm_noneqop.hpp" //Get Batch function
#include "special_functions.hpp"

namespace chebyshev
{


	int SpectralMoments_Graphene_NNN(SparseMatrixType &OP,  chebyshev::Moments1D_Graphene_NNN &chebMoms, qstates::generator& gen);




        int MeanSquareDisplacement_Graphene_NNN_2(Graphene_NNN &, int , int , double , qstates::generator&   );


  
  void evolution(Graphene_NNN&, std::vector<cdouble>&, std::vector<cdouble>&, std::vector<cdouble>&, std::vector<cdouble>& );

 

  //inline std::vector<cdouble> calculate_cn_bessel(double , double , double ); //Already included in chebyshev_solver.hpp. Defined in chebyshev_solver.cpp

  inline void get_dXY2(Graphene_NNN&, cdouble* , cdouble*, int , int );

  inline std::vector<cdouble> MomentosDelta(Graphene_NNN&, const cdouble*, int );



  
}; // namespace chebyshev

#endif
