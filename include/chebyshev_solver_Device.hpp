#ifndef CHEBYSHEV_SOLVER_DEVICE
#define CHEBYSHEV_SOLVER_DEVICE

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
#include "chebyshev_moments_Device.hpp"
#include "chebyshev_moments.hpp"

#include "chebyshev_coefficients.hpp"
#include "linear_algebra.hpp"
#include <omp.h>
#include <chrono>
#include "quantum_states.hpp"
#include "kpm_noneqop.hpp" //Get Batch function
#include "special_functions.hpp"



namespace chebyshev
{
        int TimeEvolvedOperator_Device(SparseMatrixType &OP,  chebyshev::MomentsTD_Device &chebMoms, qstates::generator& gen  );
}; // namespace chebyshev

#endif
