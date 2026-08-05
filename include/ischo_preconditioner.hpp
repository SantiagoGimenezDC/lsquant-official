#ifndef ISCHO_PRECONDITIONER_HPP
#define ISCHO_PRECONDITIONER_HPP

// ischo_preconditioner.hpp
//
// Incomplete Stabilized Cholesky (ISCHO) preconditioner for a complex
// Hermitian (positive-definite-ish) sparse overlap matrix S, as described in:
//
//   "... We use an incomplete stabilized Cholesky (ISCHO) factorization for
//   this purpose. In the ISCHO one performs a Cholesky factorization in
//   which one throws away all elements that are not within the sparsity
//   pattern of S, i.e. S is approximated by S ~ L~ L~^H = S~. This renders
//   the usually extremely stable Cholesky algorithm unstable due to the
//   appearance of negative or zero diagonal elements in S~. We correct this
//   effect by adding a diagonal matrix with small elements of equal size
//   until the resulting S~ is positive definite ..."
//
// Eigen's Eigen::IncompleteCholesky<> is exactly this algorithm (Lin & Moré,
// "Incomplete Cholesky Factorizations with Limited Memory", SIAM J. Sci.
// Comput. 21(1), 1999): it factorizes only within the sparsity pattern of
// the input matrix and adds a diagonal shift sigma*I whenever the pivots
// would otherwise be non-positive, i.e.
//
//      S_scaled_permuted + sigma*I  ~  L * L^H         (L lower triangular)
//
// where
//      S_scaled_permuted = Dscale * P * S * P^T * Dscale
//
// (Dscale: diagonal scaling, P: fill-reducing permutation).
//
// This header:
//   1) Calls Eigen's incomplete Cholesky (ISCHO) factorization on S.
//   2) Implements the preconditioner "solve" step explicitly via forward
//      substitution with L and backward substitution with L^H (i.e. L*),
//      instead of relying on Eigen's built-in solve(), so the fwd/back
//      substitution required by the paper is transparent and can be reused
//      / modified (e.g. as the PCG preconditioner application, or to build
//      a good initial guess for Eq. (8): S~ |x> = H |m-1>).



#pragma once

#include <eigen-3.4.0/Eigen/Sparse>
#include <eigen-3.4.0/Eigen/IterativeLinearSolvers>
#include <eigen-3.4.0/Eigen/OrderingMethods>

#include <complex>
#include <stdexcept>

template <typename IndexType>
class IschoPreconditioner
{
public:
    using Scalar        = std::complex<double>;
    using RealScalar     = double;

    // Matches the layout used for S in the caller's code.
    using SparseMatS    = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, IndexType>;

    // Eigen::IncompleteCholesky internally wants a column-major matrix.
    using SparseMatCol   = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, IndexType>;

    using Vector          = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVector       = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    using OrderingType   = Eigen::AMDOrdering<IndexType>;
    using ICholT           = Eigen::IncompleteCholesky<Scalar, Eigen::Lower, OrderingType>;

    using FactorType       = typename ICholT::FactorType;       // sparse lower-tri L
    using PermutationType = typename ICholT::PermutationType;  // fill-reducing permutation P

    IschoPreconditioner() = default;
    explicit IschoPreconditioner(const SparseMatS& S) { compute(S); }

    // ---------------------------------------------------------------
    // Step 1: call Eigen's incomplete Cholesky (== ISCHO) factorization
    // ---------------------------------------------------------------
    // Produces:  Dscale * P * S * P^T * Dscale + sigma*I  ~  L * L^H
    void compute(const SparseMatS& S)
    {
        if (S.rows() != S.cols())
            throw std::invalid_argument("IschoPreconditioner::compute: S must be square");

        m_n = S.rows();

        // IncompleteCholesky requires column-major storage.
        SparseMatCol Scol(S);

        m_ichol.compute(Scol);

        if (m_ichol.info() != Eigen::Success)
            throw std::runtime_error(
                "IschoPreconditioner::compute: incomplete Cholesky (ISCHO) "
                "factorization failed (matrix could not be stabilized to "
                "positive definiteness).");

        // Cache the factor, scaling, and permutation so we can do the
        // forward/backward substitution ourselves.
        m_L     = m_ichol.matrixL();     // lower triangular sparse factor L
        m_scale = m_ichol.scalingS();     // diagonal scaling Dscale (real, positive)
        m_perm  = m_ichol.permutationP(); // fill-reducing permutation P
    }

    void compute(const SparseMatS* S)
    {
        if (!S) throw std::invalid_argument("IschoPreconditioner::compute: null S pointer");
        compute(*S);
    }

    bool isComputed() const { return m_n > 0; }

    // ---------------------------------------------------------------
    // Step 2: apply the preconditioner, i.e. approximately solve
    //             S~ |x> = |vec>     for |x> = S~^{-1} |vec>
    // using explicit forward substitution with L, then backward
    // substitution with L^H = L*, matching Eq. (9): S ~ L~ L~^H = S~.
    //
    // This is the routine you plug into a PCG iteration as z = M^{-1} r,
    // and it is also exactly what you'd use to build the initial guess
    // |x0> = S~^{-1} H|m-1> mentioned in the paper.
    // ---------------------------------------------------------------
    Vector solve(const Vector& vec) const
    {
        if (!isComputed())
            throw std::runtime_error("IschoPreconditioner::solve: compute() has not been called");
        if (vec.size() != m_n)
            throw std::invalid_argument("IschoPreconditioner::solve: size mismatch");

        // 1) apply fill-reducing permutation:      x <- P * vec
        Vector x = m_perm * vec;

        // 2) apply diagonal scaling:                x <- Dscale * x
        x = m_scale.asDiagonal() * x;

        // 3) forward substitution:  solve  L * y = x   for y
        //    (L is lower triangular -> simple forward sweep)
        Vector y = m_L.template triangularView<Eigen::Lower>().solve(x);

        // 4) backward substitution: solve  L^H * z = y  for z
        //    (L^H = L* is upper triangular -> simple backward sweep)
        Vector z = m_L.adjoint().template triangularView<Eigen::Upper>().solve(y);

        // 5) undo scaling:                          z <- Dscale * z
        z = m_scale.asDiagonal() * z;

        // 6) undo permutation:                      z <- P^{-1} * z
        z = m_perm.inverse() * z;

        return z; // == S~^{-1} * vec
    }

    // Convenience operator so the preconditioner can be dropped directly
    // into hand-written PCG loops as  z = precond(r);
    Vector operator()(const Vector& vec) const { return solve(vec); }

    // --- accessors, useful for diagnostics / reuse -------------------
    const FactorType& matrixL() const { return m_L; }
    const RealVector& scaling() const { return m_scale; }
    const PermutationType& permutation() const { return m_perm; }
    IndexType size() const { return m_n; }

private:
    ICholT           m_ichol;
    FactorType       m_L;
    RealVector       m_scale;
    PermutationType  m_perm;
    IndexType        m_n = 0;
};



#endif 
