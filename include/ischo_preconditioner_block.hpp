#ifndef ISCHO_PRECONDITIONER_BLOCK_HPP
#define ISCHO_PRECONDITIONER_BLOCK_HPP

// ischo_preconditioner.hpp
//
// Incomplete Stabilized Cholesky (ISCHO) preconditioner for a complex
// Hermitian block-diagonal overlap matrix S arising from a DFT supercell
// calculation with nk k-points, each contributing an independent
// n_orb x n_orb Hermitian sub-block along the diagonal of S:
//
//      S = blockdiag( S_0, S_1, ..., S_{nk-1} ),   each S_k in C^{n_orb x n_orb}
//
// S is still passed as ONE big sparse matrix (same interface as before),
// but since the k-points do not couple to each other, an ISCHO
// factorization of the full S is mathematically identical to factorizing
// each block independently -- and factorizing block-by-block is cheaper,
// trivially parallelizable, and lets us pinpoint which k-point (if any)
// fails to stabilize to positive definiteness.
//
//      S_k ~ L_k L_k^H = S~_k         for each k = 0 .. nk-1
//
// (with the usual ISCHO diagonal scaling / permutation / stabilizing
// shift applied independently within each block, exactly as described in
// the paper for the full matrix.)
//
// Public interface:
//   IschoPreconditioner<IndexType> M(nk, n_orb);
//   M.compute(S);              // S: nk*n_orb square sparse matrix, block-diagonal
//   Vector x = M.solve(b);     // x ~= S^{-1} b   (unchanged from the non-blocked version)

#pragma once

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/Eigen/OrderingMethods>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename IndexType>
class IschoPreconditioner_block
{
public:
    using Scalar        = std::complex<double>;
    using RealScalar     = double;

    // Matches the layout used for S/H in the caller's code.
    using SparseMatS    = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, IndexType>;

    // Per-block matrix (small, n_orb x n_orb). IncompleteCholesky wants
    // column-major storage.
    using SparseBlock    = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, IndexType>;

    using Vector           = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using RealVector        = Eigen::Matrix<RealScalar, Eigen::Dynamic, 1>;

    using OrderingType    = Eigen::AMDOrdering<IndexType>;
    using ICholT            = Eigen::IncompleteCholesky<Scalar, Eigen::Lower, OrderingType>;

    using FactorType        = typename ICholT::FactorType;       // sparse lower-tri L_k
    using PermutationType  = typename ICholT::PermutationType;  // fill-reducing permutation P_k

    // nk: number of k-points, n_orb: orbitals per k-point block.
    IschoPreconditioner_block(IndexType nk, IndexType n_orb)
        : m_nk(nk), m_norb(n_orb), m_n(nk * n_orb)
    {
        if (nk <= 0 || n_orb <= 0)
            throw std::invalid_argument("IschoPreconditioner: nk and n_orb must be positive");
        m_blockL.resize(m_nk);
        m_blockScale.resize(m_nk);
        m_blockPerm.resize(m_nk);
    }

    // ---------------------------------------------------------------
    // Step 1: call Eigen's incomplete Cholesky (== ISCHO) factorization
    // independently for each of the nk diagonal n_orb x n_orb blocks.
    // ---------------------------------------------------------------
    void compute(const SparseMatS& S)
    {
        if (S.rows() != S.cols())
            throw std::invalid_argument("IschoPreconditioner::compute: S must be square");
        if (S.rows() != m_n)
            throw std::invalid_argument(
                "IschoPreconditioner::compute: S size (" + std::to_string(S.rows()) +
                ") does not match nk*n_orb (" + std::to_string(m_n) + ")");

        m_failedBlocks.clear();

        // Extract each k-point block as its own small sparse matrix, then
        // run ISCHO on it independently. This loop is embarrassingly
        // parallel across k, since blocks share no data.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            SparseBlock Sk = extractBlock(S, k);

            ICholT ichol;
            ichol.compute(Sk);

            if (ichol.info() != Eigen::Success)
            {
                // Note: throwing inside an OpenMP region is unsafe; flag
                // instead and raise after the parallel loop.
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    m_failedBlocks.push_back(k);
                }
                continue;
            }

            m_blockL[k]     = ichol.matrixL();
            m_blockScale[k] = ichol.scalingS();
            m_blockPerm[k]  = ichol.permutationP();
        }

        if (!m_failedBlocks.empty())
        {
            std::string msg = "IschoPreconditioner::compute: ISCHO failed to stabilize "
                                                 "to positive definiteness for k-block(s): ";
            for (size_t i = 0; i < m_failedBlocks.size(); ++i)
                msg += std::to_string(m_failedBlocks[i]) + (i + 1 < m_failedBlocks.size() ? ", " : "");
            throw std::runtime_error(msg);
        }

        m_computed = true;
    }

    void compute(const SparseMatS* S)
    {
        if (!S) throw std::invalid_argument("IschoPreconditioner::compute: null S pointer");
        compute(*S);
    }

    bool isComputed() const { return m_computed; }

    // ---------------------------------------------------------------
    // Step 2: apply the preconditioner block-by-block, i.e. approximately
    // solve  S~_k |x_k> = |vec_k>  independently for each k-point segment,
    // via forward substitution with L_k then backward substitution with
    // L_k^H, exactly as in the non-blocked version -- interface unchanged.
    // ---------------------------------------------------------------
    Vector solve(const Vector& vec) const
    {
        if (!m_computed)
            throw std::runtime_error("IschoPreconditioner::solve: compute() has not been called");
        if (vec.size() != m_n)
            throw std::invalid_argument("IschoPreconditioner::solve: size mismatch");

        Vector result(m_n);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            const IndexType offset = k * m_norb;
            Vector xk = vec.segment(offset, m_norb);

            // 1) permute:               x <- P_k * xk
            Vector x = m_blockPerm[k] * xk;

            // 2) scale:                  x <- Dscale_k * x
            x = m_blockScale[k].asDiagonal() * x;

            // 3) forward substitution:   solve  L_k * y = x   for y
            Vector y = m_blockL[k].template triangularView<Eigen::Lower>().solve(x);

            // 4) backward substitution:  solve  L_k^H * z = y  for z
            Vector z = m_blockL[k].adjoint().template triangularView<Eigen::Upper>().solve(y);

            // 5) undo scale:             z <- Dscale_k * z
            z = m_blockScale[k].asDiagonal() * z;

            // 6) undo permute:           z <- P_k^{-1} * z
            z = m_blockPerm[k].inverse() * z;

            result.segment(offset, m_norb) = z;
        }

        return result; // == S~^{-1} * vec, block-by-block
    }

    Vector operator()(const Vector& vec) const { return solve(vec); }

    // --- accessors, useful for diagnostics / reuse -------------------
    const FactorType& matrixL(IndexType k) const { return m_blockL.at(k); }
    const RealVector& scaling(IndexType k) const { return m_blockScale.at(k); }
    const PermutationType& permutation(IndexType k) const { return m_blockPerm.at(k); }
    IndexType nk() const { return m_nk; }
    IndexType nOrb() const { return m_norb; }
    IndexType size() const { return m_n; }

private:
    // Pull out the k-th n_orb x n_orb diagonal block of S as its own
    // sparse matrix. Assumes S is block-diagonal in the nk x n_orb
    // layout described above (no coupling between k-points); any
    // nonzero found outside the block is silently ignored (it shouldn't
    // exist for a correctly built supercell S -- extend this with an
    // assert/throw if you want strict validation of that assumption).
    SparseBlock extractBlock(const SparseMatS& S, IndexType k) const
    {
        const IndexType offset = k * m_norb;
        std::vector<Eigen::Triplet<Scalar, IndexType>> trips;
        trips.reserve(static_cast<size_t>(m_norb) * 8); // heuristic

        for (IndexType r = 0; r < m_norb; ++r)
        {
            const IndexType globalRow = offset + r;
            for (typename SparseMatS::InnerIterator it(S, globalRow); it; ++it)
            {
                const IndexType globalCol = it.col();
                if (globalCol >= offset && globalCol < offset + m_norb)
                    trips.emplace_back(r, globalCol - offset, it.value());
            }
        }

        SparseBlock Sk(m_norb, m_norb);
        Sk.setFromTriplets(trips.begin(), trips.end());
        Sk.makeCompressed();
        return Sk;
    }

    IndexType m_nk    = 0;
    IndexType m_norb = 0;
    IndexType m_n     = 0;
    bool m_computed  = false;

    std::vector<FactorType>        m_blockL;
    std::vector<RealVector>        m_blockScale;
    std::vector<PermutationType>  m_blockPerm;
    std::vector<IndexType>         m_failedBlocks;
};



#endif
