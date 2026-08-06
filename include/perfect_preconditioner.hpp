#ifndef PERFECT_PRECONDITIONER_HPP
#define PERFECT_PRECONDITIONER_HPP

// perfect_preconditioner.hpp
//
// Reference/ground-truth "preconditioner" for testing: for each k-point
// block S_k, computes the EXACT dense inverse S_k^{-1} (no incomplete
// factorization, no sparsity truncation, no stabilizing shift), and
// solve(b) returns the exact block-wise S^{-1} * b.
//
// This is not meant for production use on large n_orb (dense inverse is
// O(n_orb^3) to build and O(n_orb^2) per solve, and it materializes a
// full dense n_orb x n_orb matrix per k-point) -- it exists purely as a
// correctness reference to validate IschoPreconditioner against:
//
//   PerfectPreconditioner<IndexType> P(nk, n_orb);
//   P.compute(S);
//   Vector x_exact = P.solve(b);          // == S^{-1} b, to machine precision
//
//   IschoPreconditioner<IndexType> M(nk, n_orb);
//   M.compute(S);
//   Vector x_ischo = M.solve(b);          // approximate
//
//   (x_exact - x_ischo).norm() / x_exact.norm()   // should be small

#pragma once

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename IndexType>
class PerfectPreconditioner
{
public:
    using Scalar      = std::complex<double>;
    using RealScalar   = double;

    // Matches the layout used for S/H in the caller's code.
    using SparseMatS  = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, IndexType>;

    using Vector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using DenseBlock    = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // nk: number of k-points, n_orb: orbitals per k-point block.
    PerfectPreconditioner(IndexType nk, IndexType n_orb)
        : m_nk(nk), m_norb(n_orb), m_n(nk * n_orb)
    {
        if (nk <= 0 || n_orb <= 0)
            throw std::invalid_argument("PerfectPreconditioner: nk and n_orb must be positive");
        m_blockInv.resize(m_nk);
    }

    // ---------------------------------------------------------------
    // Compute the exact dense inverse of each n_orb x n_orb diagonal
    // block of S, independently.
    // ---------------------------------------------------------------
    void compute(const SparseMatS& S)
    {
        if (S.rows() != S.cols())
            throw std::invalid_argument("PerfectPreconditioner::compute: S must be square");
        if (S.rows() != m_n)
            throw std::invalid_argument(
                "PerfectPreconditioner::compute: S size (" + std::to_string(S.rows()) +
                ") does not match nk*n_orb (" + std::to_string(m_n) + ")");

        m_failedBlocks.clear();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            DenseBlock Sk = extractDenseBlock(S, k);

            // Exact Cholesky (not incomplete) -> exact inverse via solving
            // against the identity. Using LLT is both a correctness check
            // (Sk must be genuinely Hermitian positive definite, no
            // stabilizing shift involved) and the numerically preferred
            // way to invert an SPD/Hermitian-PD matrix.
            Eigen::LLT<DenseBlock> llt(Sk);
            if (llt.info() != Eigen::Success)
            {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    m_failedBlocks.push_back(k);
                }
                continue;
            }

            m_blockInv[k] = llt.solve(DenseBlock::Identity(m_norb, m_norb));
        }

        if (!m_failedBlocks.empty())
        {
            std::string msg = "PerfectPreconditioner::compute: block(s) not Hermitian "
                                                 "positive definite (exact Cholesky failed) for k = ";
            for (size_t i = 0; i < m_failedBlocks.size(); ++i)
                msg += std::to_string(m_failedBlocks[i]) + (i + 1 < m_failedBlocks.size() ? ", " : "");
            throw std::runtime_error(msg);
        }

        m_computed = true;
    }

    void compute(const SparseMatS* S)
    {
        if (!S) throw std::invalid_argument("PerfectPreconditioner::compute: null S pointer");
        compute(*S);
    }

    bool isComputed() const { return m_computed; }

    // ---------------------------------------------------------------
    // solve(vec) returns the EXACT S^{-1} * vec, block by block:
    //   result_k = S_k^{-1} * vec_k
    // ---------------------------------------------------------------
    Vector solve(const Vector& vec) const
    {
        if (!m_computed)
            throw std::runtime_error("PerfectPreconditioner::solve: compute() has not been called");
        if (vec.size() != m_n)
            throw std::invalid_argument("PerfectPreconditioner::solve: size mismatch");

        Vector result(m_n);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            const IndexType offset = k * m_norb;
            result.segment(offset, m_norb) = m_blockInv[k] * vec.segment(offset, m_norb);
        }

        return result; // == S^{-1} * vec, exactly (to machine precision), block-by-block
    }

    Vector operator()(const Vector& vec) const { return solve(vec); }

    // --- accessors ----------------------------------------------------
    const DenseBlock& blockInverse(IndexType k) const { return m_blockInv.at(k); }
    IndexType nk() const { return m_nk; }
    IndexType nOrb() const { return m_norb; }
    IndexType size() const { return m_n; }

private:
    // Same block-extraction logic as IschoPreconditioner, but materialized
    // as a dense n_orb x n_orb matrix instead of a sparse one.
    DenseBlock extractDenseBlock(const SparseMatS& S, IndexType k) const
    {
        const IndexType offset = k * m_norb;
        DenseBlock Sk = DenseBlock::Zero(m_norb, m_norb);

        for (IndexType r = 0; r < m_norb; ++r)
        {
            const IndexType globalRow = offset + r;
            for (typename SparseMatS::InnerIterator it(S, globalRow); it; ++it)
            {
                const IndexType globalCol = it.col();
                if (globalCol >= offset && globalCol < offset + m_norb)
                    Sk(r, globalCol - offset) = it.value();
            }
        }
        return Sk;
    }

    IndexType m_nk    = 0;
    IndexType m_norb = 0;
    IndexType m_n     = 0;
    bool m_computed  = false;

    std::vector<DenseBlock>  m_blockInv;
    std::vector<IndexType>   m_failedBlocks;
};

#endif
