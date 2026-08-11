// chebyshev_sinvsqrt_preconditioner.hpp
//
// Approximates S^{-1/2} * vec via a Chebyshev polynomial expansion of the
// scalar function f(x) = x^{-1/2} on S's spectrum, evaluated block-by-block
// for the nk x n_orb block-diagonal DFT supercell overlap matrix S.
//
// S^{-1/2} shows up e.g. in Loewdin symmetric orthogonalization
// (H' = S^{-1/2} H S^{-1/2}), so having a matrix-free way to apply it via
// pure sparse MVMs (no factorization, no dense diagonalization of the full
// S) is often exactly what you want at scale.
//
// Mechanically this mirrors ChebyshevSInvPreconditioner (which expands
// f(x)=1/x) -- same spectral-bound estimation, same three-term recursion,
// same adaptive truncation -- with only the target function and its
// Chebyshev coefficients swapped from 1/x to x^{-1/2}:
//
//   x^{-1/2}  ~=  c_0/2 + sum_{k=1}^{P} c_k T_k(t),   t = (x - center)/halfwidth
//
//   S_k^{-1/2} vec  ~=  c_0/2 * v_0 + sum_{m=1}^{P} c_m * v_m
//
// where v_0=vec, v_1 = S^_k v_0, v_{m+1} = 2 S^_k v_m - v_{m-1}, and
// S^_k = (S_k - center*I)/halfwidth is the rescaled operator mapping S_k's
// spectrum into [-1,1].
//
// Note on convergence vs. the S^{-1} expansion: x^{-1/2} is a MILDER
// singularity at x=0 than x^{-1} (it diverges more slowly), so for the
// same spectral bounds [a,b] and the same target accuracy, the required
// Chebyshev order is generally LOWER here than for the 1/x expansion --
// but it still grows as a -> 0 (i.e. with S_k's condition number), so
// poorly conditioned blocks still need a correspondingly higher order.
//
// Interface matches the other preconditioner classes:
//   ChebyshevSInvSqrtPreconditioner<IndexType> C(nk, n_orb);
//   C.compute(S);
//   Vector x = C.solve(b);     // ~= S^{-1/2} b

#pragma once

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename IndexType>
class ChebyshevSInvSqrtPreconditioner
{
public:
    using Scalar        = std::complex<double>;
    using RealScalar     = double;

    using SparseMatS    = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, IndexType>;
    using SparseBlock    = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, IndexType>;
    using DenseBlock      = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using Vector           = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    // nk:            number of k-points
    // n_orb:         orbitals per k-point block
    // maxPolyOrder:  hard cap on Chebyshev order (per block)
    // coeffTol:      relative tolerance for adaptively truncating the
    //                expansion once |c_m| < coeffTol * |c_0| for all
    //                remaining m (checked from the top down)
    // paddingFrac:   fractional safety margin added outside [lambda_min,
    //                lambda_max] when estimating each block's spectral
    //                bounds, so the mapped spectrum stays strictly inside
    //                (-1,1)
    ChebyshevSInvSqrtPreconditioner(IndexType nk, IndexType n_orb,
                                                          int maxPolyOrder = 200,
                                                          RealScalar coeffTol = 1e-10,
                                                          RealScalar paddingFrac = 0.05)
        : m_nk(nk), m_norb(n_orb), m_n(nk * n_orb),
            m_maxPolyOrder(maxPolyOrder), m_coeffTol(coeffTol), m_paddingFrac(paddingFrac)
    {
        if (nk <= 0 || n_orb <= 0)
            throw std::invalid_argument("ChebyshevSInvSqrtPreconditioner: nk and n_orb must be positive");
        if (maxPolyOrder < 1)
            throw std::invalid_argument("ChebyshevSInvSqrtPreconditioner: maxPolyOrder must be >= 1");

        m_blockS.resize(m_nk);
        m_center.resize(m_nk);
        m_halfwidth.resize(m_nk);
        m_coeffs.resize(m_nk);
        m_effectiveOrder.resize(m_nk, 0);
    }

    // ---------------------------------------------------------------
    // For each k-point block: estimate spectral bounds [a,b], build the
    // Chebyshev coefficients of x^{-1/2} on [a,b], and truncate adaptively.
    // ---------------------------------------------------------------
    void compute(const SparseMatS& S)
    {
        if (S.rows() != S.cols())
            throw std::invalid_argument("ChebyshevSInvSqrtPreconditioner::compute: S must be square");
        if (S.rows() != m_n)
            throw std::invalid_argument(
                "ChebyshevSInvSqrtPreconditioner::compute: S size (" + std::to_string(S.rows()) +
                ") does not match nk*n_orb (" + std::to_string(m_n) + ")");

        m_failedBlocks.clear();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            SparseBlock Sk = extractSparseBlock(S, k);

            // --- spectral bounds via a dense Hermitian eigensolve ---
            DenseBlock SkDense(Sk);
            Eigen::SelfAdjointEigenSolver<DenseBlock> es(SkDense, Eigen::EigenvaluesOnly);
            if (es.info() != Eigen::Success)
            {
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                { m_failedBlocks.push_back(k); }
                continue;
            }

            RealScalar lambdaMin = es.eigenvalues()(0);
            RealScalar lambdaMax = es.eigenvalues()(m_norb - 1);

            if (lambdaMin <= 0.0)
            {
                // x^{-1/2} has a singularity/is undefined here -- S must
                // be Hermitian positive definite for this expansion.
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                { m_failedBlocks.push_back(k); }
                continue;
            }

            RealScalar span    = lambdaMax - lambdaMin;
            RealScalar pad      = m_paddingFrac * std::max(span, lambdaMin * 1e-3);
            RealScalar a         = std::max(lambdaMin - pad, lambdaMin * 0.5); // never let a<=0
            RealScalar b         = lambdaMax + pad;

            RealScalar center     = 0.5 * (a + b);
            RealScalar halfwidth = 0.5 * (b - a);

            // --- Chebyshev coefficients of f(x) = x^{-1/2} on [a,b] ---
            // Chebyshev-Gauss nodes: theta_j = pi*(2j+1)/(2N), x_j = center + halfwidth*cos(theta_j)
            const int N = m_maxPolyOrder + 1;
            std::vector<RealScalar> coeffsFull(N, 0.0);
            {
                std::vector<RealScalar> fvals(N);
                std::vector<RealScalar> thetas(N);
                for (int j = 0; j < N; ++j)
                {
                    RealScalar theta = M_PI * (2.0 * j + 1.0) / (2.0 * N);
                    RealScalar xj      = center + halfwidth * std::cos(theta);
                    thetas[j]          = theta;
                    fvals[j]            = 1.0 / std::sqrt(xj);   // <-- the only substantive change
                }
                for (int m = 0; m < N; ++m)
                {
                    RealScalar sum = 0.0;
                    for (int j = 0; j < N; ++j)
                        sum += fvals[j] * std::cos(m * thetas[j]);
                    coeffsFull[m] = (2.0 / N) * sum;
                }
            }

            // --- adaptive truncation (same criterion as the 1/x version) ---
            RealScalar c0mag = std::abs(coeffsFull[0]);
            int effOrder = m_maxPolyOrder;
            for (int m = m_maxPolyOrder; m >= 1; --m)
            {
                if (std::abs(coeffsFull[m]) > m_coeffTol * c0mag)
                {
                    effOrder = m;
                    break;
                }
                effOrder = 0;
            }
            effOrder = std::max(effOrder, 1);

            m_blockS[k]         = std::move(Sk);
            m_center[k]         = center;
            m_halfwidth[k]      = halfwidth;
            m_coeffs[k].assign(coeffsFull.begin(), coeffsFull.begin() + effOrder + 1);
            m_effectiveOrder[k] = effOrder;
        }

        if (!m_failedBlocks.empty())
        {
            std::string msg = "ChebyshevSInvSqrtPreconditioner::compute: block(s) not usable "
                                                 "(non-PD or eigensolve failure) for k = ";
            for (size_t i = 0; i < m_failedBlocks.size(); ++i)
                msg += std::to_string(m_failedBlocks[i]) + (i + 1 < m_failedBlocks.size() ? ", " : "");
            throw std::runtime_error(msg);
        }

        m_computed = true;
    }

    void compute(const SparseMatS* S)
    {
        if (!S) throw std::invalid_argument("ChebyshevSInvSqrtPreconditioner::compute: null S pointer");
        compute(*S);
    }

    bool isComputed() const { return m_computed; }

    // ---------------------------------------------------------------
    // solve(vec): apply the Chebyshev expansion of S^{-1/2} block-by-block.
    // (Named solve() to match the other preconditioner classes' interface,
    // even though this is a direct function application, not a linear solve.)
    // ---------------------------------------------------------------
    Vector solve(const Vector& vec) const
    {
        if (!m_computed)
            throw std::runtime_error("ChebyshevSInvSqrtPreconditioner::solve: compute() has not been called");
        if (vec.size() != m_n)
            throw std::invalid_argument("ChebyshevSInvSqrtPreconditioner::solve: size mismatch");

        Vector result(m_n);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (IndexType k = 0; k < m_nk; ++k)
        {
            const IndexType offset = k * m_norb;
            Vector vk = vec.segment(offset, m_norb);

            const auto& Sk           = m_blockS[k];
            const auto& c             = m_coeffs[k];
            const RealScalar center     = m_center[k];
            const RealScalar halfwidth = m_halfwidth[k];

            auto applyShat = [&](const Vector& v) -> Vector
            {
                // S^_k * v = (S_k*v - center*v) / halfwidth
                return (Sk * v - center * v) / halfwidth;
            };

            Vector v0 = vk;
            Vector out = (c[0] / 2.0) * v0;

            if (c.size() > 1)
            {
                Vector v1 = applyShat(v0);
                out += c[1] * v1;

                for (size_t m = 2; m < c.size(); ++m)
                {
                    Vector v2 = 2.0 * applyShat(v1) - v0;
                    out += c[m] * v2;
                    v0 = v1;
                    v1 = v2;
                }
            }

            result.segment(offset, m_norb) = out;
        }

        return result; // ~= S^{-1/2} * vec, block-by-block
    }

    Vector operator()(const Vector& vec) const { return solve(vec); }

    // --- accessors, useful for diagnostics ----------------------------
    int effectiveOrder(IndexType k) const { return m_effectiveOrder.at(k); }
    RealScalar center(IndexType k) const { return m_center.at(k); }
    RealScalar halfwidth(IndexType k) const { return m_halfwidth.at(k); }
    IndexType nk() const { return m_nk; }
    IndexType nOrb() const { return m_norb; }
    IndexType size() const { return m_n; }

private:
    SparseBlock extractSparseBlock(const SparseMatS& S, IndexType k) const
    {
        const IndexType offset = k * m_norb;
        std::vector<Eigen::Triplet<Scalar, IndexType>> trips;
        trips.reserve(static_cast<size_t>(m_norb) * 8);

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

    int         m_maxPolyOrder;
    RealScalar m_coeffTol;
    RealScalar m_paddingFrac;

    std::vector<SparseBlock>              m_blockS;
    std::vector<RealScalar>               m_center;
    std::vector<RealScalar>               m_halfwidth;
    std::vector<std::vector<RealScalar>> m_coeffs;
    std::vector<int>                       m_effectiveOrder;
    std::vector<IndexType>                m_failedBlocks;
};
