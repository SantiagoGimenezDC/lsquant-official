#ifndef GRAPHENE_NNN
#define GRAPHENE_NNN


#include <eigen3/Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

// ─── Type aliases ─────────────────────────────────────────────────────────────
using cdouble = std::complex<double>;
using CVec    = Eigen::VectorXcd;
using RVec    = Eigen::VectorXd;
using CMat    = Eigen::MatrixXcd;
using IMat    = Eigen::MatrixXi;
using IVec    = Eigen::VectorXi;

// ─────────────────────────────────────────────────────────────────────────────
// Params: all simulation parameters (defaults match the provided params.txt)
// ─────────────────────────────────────────────────────────────────────────────
struct Params {
    // Lattice
    int    nrow    = 100;
    int    ncol    = 100;

    // Hoppings
    double teV     =  2.7;      // nearest-neighbour hopping (eV)
    double teV2    =  0.0;      // next-nearest-neighbour hopping (eV)
    double taniso  =  1.0;      // hopping anisotropy factor

    // Anderson disorder
    double W       =  0.0;      // amplitude (eV)

    // Electron-hole puddles
    double peh     =  0.04e-2;  // spatial density
    double Weh     =  0.05;     // potential amplitude (eV)
    double leh     = 100.0;     // spatial extent (Angstrom)

    // Spin-orbit / gap parameters
    double Delta   =  0.0;      // sublattice gap (eV)
    double VI_A    =  0.0;      // Kane-Mele SOC sublattice A (eV)
    double VI_B    =  0.0;      // Kane-Mele SOC sublattice B (eV)
    double VR      =  0.0;      // Rashba SOC (eV)
    double VRaniso =  0.0;      // Rashba anisotropy factor
    double VRin    =  0.0;      // in-plane Rashba along bond 3-4 (eV)
    double VPIA_A  =  0.0;      // pseudo-spin inversion sublattice A (eV)
    double VPIA_B  =  0.0;      // pseudo-spin inversion sublattice B (eV)

    // Random seed
    int    iseed   =  3;
};

// ─────────────────────────────────────────────────────────────────────────────
// UnitCell: hardcoded 8-atom graphene supercell
//
// Lattice vectors A1=(4.26, 2.45951), A2=(-4.26, 2.45951)  [Angstrom]
// Bond length acc = 1.42 Å,  NN search cutoff = 1.2 * acc
// 8 atoms per unit cell arranged as two graphene hexagons
// ─────────────────────────────────────────────────────────────────────────────
struct UnitCell {
    double A1x =  4.26,               A1y =  2.45951214674781;
    double A2x = -4.26,               A2y =  2.45951214674781;
    double acc      = 1.42;    // C-C bond length (Angstrom)
    double ncutoff  = 1.2;     // NN cutoff in units of acc
    int    nuc      = 8;       // atoms in unit cell

    // Fractional positions within the unit cell (Angstrom, absolute)
    // Index maps to i%8 sublattice for MOD checks in geom_hamilt
    std::vector<double> x0 = { -2.84, -1.42, -0.71,  0.71,
                                -0.71,  0.71,  1.42,  2.84 };
    std::vector<double> y0 = {  0.0,   0.0,   1.2297560733739,  1.2297560733739,
                                -1.2297560733739, -1.2297560733739,  0.0,  0.0  };
};



// ─────────────────────────────────────────────────────────────────────────────
// Graphene_NNN
//
// Builds a spinful tight-binding Hamiltonian from a unitcell.txt file and a
// Params struct.  The full system has N = 2*natoms_orb sites:
//   sites 0 .. natoms_orb-1             = spin-up
//   sites natoms_orb .. N-1             = spin-down
//
// All neighbour indices (nn, nnn) are 0-based.
//
// Hopping matrix column layout for row i (spin-up atom):
//   s (i, 0..near_orb-1)               up-up   diagonal hopping
//   s (i, near_orb..2*near_orb-1)      up-down Rashba hopping
// For row No+i (spin-down):
//   s (No+i, 0..near_orb-1)            down-down diagonal
//   s (No+i, near_orb..2*near_orb-1)   down-up  Rashba
// (NNN columns follow the same scheme; off-diagonal NNN = 0 in this model.)
//
// Public operations:
//   Hr_ket(ket)              returns H|ket>
//   Chebyshev_update(p, pp)  returns 2*H|p> - |pp>
// ─────────────────────────────────────────────────────────────────────────────
class Graphene_NNN
{
public:
    // ── Public Hamiltonian data ───────────────────────────────────────────────
    int   natoms;    // 2 * natoms_orb  (full spinful system)
    int   maxnn;     // 2 * maxnn_orb
    int   maxnnn;    // 2 * maxnnn_orb

    IVec  near_n;    // near_n(i)  : actual number of 1st neighbours of site i
    IVec  nnear_n;   // nnear_n(i) : actual number of 2nd neighbours of site i
    IMat  nn;        // nn(i,j)    : 0-based index of j-th 1st neighbour of i
    IMat  nnn;       // nnn(i,j)   : 0-based index of j-th 2nd neighbour of i
    RVec  e;         // on-site energies (eV)
    CMat  s;         // 1st-neighbour hopping amplitudes
    CMat  ss;        // 2nd-neighbour hopping amplitudes
    double a=10,     //Adimensionalizing constants. Halfwidth.
           b=0.0;    //Bandcenter


    int Natoms(){return natoms;};
    // ── Constructor: builds the full Hamiltonian from hardcoded unit cell ────
    Graphene_NNN(const Params& p = Params{}, const UnitCell& uc = UnitCell{})
    {
        build(uc, p);
    }


    // ── Set Hamiltonian adimensionalizing constants───────────────────────
    void Adimensionalize(double new_a, double new_b){a=new_a, b=new_b;};



  // ── H|ket> ───────────────────────────────────────────────────────────────
    void H_ket(cdouble* ket, const cdouble* p_ket)
    {
  
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < natoms; ++i) {
            cdouble val = e(i) * p_ket[i];
            for (int j = 0; j < near_n(i);  ++j) val += s(i,j)  * p_ket[nn(i,j)];
            for (int j = 0; j < nnear_n(i); ++j) val += ss(i,j) * p_ket[nnn(i,j)];
            ket[i] = (val - b * p_ket[i]) / a;
        }
    }

    // ── 2*H|p_ket> - |pp_ket>  (Chebyshev recursion step) ───────────────────
    void update_cheb(cdouble* ket,  cdouble* p_ket,  cdouble* pp_ket)
    {

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < natoms; ++i) {
            cdouble val = e(i) * p_ket[i];
            for (int j = 0; j < near_n(i);  ++j) val += s(i,j)  * p_ket[nn(i,j)];
            for (int j = 0; j < nnear_n(i); ++j) val += ss(i,j) * p_ket[nnn(i,j)];
	    ket[i] = 2.0 * (val - b * p_ket[i]) / a - pp_ket[i];
	    pp_ket[i] = p_ket[i];
        }
	
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < natoms; ++i) 
	  p_ket[i] = ket[i];
        

    }


  
private:
    using Cells3D = std::vector<std::vector<std::vector<int>>>;

    // ─────────────────────────────────────────────────────────────────────────
    // fix_dist_pbc
    // Minimum-image convention for a non-orthogonal 2D lattice.
    // A1, A2 are unit vectors; lA1, lA2 are their magnitudes.
    // ─────────────────────────────────────────────────────────────────────────
    static void fix_dist_pbc(double& dx, double& dy,
                             double A1x, double A1y,
                             double A2x, double A2y,
                             double lA1, double lA2)
    {
        double denom  = A1x*A2y - A1y*A2x;
        double p1 = -(A2x*dy - A2y*dx) / denom / lA1;
        double p2 =  (A1x*dy - A1y*dx) / denom / lA2;
        if (p1 >  0.5) { dx -= A1x*lA1; dy -= A1y*lA1; }
        if (p1 < -0.5) { dx += A1x*lA1; dy += A1y*lA1; }
        if (p2 >  0.5) { dx -= A2x*lA2; dy -= A2y*lA2; }
        if (p2 < -0.5) { dx += A2x*lA2; dy += A2y*lA2; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kane_Mele
    // Computes nu = (v1 x v2) for intrinsic SOC.  Only nu[2] (z) matters in 2D.
    // All site indices are 0-based orbital indices.
    // nsite == -1 => no common neighbour => nu = {0,0,0}.
    // ─────────────────────────────────────────────────────────────────────────
    static std::array<double,3> Kane_Mele(
        const RVec& x, const RVec& y,
        double A1x, double A1y, double A2x, double A2y, double lA1, double lA2,
        int site, int nnsite, int nsite)
    {
        std::array<double,3> nu = {};
        if (nsite < 0) return nu;

        double v1x = x(nsite)  - x(site),   v1y = y(nsite)  - y(site);
        double v2x = x(nnsite) - x(nsite),   v2y = y(nnsite) - y(nsite);
        fix_dist_pbc(v1x, v1y, A1x, A1y, A2x, A2y, lA1, lA2);
        fix_dist_pbc(v2x, v2y, A1x, A1y, A2x, A2y, lA1, lA2);

        double n1 = std::sqrt(v1x*v1x + v1y*v1y);
        double n2 = std::sqrt(v2x*v2x + v2y*v2y);
        v1x /= n1; v1y /= n1;
        v2x /= n2; v2y /= n2;

        nu[2] = v1x*v2y - v2x*v1y;   // z-component of cross product
        return nu;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // find_max_nn
    // ─────────────────────────────────────────────────────────────────────────
    static int find_max_nn(
        int No, int nrow, int ncol, int nuc,
        const RVec& x, const RVec& y, double acc, double ncutoff,
        const Cells3D& cells,
        double A1x, double A1y, double A2x, double A2y, double lA1, double lA2)
    {
        double cut2 = ncutoff * ncutoff * acc * acc;
        int mx = 0;
        for (int irow = 0; irow < nrow; ++irow)
        for (int icol = 0; icol < ncol; ++icol)
        for (int i = 0; i < nuc; ++i) {
            int n = i + nuc*(irow*ncol + icol);
            int cnt = 0;
            for (int ar = -2; ar <= 2; ++ar)
            for (int ac = -2; ac <= 2; ++ac) {
                int rm = (irow+ar+nrow)%nrow, cm = (icol+ac+ncol)%ncol;
                for (int j = 0; j < nuc; ++j) {
                    int m = cells[rm][cm][j];
                    if (m == n) continue;
                    double dx = x(n)-x(m), dy = y(n)-y(m);
                    fix_dist_pbc(dx, dy, A1x, A1y, A2x, A2y, lA1, lA2);
                    if (dx*dx+dy*dy < cut2) ++cnt;
                }
            }
            mx = std::max(mx, cnt);
        }
        return mx;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // find_nn
    // ─────────────────────────────────────────────────────────────────────────
    static void find_nn(
        int No, int maxnn_o, int nrow, int ncol, int nuc,
        const RVec& x, const RVec& y, double acc, double ncutoff,
        const Cells3D& cells,
        double A1x, double A1y, double A2x, double A2y, double lA1, double lA2,
        IMat& nn_o, IVec& near_o,
        Eigen::MatrixXd& dx_o, Eigen::MatrixXd& dy_o)
    {
        double cut2 = ncutoff*ncutoff*acc*acc;
        nn_o.setZero(No, maxnn_o); near_o.setZero(No);
        dx_o.setZero(No, maxnn_o); dy_o.setZero(No, maxnn_o);

        std::array<int,11> num = {};
        double davg = 0.0; int navg = 0;

        for (int irow = 0; irow < nrow; ++irow)
        for (int icol = 0; icol < ncol; ++icol)
        for (int i = 0; i < nuc; ++i) {
            int n = i + nuc*(irow*ncol + icol), cnt = 0;
            for (int ar = -2; ar <= 2; ++ar)
            for (int ac = -2; ac <= 2; ++ac) {
                int rm = (irow+ar+nrow)%nrow, cm = (icol+ac+ncol)%ncol;
                for (int j = 0; j < nuc; ++j) {
                    int m = cells[rm][cm][j];
                    if (m == n) continue;
                    double dx = x(n)-x(m), dy = y(n)-y(m);
                    fix_dist_pbc(dx, dy, A1x, A1y, A2x, A2y, lA1, lA2);
                    double d2 = dx*dx+dy*dy;
                    if (d2 < cut2) {
                        nn_o(n,cnt) = m; dx_o(n,cnt) = dx; dy_o(n,cnt) = dy;
                        davg += std::sqrt(d2); ++navg; ++cnt;
                    }
                }
            }
            near_o(n) = cnt;
            if (cnt <= 10) ++num[cnt];
        }
        std::cout << "\n  NUMBER OF NEAREST NEIGHBOR ATOMS:\n\n";
        for (int k=0;k<=10;++k) std::cout<<"  "<<k<<" neighbors: "<<num[k]<<"\n";
        int gt10=No; for(auto c:num) gt10-=c;
        std::cout<<" >10 neighbors: "<<gt10<<"\n";
        std::cout<<"\n  Average nn distance: "<<davg/navg<<"\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // find_max_nnn
    // ─────────────────────────────────────────────────────────────────────────
    static int find_max_nnn(int No, const IMat& nn_o, const IVec& near_o)
    {
        int mx = 0;
        for (int i = 0; i < No; ++i) {
            int cnt = 0;
            for (int j = 0; j < near_o(i); ++j) {
                int l = nn_o(i,j);
                for (int k = 0; k < near_o(l); ++k)
                    if (nn_o(l,k) != i) ++cnt;
            }
            mx = std::max(mx, cnt);
        }
        return mx;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // find_nnn
    // ncom(i,j) = 0-based common-neighbour index, or -1 if not found.
    // ─────────────────────────────────────────────────────────────────────────
    static void find_nnn(
        int No, const IMat& nn_o, const IVec& near_o, int maxnnn_o,
        const RVec& x, const RVec& y,
        double A1x, double A1y, double A2x, double A2y, double lA1, double lA2,
        IMat& nnn_o, IVec& nnear_o, IMat& ncom,
        Eigen::MatrixXd& ddx_o, Eigen::MatrixXd& ddy_o)
    {
        nnn_o.setZero(No, maxnnn_o); nnear_o.setZero(No);
        ncom.setConstant(No, maxnnn_o, -1);
        ddx_o.setZero(No, maxnnn_o); ddy_o.setZero(No, maxnnn_o);

        std::array<int,11> num = {};
        double davg = 0.0; int navg = 0;

        for (int i = 0; i < No; ++i) {
            int cnt = 0;
            for (int j = 0; j < near_o(i); ++j) {
                int l = nn_o(i,j);
                for (int k = 0; k < near_o(l); ++k) {
                    int m = nn_o(l,k);
                    if (m == i) continue;
                    bool dup = false;
                    for (int n = 0; n < cnt; ++n)
                        if (nnn_o(i,n)==m) { dup=true; break; }
                    if (!dup) {
                        nnn_o(i,cnt) = m;
                        double dx = x(i)-x(m), dy = y(i)-y(m);
                        fix_dist_pbc(dx, dy, A1x, A1y, A2x, A2y, lA1, lA2);
                        ddx_o(i,cnt) = dx; ddy_o(i,cnt) = dy;
                        davg += std::sqrt(dx*dx+dy*dy); ++navg; ++cnt;
                    }
                }
            }
            nnear_o(i) = cnt;
            if (cnt <= 10) ++num[cnt];
        }
        std::cout<<"\n  NUMBER OF NEXT-NEAREST NEIGHBOR ATOMS:\n\n";
        for (int k=0;k<=10;++k) std::cout<<"  "<<k<<" neighbors: "<<num[k]<<"\n";
        int gt10=No; for(auto c:num) gt10-=c;
        std::cout<<" >10 neighbors: "<<gt10<<"\n";
        std::cout<<"\n  Average nnn distance: "<<davg/navg<<"\n";

        // Common neighbour: atom shared between i and nnn[i][j]
        for (int i = 0; i < No; ++i)
        for (int j = 0; j < nnear_o(i); ++j) {
            int k = nnn_o(i,j);
            for (int ii = 0; ii < near_o(i);  ++ii)
            for (int jj = 0; jj < near_o(k); ++jj)
                if (nn_o(i,ii) == nn_o(k,jj))
                    ncom(i,j) = nn_o(i,ii);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // geom_hamilt
    //
    // Fills e (size N=2*No), s (N x 2*maxnn_o), ss (N x 2*maxnnn_o).
    // Uses the *orbital* nn_o/nnn_o arrays (0-based, size No).
    //
    // Key index translations from Fortran (1-based) to C++ (0-based):
    //   Sublattice A:  i%2==0   (was MOD(i,2)==1 in 1-based Fortran)
    //   Sublattice B:  i%2==1
    //   Bond 3-4:      (i%8==2 && nb%8==3) or (i%8==3 && nb%8==2)
    //                  (was MOD(i,8)==3 && MOD(nb,8)==4 etc. in 1-based Fortran)
    // ─────────────────────────────────────────────────────────────────────────
    static void geom_hamilt(
        int No, int maxnn_o, int maxnnn_o,
        const IMat& nn_o,  const IVec& near_o,
        const IMat& nnn_o, const IVec& nnear_o,
        const IMat& ncom,
        double acc,
        double A1x, double A1y, double A2x, double A2y, double lA1, double lA2,
        const RVec& x, const RVec& y,
        double teV, double teV2, double taniso,
        double W, int neh, const std::vector<int>& ieh, double Weh, double leh,
        double Delta, double VI_A, double VI_B,
        double VR, double VRaniso, double VRin, double VPIA_A, double VPIA_B,
        std::mt19937& rng,
        RVec& e_out, CMat& s_out, CMat& ss_out)
    {
        constexpr cdouble im{0.0, 1.0};
        int N = 2*No;
        e_out.setZero(N);
        s_out.setZero(N, 2*maxnn_o);
        ss_out.setZero(N, 2*maxnnn_o);

        std::uniform_real_distribution<double> uni(0.0, 1.0);

        // ── Main loop over orbital atoms ──────────────────────────────────
        for (int i = 0; i < No; ++i) {

            // On-site energy and sublattice SOC
            double VI, VPIA;
            if (i % 2 == 0) {                          // sublattice A
                e_out(i) = e_out(No+i) =  Delta;
                VI = VI_A; VPIA = VPIA_A;
            } else {                                    // sublattice B
                e_out(i) = e_out(No+i) = -Delta;
                VI = VI_B; VPIA = VPIA_B;
            }

            int nup = near_o(i);   // orbital NN count (= spin-up NN count)

            // ── Nearest-neighbour hoppings ────────────────────────────────
            for (int j = 0; j < nup; ++j) {
                int nb = nn_o(i,j);   // 0-based orbital neighbour

                double dx = x(nb)-x(i), dy = y(nb)-y(i);
                fix_dist_pbc(dx, dy, A1x, A1y, A2x, A2y, lA1, lA2);

                // Anisotropic bond 3-4 check (0-based)
                bool b_ij = (i%8==2 && nb%8==3);   // was MOD(i,8)==3 && MOD(nb,8)==4
                bool b_ji = (i%8==3 && nb%8==2);   // was MOD(i,8)==4 && MOD(nb,8)==3
                bool bond34 = b_ij || b_ji;

                // Diagonal hopping (up-up, down-down)
                cdouble t = bond34 ? cdouble(teV*(12.0-11.0*taniso), 0.0)
                                   : cdouble(teV*taniso, 0.0);
                cdouble t_up = t, t_dn = t;

                // In-plane Rashba correction on bond 3-4
                if (b_ij) { t_up += 4.0*im*VRin;  t_dn -= 4.0*im*VRin; }
                if (b_ji) { t_up -= 4.0*im*VRin;  t_dn += 4.0*im*VRin; }

                s_out(i,    j) = t_up;    // up   -> up
                s_out(No+i, j) = t_dn;   // down -> down

                // Off-diagonal Rashba hopping (spin flip)
                // Column nup+j corresponds to nn_full(i, nup+j) = nb + No
                double rfac = bond34 ? (12.0-11.0*VRaniso) : VRaniso;
                s_out(i,    nup+j) = -2.0/3.0*im*VR*(dy + im*dx)/acc * rfac;  // up -> down
                s_out(No+i, nup+j) = -2.0/3.0*im*VR*(dy - im*dx)/acc * rfac;  // down -> up
            }

            // ── Next-nearest-neighbour hoppings (Kane-Mele SOC) ──────────
            for (int jj = 0; jj < nnear_o(i); ++jj) {
                int nsite = ncom(i,jj);   // common neighbour, -1 if missing
                if (nsite < 0)
                    std::cerr << "WARNING: ncom not found for i=" << i << " jj=" << jj << "\n";

                auto nu = Kane_Mele(x, y, A1x, A1y, A2x, A2y, lA1, lA2,
                                    i, nnn_o(i,jj), nsite);

                // Diagonal spin blocks only (no off-diagonal NNN in this model)
                ss_out(i,    jj) = cdouble(teV2,0.0) + (2.0/9.0)*VI*nu[2]*im;
                ss_out(No+i, jj) = cdouble(teV2,0.0) - (2.0/9.0)*VI*nu[2]*im;
            }
        }

        // ── Anderson disorder (same realisation for both spins) ───────────
        for (int i = 0; i < No; ++i) {
            double v = W*2.0*(uni(rng)-0.5);
            e_out(i) += v;  e_out(No+i) += v;
        }

        // ── Electron-hole puddles ─────────────────────────────────────────
        for (int p = 0; p < neh; ++p) {
            double r = uni(rng);
            int ip = ieh[p];
            for (int i = 0; i < No; ++i) {
                double dx = x(i)-x(ip), dy = y(i)-y(ip);
                fix_dist_pbc(dx, dy, A1x, A1y, A2x, A2y, lA1, lA2);
                double v = Weh*2.0*(r-0.5)*std::exp(-0.5*(dx*dx+dy*dy)/(leh*leh));
                e_out(i) += v;  e_out(No+i) += v;
            }
        }

        std::cout << "\n  VI_A:    " << VI_A
                  << "\n  VI_B:    " << VI_B
                  << "\n  VR:      " << VR
                  << "\n  VRaniso: " << VRaniso
                  << "\n  VRin:    " << VRin
                  << "\n  VPIA_A:  " << VPIA_A
                  << "\n  VPIA_B:  " << VPIA_B << "\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // build: orchestrates reading unitcell.txt and constructing the Hamiltonian
    // ─────────────────────────────────────────────────────────────────────────
    void build(const UnitCell& uc, const Params& p)
    {
        // ── 1. Set up atom positions from UnitCell struct ────────────────────
        std::cout << "\nSetting up atom positions from built-in unit cell...\n\n";

        double A1x = uc.A1x, A1y = uc.A1y;
        double A2x = uc.A2x, A2y = uc.A2y;
        double acc = uc.acc, ncutoff = uc.ncutoff;
        int    nuc = uc.nuc;

        int No = nuc * p.nrow * p.ncol;
        std::cout << "  natoms (orbital): " << No << "\n\n";

        RVec x(No), y(No);
        x.setZero(); y.setZero();

        Cells3D cells(p.nrow, std::vector<std::vector<int>>(
                               p.ncol, std::vector<int>(nuc, 0)));

        for (int i = 0; i < nuc; ++i) {
            double xi = uc.x0[i], yi = uc.y0[i];
            for (int irow = 0; irow < p.nrow; ++irow)
            for (int icol = 0; icol < p.ncol; ++icol) {
                int idx = i + nuc*(irow*p.ncol + icol);
                x(idx) = xi + icol*A1x + irow*A2x;
                y(idx) = yi + icol*A1y + irow*A2y;
                cells[irow][icol][i] = idx;
            }
        }

        // Scale lattice vectors to full supercell, then convert to unit vectors
        A1x *= p.ncol; A1y *= p.ncol;
        A2x *= p.nrow; A2y *= p.nrow;
        double lA1 = std::sqrt(A1x*A1x+A1y*A1y); A1x /= lA1; A1y /= lA1;
        double lA2 = std::sqrt(A2x*A2x+A2y*A2y); A2x /= lA2; A2y /= lA2;

        // ── 2. Orbital nearest neighbours ────────────────────────────────────
        int maxnn_o = find_max_nn(No, p.nrow, p.ncol, nuc, x, y, acc, ncutoff,
                                  cells, A1x, A1y, A2x, A2y, lA1, lA2);
        std::cout << "\n  maxnn (orbital): " << maxnn_o << "\n";

        IMat nn_o; IVec near_o;
        Eigen::MatrixXd dx_o, dy_o;
        find_nn(No, maxnn_o, p.nrow, p.ncol, nuc, x, y, acc, ncutoff,
                cells, A1x, A1y, A2x, A2y, lA1, lA2,
                nn_o, near_o, dx_o, dy_o);

        // ── 3. Orbital next-nearest neighbours ───────────────────────────────
        int maxnnn_o = find_max_nnn(No, nn_o, near_o);
        std::cout << "\n  maxnnn (orbital): " << maxnnn_o << "\n";

        IMat nnn_o, ncom; IVec nnear_o;
        Eigen::MatrixXd ddx_o, ddy_o;
        find_nnn(No, nn_o, near_o, maxnnn_o, x, y,
                 A1x, A1y, A2x, A2y, lA1, lA2,
                 nnn_o, nnear_o, ncom, ddx_o, ddy_o);
        std::cout << "...done\n\n";

        // ── 4. Spin doubling ──────────────────────────────────────────────────
        // Sites 0..No-1 = spin-up, No..2No-1 = spin-down.
        // NN columns: [0..nup-1] diagonal, [nup..2nup-1] off-diagonal (Rashba).
        // NNN columns: same pattern but off-diagonal NNN hoppings are zero.
        std::cout << "Copying spin-up to spin-down...\n";
        int N = 2*No, mn = 2*maxnn_o, mn3 = 2*maxnnn_o;

        IVec near_f(N);  near_f.setZero();
        IVec nnear_f(N); nnear_f.setZero();
        IMat nn_f(N, mn);   nn_f.setZero();
        IMat nnn_f(N, mn3); nnn_f.setZero();

        for (int i = 0; i < No; ++i) {
            int nup = near_o(i), nnup = nnear_o(i);
            near_f(i) = near_f(No+i) = 2*nup;
            nnear_f(i) = nnear_f(No+i) = 2*nnup;

            for (int j = 0; j < nup; ++j) {
                int nb = nn_o(i,j);
                nn_f(i,    j)     = nb;        // up   -> up
                nn_f(i,    nup+j) = nb + No;   // up   -> down
                nn_f(No+i, j)     = nb + No;   // down -> down
                nn_f(No+i, nup+j) = nb;        // down -> up
            }
            for (int j = 0; j < nnup; ++j) {
                int nb = nnn_o(i,j);
                nnn_f(i,    j)      = nb;        // up   -> up
                nnn_f(i,    nnup+j) = nb + No;   // up   -> down (zero amplitude)
                nnn_f(No+i, j)      = nb + No;   // down -> down
                nnn_f(No+i, nnup+j) = nb;        // down -> up   (zero amplitude)
            }
        }
        std::cout << "...done\n\n";

        // ── 5. Electron-hole puddles ──────────────────────────────────────────
        std::cout << "\nFinding electron-hole puddles...\n";
        std::mt19937 rng_e(p.iseed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<int> ieh;
        ieh.reserve(static_cast<int>(No*p.peh)+10);
        for (int i = 0; i < No; ++i)
            if (uni(rng_e) <= p.peh) ieh.push_back(i);
        std::cout << "\n  Number of puddles: " << ieh.size() << "\n...done\n\n";

        // ── 6. Build Hamiltonian ──────────────────────────────────────────────
        // Re-seed with the same seed so disorder sequence matches Fortran.
        std::mt19937 rng_h(p.iseed);
        std::cout << "\nBuilding the Hamiltonian...\n\n";
        RVec e_tmp; CMat s_tmp, ss_tmp;
        geom_hamilt(No, maxnn_o, maxnnn_o,
                    nn_o, near_o, nnn_o, nnear_o, ncom,
                    acc, A1x, A1y, A2x, A2y, lA1, lA2, x, y,
                    p.teV, p.teV2, p.taniso, p.W,
                    (int)ieh.size(), ieh, p.Weh, p.leh,
                    p.Delta, p.VI_A, p.VI_B, p.VR, p.VRaniso, p.VRin,
                    p.VPIA_A, p.VPIA_B, rng_h,
                    e_tmp, s_tmp, ss_tmp);
        std::cout << "\n...done\n\n";

        // ── 7. Store in public class members ─────────────────────────────────
        natoms  = N;
        maxnn   = mn;
        maxnnn  = mn3;
        near_n  = near_f;
        nnear_n = nnear_f;
        nn      = nn_f;
        nnn     = nnn_f;
        e       = e_tmp;
        s       = s_tmp;
        ss      = ss_tmp;
    }
};


#endif //Graphene_NNN
