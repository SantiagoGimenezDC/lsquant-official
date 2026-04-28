// Used for OPENMP functions
#include "chebyshev_solver_Graphene_NNN.hpp"
#include "chebyshev_solver.hpp"

#include <eigen3/Eigen/Core>
#include "Graphene_NNN.hpp"
#include <boost/math/special_functions/bessel.hpp>

using namespace chebyshev;





int chebyshev::SpectralMoments_Graphene_NNN( SparseMatrixType &OP,  chebyshev::Moments1D_Graphene_NNN &chebMoms, qstates::generator& gen )
{
	const auto Dim = chebMoms.SystemSize();
	const auto NumMoms = chebMoms.HighestMomentNumber();

	gen.SystemSize(Dim);
	while( gen.getQuantumState() )
	{		
		auto Phi = gen.State();
		//Set the evolved vector as initial vector of the chebyshev iterations
		if (OP.isIdentity() )
			chebMoms.SetInitVectors( Phi );
		else
			chebMoms.SetInitVectors( OP,Phi );
			
		for(int m = 0 ; m < NumMoms ; m++ )
		{
			double scal=2.0/gen.NumberOfStates();
			if( m==0) scal*=0.5;
			chebMoms(m) += scal*linalg::vdot( Phi, chebMoms.Chebyshev0() ) ;
			chebMoms.Iterate();
		}
	}
	return 0;
};



int chebyshev::MeanSquareDisplacement_Graphene_NNN_2(Graphene_NNN &H, int nMom_DOS, int nT, double tstep, qstates::generator& gen  )
{
  // ── 1. Get Hamiltonian and simulation parameters ──────────────────────────

    const int    N       = H.natoms;//.SystemSize();
    std::vector<cdouble> c = calculate_cn_bessel(tstep, H.a, H.b);

    
    int    nMom_tEvol    = c.size();     // nrecurs in Fortran
   

    // ── 2. Compute Chebyshev coefficients for U(T) ───────────────────────────
    // Fortran: CALL calculate_cn_bessel(tstep*1e-15*a*teV/hbar, a, b, npol, c)
    // The dimensionless time argument is  T_bar = tstep / hbar = 1/eV, where tstep is in femtoseconds (hbar is in eV . fs)
    // (H.a already absorbs teV since Adimensionalize was called with a*teV)
    

    // npol is determined by when J_n(T_bar) becomes negligibly small.
    // A safe upper bound: npol ~ T_bar + 10*cbrt(T_bar) (standard KPM rule).
    // Here we let nMom serve as npol, consistent with the Fortran usage.
    
    
    std::cout << "\nComputed " << nMom_tEvol
              << " Chebyshev coefficients for timestep = " << tstep << "\n";
    
    
    // ── 3. Allocate wave-function vectors ─────────────────────────────────────
    std::vector<cdouble> psi(N),   xupsi(N, cdouble{0,0}),
                                   yupsi(N, cdouble{0,0});

    // ── 4. Initialise random state ────────────────────────────────────────────
    // gen.random_state fills psi with a normalised random vector.

    //Initialize the Random Phase vector used for the Trace approximation
    gen.SystemSize(N);	
    gen.getQuantumState();
    psi = gen.State();


    auto mu_0 = chebyshev::MomentosDelta(H, psi.data(), nMom_DOS);

    // ── 5. Write moments to file ─────────────────────────────────────────────
    // Filename: mu_01.txt  
    std::ofstream f("mu_01.txt");
    if (!f) throw std::runtime_error(std::string("Cannot open mu_01.txt") );

    // Moments (1-based index in file to match Fortran output)
    for (int i = 0; i < nMom_DOS; ++i) {
        f << (i+1)
          << " " << std::real(mu_0[i]) << " " << std::imag(mu_0[i])
	  << "\n";
    }

    f.close();
    std::cout << "  Written mu_01.txt   \n";


    
      
    double inorm2 = 0.0;
    for (int i = 0; i < N; ++i) inorm2 += std::norm(psi[i]);
    std::cout << "\n  initial psi |U(t)|  = " << std::sqrt(inorm2) << "\n";

		
    // ── 6. Time-evolution loop ────────────────────────────────────────────────
    for (int it = 1; it <= nT; ++it) {
        std::cout << "\n\nTime step = " << it << "\n";

        // Evolve psi, xupsi, yupsi by one step T:
        //   psi   →  U(T)|psi>
        //   xupsi →  U(T)|xupsi> + [X, U(T)]|psi>
        //   yupsi →  U(T)|yupsi> + [Y, U(T)]|psi>
        evolution(H, c, psi, xupsi, yupsi);

        // Compute Chebyshev moments of ||xupsi||² and ||yupsi||² and write
        // them to  muxy_NNNNN.txt  (KPM input for post-processing dX²/dY²)
        get_dXY2(H, xupsi.data(), yupsi.data(), nMom_DOS, it);
    }
  return 0;
}





inline std::vector<cdouble> chebyshev::MomentosDelta(Graphene_NNN&  H,
                                          const cdouble*        psi,
                                          int                   nMom)
{
    const int N = H.natoms;

    std::vector<cdouble> mu(nMom + 1, cdouble{0.0, 0.0});
    std::vector<cdouble> r_n(N), r_nm1(N);

    // ── mu[0] = <psi|psi> = 1 ────────────────────────────────────────────────
    mu[0] = cdouble{1.0, 0.0};

    // ── Build r_nm1 = |psi>,  r_n = H_bar|psi> ───────────────────────────────
    // and collect mu[1] = <psi|H_bar|psi>,  mu[2] = 2<r_n|r_n> - mu[0]
    cdouble suma{0.0, 0.0}, sumb{0.0, 0.0};

    for (int i = 0; i < N; ++i) r_nm1[i] = psi[i];

    H.H_ket(r_n.data(), r_nm1.data());

    for (int i = 0; i < N; ++i) {
        suma += r_n[i] * std::conj(psi[i]);           // <psi|H_bar|psi>
        sumb += 2.0 * r_n[i] * std::conj(r_n[i]);    // 2<r_n|r_n>
    }

    mu[1] = suma;
    mu[2] = sumb - mu[0];

    // ── Main recursion: n = 3 .. nMom/2+1  (0-based: n-1 = 2 .. nMom/2) ─────
    for (int n = 3; n <= nMom / 2 + 1; ++n) {
        // One Chebyshev step + moment accumulation
        H.mv_2mu(r_n.data(), r_nm1.data(), suma, sumb);

        mu[2*n - 3] = suma - mu[1];
        mu[2*n - 2] = sumb - mu[0];

        // Update: swap r_n <-> r_nm1
        // After mv_2mu: r_nm1 = 2*H_bar*r_n_old - r_nm1_old  (new T_{n+1})
        //               r_n   = old r_n  (still T_n, unchanged by mv_2mu)
        std::swap(r_n, r_nm1);
    }

    return mu;
}







// ─────────────────────────────────────────────────────────────────────────────
// get_dXY2
//
// Computes the mean-square displacement along x and y by running the
// Chebyshev recursion on the commutator vectors xupsi and yupsi.
//
// Workflow:
//   1. Normalise xupsi and yupsi (save norms for later restoration).
//   2. Run MomentosDelta on each to get mux and muy.
//   3. Restore xupsi and yupsi to their original (un-normalised) values.
//   4. Write norms + moments to file  muxy_NNNNN.txt
//
// Arguments:
//   H      : Graphene_NNN (adimensionalised)
//   xupsi  : [X, U(T)]|psi>  vector (modified in place, restored on return)
//   yupsi  : [Y, U(T)]|psi>  vector (modified in place, restored on return)
//   nMom   : number of Chebyshev moments
//   it     : time-step index (used for filename)
// ─────────────────────────────────────────────────────────────────────────────


inline void chebyshev::get_dXY2(Graphene_NNN& H,
                     cdouble*            xupsi,
                     cdouble*            yupsi,
                     int                 nMom,
                     int                 it)
{
    const int N = H.natoms;

    // ── 1. Compute norms and normalise ───────────────────────────────────────
    double normdx = 0.0, normdy = 0.0;
    for (int i = 0; i < N; ++i) {
        normdx += std::norm(xupsi[i]);
        normdy += std::norm(yupsi[i]);
    }
    normdx = std::sqrt(normdx);
    normdy = std::sqrt(normdy);

    for (int i = 0; i < N; ++i) {
        xupsi[i] /= normdx;
        yupsi[i] /= normdy;
    }

    // ── 2. Run Chebyshev recursion ───────────────────────────────────────────
    std::cout << "\nRunning Chebyshev recursion (dX2) ...\n";

    auto mux = chebyshev::MomentosDelta(H, xupsi, nMom);
    auto muy = chebyshev::MomentosDelta(H, yupsi, nMom);

    std::cout << "...done\n\n";

    // ── 3. Restore original (un-normalised) vectors ──────────────────────────
    for (int i = 0; i < N; ++i) {
        xupsi[i] *= normdx;
        yupsi[i] *= normdy;
    }



    
    // ── 4. Write moments to file ─────────────────────────────────────────────
    // Filename: muxy_NNNNN.txt  (zero-padded 5-digit time index)
    char buf[32];
    std::snprintf(buf, sizeof(buf), "muxy_%05d.txt", it);
    std::ofstream f(buf);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + buf);

    // Header: norms (mirrors Fortran: WRITE(106,*) "#",normdx,normdy)
    f << "# " << normdx << " " << normdy << "\n";

    // Moments (1-based index in file to match Fortran output)
    for (int i = 0; i < nMom; ++i) {
        f << (i+1)
          << " " << std::real(mux[i]) << " " << std::imag(mux[i])
          << " " << std::real(muy[i]) << " " << std::imag(muy[i])
          << "\n";
    }

    f.close();
    std::cout << "  Written " << buf << "\n";
}







// ─────────────────────────────────────────────────────────────────────────────
// evolution
//
// Evolves the wave packet one time step and tracks position via commutators.
// All three vectors are updated in-place:
//
//   psi    →  U(T)|psi>
//   xupsi  →  U(T)|xupsi> + [X, U(T)]|psi>
//   yupsi  →  U(T)|yupsi> + [Y, U(T)]|psi>
//
// PART 1 — U(T)|xupsi>, U(T)|yupsi>:
//   Standard Chebyshev sum using H.H_ket() and H.update_cheb().
//
// PART 2 — [X/Y, U(T)]|psi> and U(T)|psi>:
//   Coupled alpha/beta recursion:
//     alpha_{n+1} = 2*H_bar*alpha_n - alpha_{n-1}
//     beta_{n+1}  = 2*H_bar*beta_n + 2*[X,H_bar]*alpha_n - beta_{n-1}
//   where [X,H_bar]*alpha_n is computed cleanly via H.XYH_ket().
// ─────────────────────────────────────────────────────────────────────────────


void chebyshev::evolution(Graphene_NNN&         H,
               std::vector<cdouble>& c,
               std::vector<cdouble>&                    psi,
               std::vector<cdouble>&                    xupsi,
               std::vector<cdouble>&                    yupsi)
{
    const int N    = H.natoms;
    const int npol = static_cast<int>(c.size());

    

    std::vector<cdouble> pnpsi(N),pn1psi(N), vel_tmp(N),
                         xpnpsi(N),xpn1psi = xupsi,
                         ypnpsi(N),ypn1psi = yupsi;

    
    cdouble xtemp,ytemp,ctemp, cnum, xcnum, ycnum; 

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 1:  U(T)|xupsi>  and  U(T)|yupsi>
    // ═══════════════════════════════════════════════════════════════════════════
    {
      
	H.H_ket(xpnpsi.data(), xpn1psi.data());
	H.H_ket(ypnpsi.data(), ypn1psi.data());

  // n=0:  T_0 = I
        for (int i = 0; i < N; ++i) {
            xupsi[i] = c[0] * xupsi[i];
            yupsi[i] = c[0] * yupsi[i];
        }
	

	
        for (int n = 1; n < npol; ++n) {
            for (int i = 0; i < N; ++i) {
                xupsi[i] += c[n] * xpnpsi[i];
                yupsi[i] += c[n] * ypnpsi[i];
            }

	    // T_{n+1} = 2*H_bar*T_n - T_{n-1}
	    H.Multiply(2.0, xpnpsi.data(), -1.0, xpn1psi.data());
	    H.Multiply(2.0, ypnpsi.data(), -1.0, ypn1psi.data());
	    
            std::swap(xpnpsi, xpn1psi);
            std::swap(ypnpsi, ypn1psi);

        }
    }



    

    // ═══════════════════════════════════════════════════════════════════════════
    // PART 2:  U(T)|psi>  and  [X/Y, U(T)]|psi>
    // ═══════════════════════════════════════════════════════════════════════════

    // Initialise:
    //   app = alpha_{n-1} = T_0|psi> = |psi>
    //   ap  = alpha_n     = T_1|psi> = H_bar|psi>
    //   xbpp = beta_x_{n-1} = 0
    //   xbp  = beta_x_n     = [X,H_bar]|psi>   (first non-trivial beta)
    pn1psi = psi;
    
    std::fill(xpn1psi.begin(), xpn1psi.end(), cdouble{0,0});
    std::fill(ypn1psi.begin(), ypn1psi.end(), cdouble{0,0});
    

    H.vx_ket(xpnpsi.data(), pn1psi.data());
    H.vy_ket(ypnpsi.data(), pn1psi.data());   

    H.H_ket(pnpsi.data(), pn1psi.data());


    for (int i = 0; i < N; ++i) 
        psi[i]    = c[0] * psi[i];


    //2(H*xpnpsi+vx*pnpsi)-xpn1psi
    for (int n = 1; n < npol; ++n) {
        // ap  = T_n|psi>   xbp = beta_x_n
        for (int i = 0; i < N; ++i) {
            psi[i]   += c[n] * pnpsi[i];
            xupsi[i] += c[n] * xpnpsi[i];
            yupsi[i] += c[n] * ypnpsi[i];
        }

	H.Multiply(2.0, pnpsi.data(), -1.0, pn1psi.data());
	H.Multiply(2.0, xpnpsi.data(), -1.0, xpn1psi.data());	
	H.Multiply(2.0, ypnpsi.data(), -1.0, ypn1psi.data());

	
	H.vx_ket(vel_tmp.data(), pnpsi.data());
	for (int i = 0; i < N; ++i) 
           xpn1psi[i]   +=  2.0 * vel_tmp[i];

	H.vy_ket(vel_tmp.data(), pnpsi.data());
	for (int i = 0; i < N; ++i) 
           ypn1psi[i]   +=  2.0 * vel_tmp[i];

	
	
 
	std::swap(pnpsi, pn1psi);
        std::swap(xpnpsi, xpn1psi);
        std::swap(ypnpsi, ypn1psi);

    }

    // Unitarity check

    double norm2=0;
    for (int i = 0; i < N; ++i) norm2 += std::norm(psi[i]);
    std::cout << "\n  |U(t)|  = " << std::sqrt(norm2) << "\n";

}


























