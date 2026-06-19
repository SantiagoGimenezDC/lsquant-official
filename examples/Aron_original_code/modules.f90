!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE constants

! Fundamental physical constants
DOUBLE PRECISION, PARAMETER :: pi      = 3.1415926536d0
DOUBLE PRECISION, PARAMETER :: hPlanck = 6.62606957d-34
DOUBLE PRECISION, PARAMETER :: q       = 1.602176565d-19
DOUBLE PRECISION, PARAMETER :: phi0    = hPlanck/q
DOUBLE PRECISION, PARAMETER :: hbar    = hPlanck/(2.0d0*pi)/q

END MODULE constants
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE simulation


! Parameters from input file
INTEGER          :: nn2, nrecurs, nt, nrow,ncol, iseed
DOUBLE PRECISION :: tstep, Emin,Emax,dE,eta, W, peh,Weh,leh, &
                    Delta,VI_A,VI_B,VR,VRaniso,VRin,VPIA_A,VPIA_B, &
                    teV,teV2,taniso
DOUBLE PRECISION :: a,b
CHARACTER        :: axis

! Namelist for reading input file
NAMELIST /params/ nn2,     &  ! Flag telling whether or not to consider 2nd-nearest neighbors
                  nrecurs, &  ! Number of Chebyshev iterations
                  tstep,   &  ! Time step, in units of fs
                  nt,      &  ! Total number of time steps in wave packet evolution
                  Emin,    &  ! Minimum energy for DOS(E) and dX^2(E,t)
                  Emax,    &  ! Maximum energy for DOS(E) and dX^2(E,t)
                  dE,      &  ! Energy resolution of DOS(E) and dX^2(E,t)
                  eta,     &  ! Energy smearing
                  nrow,    &  ! Sample size multiplication: number of rows
                  ncol,    &  ! Sample size multiplication: number of columns
                  W,       &  ! Anderson disorder
                  peh,     &  ! e-h puddle density
                  Weh,     &  ! e-h puddle height
                  leh,     &  ! e-h puddle width
                  Delta,   &  ! Staggered sublattice potential
                  VI_A,    &  ! Intrinsic spin-orbit coupling strength on sublattice A
                  VI_B,    &  ! Intrinsic spin-orbit coupling strength on sublattice B
                  VR,      &  ! Rashba spin-orbit coupling strength
                  VRaniso, &  ! Anisotropy of Rashba SOC (1=isotropic, 0=perfectly anisotropic)
                  VRin,    &  ! Rashba from an in-plane field
                  VPIA_A,  &  ! PIA spin-orbit coupling strength on A sublattice
                  VPIA_B,  &  ! PIA spin-orbit coupling strength on B sublattice
                  a,       &  ! Width of the Hamiltonian spectrum
                  b,       &  ! Center of the Hamiltonian spectrum
                  teV,     &  ! Nearest neighbor hopping energy
                  teV2,    &  ! Second-nearest neighbor hopping energy
                  taniso,  &  ! Anisotropy in nearest-neighbor hopping (1=isotropic, 0=perfectly anisotropic)
                  iseed,   &  ! Seed for random number generator (0 = use the system clock)
                  axis        ! Axis along which to calculate spin


END MODULE simulation
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
