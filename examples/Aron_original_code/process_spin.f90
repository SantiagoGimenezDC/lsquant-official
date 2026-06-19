!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Program: process_spin                                                            !
! Description: calculates energy-dependent spin polarization vs. time              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM process_spin

IMPLICIT NONE

! Input file parameters
INTEGER                       :: nn2, nrecurs, npol, nt, nrow, ncol, iseed
DOUBLE PRECISION              :: tstep, Emin,Emax,dE,eta, W, peh,Weh,leh, &
                                 Delta,VI_A,VI_B,VR,VRaniso,VRin,VPIA_A,VPIA_B, a,b,teV,teV2
CHARACTER                     :: axis

! Number of energy bins
INTEGER                       :: nE

! Needed for length of time step
DOUBLE PRECISION              :: gamma0, hbar, q

! Lanczos recursion coefficients
DOUBLE PRECISION              :: normx,normy
DOUBLE PRECISION, ALLOCATABLE :: mu(:)
DOUBLE COMPLEX,   ALLOCATABLE :: muspin(:)

! Density of states and spin polarization
DOUBLE PRECISION, ALLOCATABLE :: dos(:)
DOUBLE COMPLEX,   ALLOCATABLE :: spin(:)

! Loop index variables
INTEGER                       :: i,j

! File I/O
INTEGER          :: itemp
DOUBLE PRECISION :: rtemp1,rtemp2
CHARACTER        :: ctemp
CHARACTER(5)     :: ci
CHARACTER(16)    :: file_muspin
CHARACTER(13)    :: file_dxy2

! Namelist for reading input file
NAMELIST /params/ nn2,     &  ! Flag telling whether or not to consider 2nd-nearest neighbors
                  nrecurs, &  ! Number of Lanczos iterations
                  tstep,   &  ! Time step, in units of fs
                  nt,      &  ! Total number of time steps in wave packet evolution
                  Emin,    &  ! Minimum energy for DOS(E) and dx^2(E,t)
                  Emax,    &  ! Maximum energy for DOS(E) and dx^2(E,t)
                  dE,      &  ! Energy resolution of DOS(E) and dx^2(E,t)
                  eta,     &  ! Energy smearing
                  nrow,    &  ! Sample size multiplication: number of rows
                  ncol,    &  ! Sample size multiplication: number of columns
                  W,       &  ! Anderson disorder
                  peh,     &  ! Density of e-h puddles
                  Weh,     &  ! Height of e-h puddles
                  leh,     &  ! Width of e-h puddles
                  Delta,   &  ! Staggered sublattice potential
                  VI_A,    &  ! Intrinsic spin-orbit coupling strength on sublattice A
                  VI_B,    &  ! Intrinsic spin-orbit coupling strength on sublattice B
                  VR,      &  ! Rashba spin-orbit coupling strength
                  VRaniso, &  ! Anisotropy of Rashba SOC (1=isotropic, 0=perfectly anisotropic)
                  VRin,    &  ! Rashba from in-plane field
                  VPIA_A,  &  ! PIA spin-orbit coupling strength on A sublattice
                  VPIA_B,  &  ! PIA spin-orbit coupling strength on B sublattice
                  a,       &  ! Width of the Hamiltonian spectrum (units of teV)
                  b,       &  ! Center of the Hamiltonian spectrum (units of teV)
                  teV,     &  ! NN hopping energy in eV
                  teV2,    &  ! 2-NN hopping energy in eV
                  iseed,   &  ! Seed for random number generator (0 means use the system clock)
                  axis        ! Spin polarization axis




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 0: read in simulation parameters, determine time step !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
OPEN(1,file='params.txt',status='old')
READ(1,nml=params)
CLOSE(1)

q      = 1.602176565d-19
hbar   = 1.054571726d-34/q
tstep  = tstep/1000.0d0   ! units of ps
PRINT*,tstep
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 1: allocate space for various arrays !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ALLOCATE(mu(nrecurs))
ALLOCATE(muspin(nrecurs))

nE = (Emax-Emin)/dE
ALLOCATE(dos(0:nE))
ALLOCATE(spin(0:nE))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 2: calculate the density of states !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Read in Chebyshev moments
OPEN(1,file='mu_01.txt',status='old')
DO j = 1,nrecurs
  READ(1,*)itemp,mu(j),rtemp1
ENDDO
CLOSE(1)


! Reconstruct DOS from moments
dos = 0.0d0
CALL densite(nrecurs, mu, Emin,dE,nE, eta, a,teV, dos)
dos = dos
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 3: for each time step, read in Chebyshev moments and calculate spin polarization        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
IF(axis.eq.'x') THEN
  OPEN(11,file='sxout.txt',status='replace')
ELSEIF(axis.eq.'y') THEN
  OPEN(11,file='syout.txt',status='replace')
ELSE
  OPEN(11,file='szout.txt',status='replace')
ENDIF
DO i = 0,nt

  IF(MOD(i,10) .eq. 0) PRINT*,'time (ps) = ',DBLE(i)*tstep


! Open files for reading
  WRITE(ci,'(I0.5)')i
  file_muspin = 'muspinz_'//ci//'.tx'
  IF(axis.eq.'x') file_muspin = 'muspinx_'//ci//'.tx'
  IF(axis.eq.'y') file_muspin = 'muspiny_'//ci//'.tx'
  OPEN(1,file=file_muspin,status='old')


! Read in Chebyshev moments
  DO j = 1,nrecurs
    READ(1,*)itemp,rtemp1,rtemp2
    muspin(j) = DCMPLX(rtemp1,rtemp2)
  ENDDO
  CLOSE(1)


! Calculate energy-dependent mean-square spreading
  spin = 0.0d0
  CALL densite_spin(nrecurs, muspin, Emin,dE,nE, eta, a,teV, spin)
  spin = spin / dos


! Write spin polarization to a file
  WRITE(11,*)DREAL(spin)

ENDDO
CLOSE(11)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 4: write some shiznit to a file !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
OPEN(1,file='t.txt',status='replace')
DO i = 0,nt
  WRITE(1,*)DBLE(i)*tstep
ENDDO
CLOSE(1)

OPEN(1,file='E.txt',status='replace')
DO i = 0,nE
  WRITE(1,*)(Emin + DBLE(i)*dE)
ENDDO
CLOSE(1)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



! Deallocate yo mama
DEALLOCATE(mu)
DEALLOCATE(muspin)
DEALLOCATE(dos)
DEALLOCATE(spin)



END PROGRAM process_spin
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: densite()                                                                              !
! Description: calculates the DOS of a tridiagonalized system using the continued fraction expansion !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE densite(nMom, mu, Emin,dE,nE,eta, a,teV, dos)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER                       :: nMom
DOUBLE PRECISION              :: mu(nMom)
DOUBLE PRECISION              :: Emin,dE
INTEGER                       :: nE
DOUBLE PRECISION              :: eta
DOUBLE PRECISION              :: a,teV
DOUBLE PRECISION              :: dos(0:nE)

INTEGER                       :: i,n, nMom_eta
DOUBLE PRECISION              :: E,lambda
DOUBLE PRECISION, ALLOCATABLE :: g(:)
DOUBLE PRECISION              :: suma

DOUBLE PRECISION, PARAMETER   :: pi = 3.14159265359d0



! Determine number of moments for desired (Lorentzian) broadening
!lambda = 4.0d0
!nMom_eta = lambda/(eta/(a*teV))
!IF(nMom_eta .gt. nMom) THEN
!  lambda = DBLE(nMom)*eta/(a*teV)
!  nMom_eta = nMom
!  PRINT*,'Warning: not enough moments ---> lambda adjusted to ',lambda
!ENDIF
!nMom = nMom_eta

! Get moments for Lorentzian broadening
!ALLOCATE(g(nMom))
!CALL LorentzKernel(nMom,g,lambda)


! Determine number of moments for desired (Gaussian) broadening
lambda = pi
nMom_eta = lambda/(eta/(a*teV))
IF(nMom_eta .gt. nMom) THEN
  nMom_eta = nMom
  PRINT*,'Warning: not enough moments ---> broadening is ',lambda/DBLE(nMom)*(a*teV)
ENDIF
nMom = nMom_eta

! Get moments for Jackson (Gaussian) broadening
ALLOCATE(g(nMom))
CALL JacksonKernel(nMom,g)



!$OMP PARALLEL SHARED(Emin,dE,g,a,mu,dos) &
!$OMP PRIVATE(i,n,suma)
!$OMP DO
DO i = 0,nE

  E    = Emin + DBLE(i)*dE
  E    = E/(a*teV)

  suma = g(1)*mu(1)
  DO n = 2,nMom
    suma = suma + 2.0d0*g(n)*mu(n)*DCOS(DBLE(n-1)*DACOS(E))
  ENDDO

  dos(i) = suma/pi/DSQRT(1.0d0-E*E)

ENDDO
!$OMP END DO
!$OMP END PARALLEL


DEALLOCATE(g)



END SUBROUTINE densite
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: densite_spin()                                                                         !
! Description: calculates energy-dependent spin polarization from the Chebyshev moments              !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE densite_spin(nMom, mu, Emin,dE,nE,eta, a,teV, spin)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER                       :: nMom
DOUBLE COMPLEX                :: mu(nMom)
DOUBLE PRECISION              :: Emin,dE
INTEGER                       :: nE
DOUBLE PRECISION              :: eta
DOUBLE PRECISION              :: a,teV
DOUBLE COMPLEX                :: spin(0:nE)

INTEGER                       :: i,n, nMom_eta
DOUBLE PRECISION              :: E,lambda
DOUBLE PRECISION, ALLOCATABLE :: g(:)
DOUBLE COMPLEX                :: suma

DOUBLE PRECISION, PARAMETER   :: pi = 3.14159265359d0



! Determine number of moments for desired (Lorentzian) broadening
!lambda = 4.0d0
!nMom_eta = lambda/(eta/(a*teV))
!IF(nMom_eta .gt. nMom) THEN
!  lambda = DBLE(nMom)*eta/(a*teV)
!  nMom_eta = nMom
!  PRINT*,'Warning: not enough moments ---> lambda adjusted to ',lambda
!ENDIF
!nMom = nMom_eta

! Get moments for Lorentzian broadening
!ALLOCATE(g(nMom))
!CALL LorentzKernel(nMom,g,lambda)


! Determine number of moments for desired (Gaussian) broadening
lambda = pi
nMom_eta = lambda/(eta/(a*teV))
IF(nMom_eta .gt. nMom) THEN
  nMom_eta = nMom
  PRINT*,'Warning: not enough moments ---> broadening is ',lambda/DBLE(nMom)*(a*teV)
ENDIF
nMom = nMom_eta

! Get moments for Jackson (Gaussian) broadening
ALLOCATE(g(nMom))
CALL JacksonKernel(nMom,g)



!$OMP PARALLEL SHARED(Emin,dE,g,a,mu,spin) &
!$OMP PRIVATE(i,n,suma)
!$OMP DO
DO i = 0,nE

  E    = Emin + DBLE(i)*dE
  E    = E/(a*teV)

  suma = g(1)*mu(1)
  DO n = 2,nMom
    suma = suma + 2.0d0*g(n)*mu(n)*DCOS(DBLE(n-1)*DACOS(E))
  ENDDO

  spin(i) = suma/pi/DSQRT(1.0d0-E*E)

ENDDO
!$OMP END DO
!$OMP END PARALLEL


DEALLOCATE(g)



END SUBROUTINE densite_spin
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: LorentzKernel()                                                                          !
! Description: calculates the kernel corresponding to Lorentzian broadening in the Chebyshev expansion !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE LorentzKernel(nMom,g,lambda)

IMPLICIT NONE

INTEGER          :: nMom
DOUBLE PRECISION :: lambda
DOUBLE PRECISION :: g(nMom)
INTEGER          :: n



DO n = 0,nMom-1
  g(n+1) = DSINH( lambda*(1.0d0 - DBLE(n)/DBLE(nMom)) ) / DSINH(lambda)
ENDDO



END SUBROUTINE LorentzKernel
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: JacksonKernel()                                                                          !
! Description: calculates the kernel corresponding to Gaussian broadening in the Chebyshev expansion   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE JacksonKernel(nMom,g)

IMPLICIT NONE

INTEGER          :: nMom
DOUBLE PRECISION :: g(nMom)
INTEGER          :: n

DOUBLE PRECISION, PARAMETER   :: pi = 3.14159265359d0



DO n = 0,nMom-1
  g(n+1) = ((DBLE(nMom)-DBLE(n)+1.0d0)*DCOS(pi*DBLE(n)/(DBLE(nMom)+1.0d0)) &
         + DSIN(pi*DBLE(n)/(DBLE(nMom)+1.0d0))/DTAN(pi/(DBLE(nMom)+1.0d0))) &
         / (DBLE(nMom)+1.0d0)
ENDDO



END SUBROUTINE JacksonKernel
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
