PROGRAM transportKubo_spin

USE constants
USE simulation
USE hamiltonian
USE operators
USE utils
USE wavepacket, ONLY: evolution,evolution_x,evolution_spin, &
                      init_state_random, &
                      init_state_zp,init_state_zm, &
                      init_state_xp,init_state_xm, &
                      init_state_yp,init_state_ym

IMPLICIT NONE

! Variables defining the Hamiltonian
INTEGER                       :: natoms, maxnn,maxnnn
INTEGER,          ALLOCATABLE :: nn(:,:), near(:), nnn(:,:), nnear(:)
DOUBLE PRECISION, ALLOCATABLE :: e(:), dx(:,:), dy(:,:), ddx(:,:), ddy(:,:)
DOUBLE COMPLEX,   ALLOCATABLE :: s(:,:), ss(:,:)

! Variables related to the wavepacket and time evolution
INTEGER                       :: npol
DOUBLE COMPLEX,   ALLOCATABLE :: psi(:), xupsi(:), yupsi(:), c(:)

! Variables related to density of states, and other operators
INTEGER                       :: nE
DOUBLE PRECISION, ALLOCATABLE :: dos(:)

! Loop index variables
INTEGER                       :: it
INTEGER                       :: i,j

DOUBLE COMPLEX, PARAMETER     :: zero = (0.d0,0.d0)
DOUBLE COMPLEX, PARAMETER     :: ii   = (0.d0,1.d0)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 1: Read in simulation and sample info and build the Hamiltonian !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
OPEN(1,file='params.txt',status='old')
READ(1,nml=params)
CLOSE(1)
CALL init_random_seed(iseed)
CALL make_h(natoms,maxnn, maxnnn, nn,nnn, near,nnear, e,s,ss, dx,dy,ddx,ddy)
CALL checkingHermiticity(2*natoms, 2*maxnn,nn,s, 2*maxnnn,nnn,ss)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 1.1: Center and normalize the Hamiltonian to [1-,1] !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
e  = ( e - b )/(a*teV)
s  = s/(a*teV)
ss = ss/(a*teV)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 2: allocate space for various quantities !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Density of states
nE = int( (Emax-Emin)/dE )
ALLOCATE(dos(0:nE))
dos = 0.0d0

! Wavepacket
ALLOCATE(  psi(2*natoms))
ALLOCATE(xupsi(2*natoms))
ALLOCATE(yupsi(2*natoms))
xupsi = zero
yupsi = zero



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 3: Calculate the coefficients for the time evolution operator !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CALL calculate_cn_bessel(tstep*1.0d-15*a*teV/hbar, a,b, npol,c)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 4: Define an initial state and calculate the initial DOS and spin !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Initialize wavepacket
IF(axis.eq.'x') THEN
  CALL init_state_xp(natoms, psi)
  !CALL init_state_xm(natoms, psi)
ELSEIF(axis.eq.'y') THEN
  CALL init_state_yp(natoms, psi)
  !CALL init_state_ym(natoms, psi)
ELSEIF(axis.eq.'z') THEN
  CALL init_state_zp(natoms, psi)
  !CALL init_state_zm(natoms, psi)
ELSE
  CALL init_state_random(natoms, psi)
ENDIF


! Calculate DOS and Hamiltonian spectrum
CALL get_dos(2*natoms,2*maxnn,2*maxnnn, nE, psi, e,s,ss,nn,nnn,near,nnear, 1, dos)

! Calculate and print out initial spin polarization
!CALL get_spin(natoms,maxnn,maxnnn, e,s,ss,nn,nnn,near,nnear, psi, nE,dos, 0,axis)



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 5: Time evolution of the initial wavepacket !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Loop over each time step
DO it = 1,nt

  PRINT*
  PRINT*
  PRINT*,'Time step = ',it

! Do time evolution of the wave packet
  CALL evolution(2*natoms,2*maxnn,2*maxnnn,npol, dx,dy,ddx,ddy, e,s,ss,nn,nnn,near,nnear, c, psi,xupsi,yupsi)
  !CALL evolution_x(2*natoms,2*maxnn,2*maxnnn,npol, dx,ddx, e,s,ss,nn,nnn,near,nnear, c, psi,xupsi)
  !CALL evolution_spin(2*natoms,2*maxnn,2*maxnnn,npol, e,s,ss,nn,nnn,near,nnear, c, psi)

! Calculate and print out dX² and dY²
  !CALL get_dX2(2*natoms,2*maxnn,2*maxnnn, e,s,ss,nn,nnn,near,nnear, xupsi, nE, dos, it)
  CALL get_dXY2(2*natoms,2*maxnn,2*maxnnn, e,s,ss,nn,nnn,near,nnear, xupsi,yupsi, nE, dos, it)

! Calculate and print out spin polarization
  !CALL get_spin(natoms,maxnn,maxnnn, e,s,ss,nn,nnn,near,nnear, psi, nE,dos, it,axis)


ENDDO




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Final step: deallocate yo mama !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DEALLOCATE(near)
DEALLOCATE(nn)
DEALLOCATE(dx)
DEALLOCATE(dy)
DEALLOCATE(nnear)
DEALLOCATE(nnn)
DEALLOCATE(ddx)
DEALLOCATE(ddy)
DEALLOCATE(e)
DEALLOCATE(s)
DEALLOCATE(ss)
DEALLOCATE(dos)
DEALLOCATE(psi)
DEALLOCATE(xupsi)
DEALLOCATE(yupsi)
DEALLOCATE(c)

PRINT*
PRINT*
PRINT*




END PROGRAM transportKubo_spin
