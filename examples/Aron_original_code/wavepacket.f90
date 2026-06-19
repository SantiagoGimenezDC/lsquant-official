MODULE wavepacket
CONTAINS



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_random                                                                       !
! Description: create an initial wavepacket state with random phase and spin orientation at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_random(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase, theta, phi

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with random spin orientiation at each site
DO i = 1,natoms
  CALL random_number(phase)
  CALL random_number(theta)
  CALL random_number(phi)

  psi(       i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase)       * DCOS(2.0d0*pi*theta/2.0d0)
  psi(natoms+i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*(phase+phi)) * DSIN(2.0d0*pi*theta/2.0d0)
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))




END SUBROUTINE init_state_random
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_zp                                                                  !
! Description: create an initial wavepacket state with random phase and spin up at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_zp(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase,theta,phi

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)

! Construct a random phase state with spin up at each site
DO i = 1,natoms
  CALL random_number(phase)
  CALL random_number(theta)
  CALL random_number(phi)

  psi(i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase)
ENDDO

! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))



! TEST: localized spin-up state on one atom
!!!psi        = (0.0d0,0.0d0)
!!!psi(68600) = (1.0d0,0.0d0)




END SUBROUTINE init_state_zp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_zm                                                                    !
! Description: create an initial wavepacket state with random phase and spin down at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_zm(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with spin down at each site
DO i = 1,natoms
  CALL random_number(phase)
  psi(natoms+i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase)       
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))




END SUBROUTINE init_state_zm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_xp                                                                  !
! Description: create an initial wavepacket state with random phase and spin +x at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_xp(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with spin +x at each site
DO i = 1,natoms
  CALL random_number(phase)

  psi(       i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase) / DSQRT(2.0d0)
  psi(natoms+i) = psi(i)
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))



! TEST: localized spin-x state on one atom
!!!psi               = (0.0d0,0.0d0)
!!!psi(68617)        = (1.0d0,0.0d0) / DSQRT(2.0d0)
!!!psi(68617+natoms) = psi(68617)



END SUBROUTINE init_state_xp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_xm                                                                  !
! Description: create an initial wavepacket state with random phase and spin -x at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_xm(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with spin -x at each site
DO i = 1,natoms
  CALL random_number(phase)

  psi(       i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase) / DSQRT(2.0d0)
  psi(natoms+i) = -psi(i)
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))




END SUBROUTINE init_state_xm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_yp                                                                  !
! Description: create an initial wavepacket state with random phase and spin +y at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_yp(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with spin +y at each site
DO i = 1,natoms
  CALL random_number(phase)

  psi(       i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase) / DSQRT(2.0d0)
  psi(natoms+i) = (0.0d0,1.0d0)*psi(i)
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))




END SUBROUTINE init_state_yp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_state_ym                                                                  !
! Description: create an initial wavepacket state with random phase and spin -y at each site !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_state_ym(natoms, psi)

USE constants
USE utils

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms

! Subroutine outputs
DOUBLE COMPLEX   :: psi(2*natoms)

! For random phase state
DOUBLE PRECISION :: phase

! Loop indices
INTEGER          :: i




psi = (0.0d0,0.0d0)


! Construct a random phase state with spin -y at each site
DO i = 1,natoms
  CALL random_number(phase)

  psi(       i) = CDEXP(2.0d0*pi*(0.0d0,1.0d0)*phase) / DSQRT(2.0d0)
  psi(natoms+i) = -(0.0d0,1.0d0)*psi(i)
ENDDO


! Normalize the wavepacket
psi = psi / DSQRT(DBLE(natoms))




END SUBROUTINE init_state_ym
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: calculate_cn_quad()                                   !
! Description: calculate the Chebyshev coefficients analytically    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE calculate_cn_quad(npol, T, ac,bc, c)

USE constants

IMPLICIT NONE

INCLUDE 'omp_lib.h'

! Subroutine inputs
INTEGER          :: npol
DOUBLE PRECISION :: T, ac,bc

! Subroutine outputs
DOUBLE COMPLEX   :: c(npol)

! Loop index variables
INTEGER          :: i,j,jmax

! Timing variables
DOUBLE PRECISION :: t1,t2




PRINT*
PRINT*,'Calculating the Chebyshev expansion coefficients...'


CALL cpu_time(t1)
!$ t1 = omp_get_wtime()



! Calculation of the Chebyshev expansion coefficients
! using Chebyshev-Gauss quadrature
jmax = npol+1
DO i = 1,npol

  c(i) = (0.0d0,0.0d0)

  DO j = 1,jmax
    c(i) = c(i) + DCOS(DBLE(i-1)*pi*(DBLE(j)-0.5d0)/DBLE(jmax)) &
                * CDEXP((0.0d0,-1.0d0)*2.0d0*bc*T*DCOS(pi*(DBLE(j)-0.5d0)/DBLE(jmax)))
  ENDDO

  c(i) = c(i) * CDEXP((0.0d0,-1.0d0)*ac*T) / DBLE(jmax)
  IF(i .gt. 1) c(i) = c(i) * DSQRT(2.0d0)

ENDDO



CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_cn = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*

OPEN(1,file='cn.txt',status='replace')
DO i = 1,npol
  WRITE(1,*)i,CDABS(c(i)),DREAL(c(i)),DIMAG(c(i))
ENDDO
CLOSE(1)




RETURN
END SUBROUTINE calculate_cn_quad
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: calculate_cn_bessel()                                 !
! Description: calculate the Chebyshev coefficients analytically    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE calculate_cn_bessel(T, ac,bc, npol,c)

USE constants

IMPLICIT NONE

INCLUDE 'omp_lib.h'

! Subroutine inputs
DOUBLE PRECISION :: T, ac,bc

! Subroutine outputs
INTEGER                     :: npol
DOUBLE COMPLEX, ALLOCATABLE :: c(:)

! For finding npol
INTEGER        :: flag
DOUBLE COMPLEX :: ctemp

! Loop index variables
INTEGER :: i,j,jmax

! Timing variables
DOUBLE PRECISION :: t1,t2




PRINT*
PRINT*,'Calculating the Chebyshev expansion coefficients...'


CALL cpu_time(t1)
!$ t1 = omp_get_wtime()



! Determine number of Chebyshev polynomials
flag = 0
i = 1
DO WHILE(flag.eq.0)

  ctemp = (0.0d0,-1.0d0)**DBLE(i-1) * DBESJN(i-1,2.0d0*bc*T)
  ctemp = ctemp * CDEXP((0.0d0,-1.0d0)*ac*T)
  ctemp = ctemp * DSQRT(2.0d0)
  IF(CDABS(ctemp) .lt. 1.0d-20) flag = 1
  i = i+1

ENDDO

npol = i
ALLOCATE(c(npol))
PRINT*,'  npol = ',npol


! Calculation of the Chebyshev expansion coefficients using Bessel functions
c(1) = exp((0.d0,-1.d0)*ac*T) * DBESJ0(2.d0*bc*T)  
DO i = 2,npol

  c(i) = (0.0d0,-1.0d0)**DBLE(i-1) * DBESJN(i-1,2.0d0*bc*T)
  c(i) = c(i) * CDEXP((0.0d0,-1.0d0)*ac*T)
  c(i) = c(i) * DSQRT(2.0d0)

ENDDO



CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_cn = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*

OPEN(1,file='cn.txt',status='replace')
DO i = 1,npol
  WRITE(1,*)i,CDABS(c(i)),DREAL(c(i)),DIMAG(c(i))
ENDDO
CLOSE(1)




RETURN
END SUBROUTINE calculate_cn_bessel
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: evolution()                                       !
! Description: evolves the wave packet forward one step in time !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE evolution(natoms,maxnn,maxnnn,npol, dx,dy,ddx,ddy, e,s,ss,nn,nnn,near,nnear, c, psi,xupsi,yupsi)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

! Subroutine inputs
INTEGER          :: natoms, maxnn,maxnnn, npol
DOUBLE PRECISION :: dx(natoms,maxnn),dy(natoms,maxnn), ddx(natoms,maxnnn),ddy(natoms,maxnnn), e(natoms)
DOUBLE COMPLEX   :: s(natoms,maxnn),ss(natoms,maxnnn)
INTEGER          :: nn(natoms,maxnn),nnn(natoms,maxnnn), near(natoms),nnear(natoms)
DOUBLE COMPLEX   :: c(npol)

! Subroutine outputs
DOUBLE COMPLEX   :: psi(natoms), xupsi(natoms),yupsi(natoms)

! Temporary storage for calculating wavepacket evolution
DOUBLE COMPLEX   :: pnpsi(natoms),pn1psi(natoms), xpnpsi(natoms),xpn1psi(natoms), ypnpsi(natoms),ypn1psi(natoms)
DOUBLE COMPLEX   :: xtemp,ytemp,ctemp
DOUBLE COMPLEX   :: cnum, xcnum,ycnum

! Loop index variables
INTEGER          :: i,j,n

! Timing variables
DOUBLE PRECISION :: t1,t2




! Get initial time
!!!CALL cpu_time(t1)
!$ t1 = omp_get_wtime()



! Recall that for the time evolution,
!    [X,U((m+1)T)]
!  = [X,U(T)*U(mT)]
!  = [X,U(T)]*U(mT)|psi0> + U(T)*[X,U(mT)]|psi0>
!  = [X,U(T)]*|psi> + U(T)*|xupsi>

! Let's calculate U(T)*|xupsi> first
!
! NOTE: in this portion,
!       xpnpsi  = alpha_n,
!       xpn1psi = alpha_n+1 or alpha_n-1
!
! First step of the expansion
DO i = 1,natoms
  xpn1psi(i) = xupsi(i)
  ypn1psi(i) = yupsi(i)
ENDDO

DO i = 1,natoms
  xcnum = e(i)*xpn1psi(i)
  ycnum = e(i)*ypn1psi(i)
  DO j = 1,near(i)
    xcnum = xcnum + s(i,j)*xpn1psi(nn(i,j))
    ycnum = ycnum + s(i,j)*ypn1psi(nn(i,j))
  ENDDO
  DO j = 1,nnear(i)
    xcnum = xcnum + ss(i,j)*xpn1psi(nnn(i,j))
    ycnum = ycnum + ss(i,j)*ypn1psi(nnn(i,j))
  ENDDO

  xpnpsi(i) = xcnum
  ypnpsi(i) = ycnum
  xupsi(i)  = c(1)*xupsi(i)
  yupsi(i)  = c(1)*yupsi(i)
ENDDO


! Subsequent steps of the expansion
DO n = 2,npol

!$OMP PARALLEL SHARED(e,s,ss,nn,nnn,xpnpsi,ypnpsi,xpn1psi,ypn1psi) &
!$OMP PRIVATE(i,j,xcnum,ycnum)
!$OMP DO
  DO i = 1,natoms
    xcnum = e(i)*xpnpsi(i)
    ycnum = e(i)*ypnpsi(i)
    DO j = 1,near(i)
      xcnum = xcnum + s(i,j)*xpnpsi(nn(i,j))
      ycnum = ycnum + s(i,j)*ypnpsi(nn(i,j))
    ENDDO
    DO j = 1,nnear(i)
      xcnum = xcnum + ss(i,j)*xpnpsi(nnn(i,j))
      ycnum = ycnum + ss(i,j)*ypnpsi(nnn(i,j))
    ENDDO
    xpn1psi(i) = 2.0d0*xcnum - xpn1psi(i)
    ypn1psi(i) = 2.0d0*ycnum - ypn1psi(i)
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL SHARED(xupsi,yupsi,xpnpsi,ypnpsi,xpn1psi,ypn1psi,c) &
!$OMP PRIVATE(i,xtemp,ytemp)
!$OMP DO
  DO i = 1,natoms
    xupsi(i)   = xupsi(i) + c(n)*xpnpsi(i)
    yupsi(i)   = yupsi(i) + c(n)*ypnpsi(i)
    xtemp      = xpnpsi(i)
    ytemp      = ypnpsi(i)
    xpnpsi(i)  = xpn1psi(i)
    ypnpsi(i)  = ypn1psi(i)
    xpn1psi(i) = xtemp
    ypn1psi(i) = ytemp
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

ENDDO



! Now let's calculate [X,U(T)]*|psi>
!
! NOTE: in this portion,
!       pnpsi  = alpha_n,
!       pn1psi = alpha_n+1 or alpha_n-1
!
!       xpnpsi  = betax_n,
!       xpn1psi = betax_n+1 or betax_n-1
!
!       ypnpsi  = betay_n,
!       ypn1psi = betay_n+1 or betay_n-1
!
! First step of the expansion
DO i = 1,natoms
  pn1psi(i)  = psi(i)
  xpn1psi(i) = (0.0d0,0.0d0)
  ypn1psi(i) = (0.0d0,0.0d0)
ENDDO

DO i = 1,natoms
  cnum  = e(i)*pn1psi(i)
  xcnum = (0.0d0,0.0d0)
  ycnum = (0.0d0,0.0d0)
  DO j = 1,near(i)
    cnum  = cnum  +         s(i,j)*pn1psi(nn(i,j))
    xcnum = xcnum - dx(i,j)*s(i,j)*pn1psi(nn(i,j))
    ycnum = ycnum - dy(i,j)*s(i,j)*pn1psi(nn(i,j))
  ENDDO
  DO j = 1,nnear(i)
    cnum  = cnum  +          ss(i,j)*pn1psi(nnn(i,j))
    xcnum = xcnum - ddx(i,j)*ss(i,j)*pn1psi(nnn(i,j))
    ycnum = ycnum - ddy(i,j)*ss(i,j)*pn1psi(nnn(i,j))
  ENDDO
  pnpsi(i)  =  cnum
  xpnpsi(i) = xcnum
  ypnpsi(i) = ycnum
  psi(i)    = c(1)*psi(i)
ENDDO


! Subsequent steps of the expansion
DO n = 2,npol

!$OMP PARALLEL SHARED(dx,dy,ddx,ddy,e,s,ss,nn,nnn,pnpsi,xpnpsi,ypnpsi,pn1psi,xpn1psi,ypn1psi) &
!$OMP PRIVATE(i,j,cnum,xcnum,ycnum)
!$OMP DO
  DO i = 1,natoms
    cnum  = e(i)*pnpsi(i)
    xcnum = e(i)*xpnpsi(i)
    ycnum = e(i)*ypnpsi(i)
    DO j = 1,near(i)
      cnum  = cnum  +         s(i,j)*pnpsi(nn(i,j))
      xcnum = xcnum - dx(i,j)*s(i,j)*pnpsi(nn(i,j)) + s(i,j)*xpnpsi(nn(i,j))
      ycnum = ycnum - dy(i,j)*s(i,j)*pnpsi(nn(i,j)) + s(i,j)*ypnpsi(nn(i,j))
    ENDDO
    DO j = 1,nnear(i)
      cnum  = cnum  +          ss(i,j)*pnpsi(nnn(i,j))
      xcnum = xcnum - ddx(i,j)*ss(i,j)*pnpsi(nnn(i,j)) + ss(i,j)*xpnpsi(nnn(i,j))
      ycnum = ycnum - ddy(i,j)*ss(i,j)*pnpsi(nnn(i,j)) + ss(i,j)*ypnpsi(nnn(i,j))
    ENDDO
    pn1psi(i)  = 2.0d0* cnum -  pn1psi(i)
    xpn1psi(i) = 2.0d0*xcnum - xpn1psi(i)
    ypn1psi(i) = 2.0d0*ycnum - ypn1psi(i)
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL SHARED(psi,xupsi,yupsi,pnpsi,xpnpsi,ypnpsi,pn1psi,xpn1psi,ypn1psi,c) &
!$OMP PRIVATE(i,ctemp,xtemp,ytemp)
!$OMP DO
  DO i = 1,natoms
    psi(i)     = psi(i)   + c(n)*pnpsi(i)
    xupsi(i)   = xupsi(i) + c(n)*xpnpsi(i)
    yupsi(i)   = yupsi(i) + c(n)*ypnpsi(i)
    ctemp      = pnpsi(i)
    xtemp      = xpnpsi(i)
    ytemp      = ypnpsi(i)
    pnpsi(i)   = pn1psi(i)
    xpnpsi(i)  = xpn1psi(i)
    ypnpsi(i)  = ypn1psi(i)
    pn1psi(i)  = ctemp
    xpn1psi(i) = xtemp
    ypn1psi(i) = ytemp
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

ENDDO



! To test the unitarity of U(T)
cnum = (0.0d0,0.0d0)
DO i = 1,natoms
  cnum = cnum + psi(i)*DCONJG(psi(i))
ENDDO
PRINT*
PRINT*,'  |U(t)|           = ',DSQRT(CDABS(cnum))



! Get final time and print out total time to evolve wavepacket
!!!CALL cpu_time(t2)
!$ t2 = omp_get_wtime()
PRINT*,'  t_evolution      = ',t2-t1,'seconds'




END SUBROUTINE evolution
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: evolution_x()                                     !
! Description: evolves the wave packet forward one step in time !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE evolution_x(natoms,maxnn,maxnnn,npol, dx,ddx, e,s,ss,nn,nnn,near,nnear, c, psi,xupsi)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

! Subroutine inputs
INTEGER          :: natoms, maxnn,maxnnn, npol
DOUBLE PRECISION :: dx(natoms,maxnn), ddx(natoms,maxnnn), e(natoms)
DOUBLE COMPLEX   :: s(natoms,maxnn),ss(natoms,maxnnn)
INTEGER          :: nn(natoms,maxnn),nnn(natoms,maxnnn), near(natoms),nnear(natoms)
DOUBLE COMPLEX   :: c(npol)

! Subroutine outputs
DOUBLE COMPLEX   :: psi(natoms), xupsi(natoms)

! Temporary storage for calculating wavepacket evolution
DOUBLE COMPLEX   :: pnpsi(natoms),pn1psi(natoms), xpnpsi(natoms),xpn1psi(natoms)
DOUBLE COMPLEX   :: xtemp,ctemp
DOUBLE COMPLEX   :: cnum, xcnum

! Loop index variables
INTEGER          :: i,j,n

! Timing variables
DOUBLE PRECISION :: t1,t2




! Get initial time
!!!CALL cpu_time(t1)
!$ t1 = omp_get_wtime()



! Recall that for the time evolution,
!    [X,U((m+1)T)]
!  = [X,U(T)*U(mT)]
!  = [X,U(T)]*U(mT)|psi0> + U(T)*[X,U(mT)]|psi0>
!  = [X,U(T)]*|psi> + U(T)*|xupsi>

! Let's calculate U(T)*|xupsi> first
!
! NOTE: in this portion,
!       xpnpsi  = alpha_n,
!       xpn1psi = alpha_n+1 or alpha_n-1
!
! First step of the expansion
DO i = 1,natoms
  xpn1psi(i) = xupsi(i)
ENDDO

DO i = 1,natoms
  xcnum = e(i)*xpn1psi(i)
  DO j = 1,near(i)
    xcnum = xcnum + s(i,j)*xpn1psi(nn(i,j))
  ENDDO
  DO j = 1,nnear(i)
    xcnum = xcnum + ss(i,j)*xpn1psi(nnn(i,j))
  ENDDO

  xpnpsi(i) = xcnum
  xupsi(i)  = c(1)*xupsi(i)
ENDDO


! Subsequent steps of the expansion
DO n = 2,npol

!$OMP PARALLEL SHARED(e,s,ss,nn,nnn,xpnpsi,xpn1psi) &
!$OMP PRIVATE(i,j,xcnum)
!$OMP DO
  DO i = 1,natoms
    xcnum = e(i)*xpnpsi(i)
    DO j = 1,near(i)
      xcnum = xcnum + s(i,j)*xpnpsi(nn(i,j))
    ENDDO
    DO j = 1,nnear(i)
      xcnum = xcnum + ss(i,j)*xpnpsi(nnn(i,j))
    ENDDO
    xpn1psi(i) = 2.0d0*xcnum - xpn1psi(i)
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL SHARED(xupsi,xpnpsi,xpn1psi,c) &
!$OMP PRIVATE(i,xtemp)
!$OMP DO
  DO i = 1,natoms
    xupsi(i)   = xupsi(i) + c(n)*xpnpsi(i)
    xtemp      = xpnpsi(i)
    xpnpsi(i)  = xpn1psi(i)
    xpn1psi(i) = xtemp
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

ENDDO



! Now let's calculate [X,U(T)]*|psi>
!
! NOTE: in this portion,
!       pnpsi  = alpha_n,
!       pn1psi = alpha_n+1 or alpha_n-1
!
!       xpnpsi  = betax_n,
!       xpn1psi = betax_n+1 or betax_n-1
!
! First step of the expansion
DO i = 1,natoms
  pn1psi(i)  = psi(i)
  xpn1psi(i) = (0.0d0,0.0d0)
ENDDO

DO i = 1,natoms
  cnum  = e(i)*pn1psi(i)
  xcnum = (0.0d0,0.0d0)
  DO j = 1,near(i)
    cnum  = cnum  +         s(i,j)*pn1psi(nn(i,j))
    xcnum = xcnum - dx(i,j)*s(i,j)*pn1psi(nn(i,j))
  ENDDO
  DO j = 1,nnear(i)
    cnum  = cnum  +          ss(i,j)*pn1psi(nnn(i,j))
    xcnum = xcnum - ddx(i,j)*ss(i,j)*pn1psi(nnn(i,j))
  ENDDO
  pnpsi(i)  =  cnum
  xpnpsi(i) = xcnum
  psi(i)    = c(1)*psi(i)
ENDDO


! Subsequent steps of the expansion
DO n = 2,npol

!$OMP PARALLEL SHARED(dx,ddx,e,s,ss,nn,nnn,pnpsi,xpnpsi,pn1psi,xpn1psi) &
!$OMP PRIVATE(i,j,cnum,xcnum)
!$OMP DO
  DO i = 1,natoms
    cnum  = e(i)*pnpsi(i)
    xcnum = e(i)*xpnpsi(i)
    DO j = 1,near(i)
      cnum  = cnum  +         s(i,j)*pnpsi(nn(i,j))
      xcnum = xcnum - dx(i,j)*s(i,j)*pnpsi(nn(i,j)) + s(i,j)*xpnpsi(nn(i,j))
    ENDDO
    DO j = 1,nnear(i)
      cnum  = cnum  +          ss(i,j)*pnpsi(nnn(i,j))
      xcnum = xcnum - ddx(i,j)*ss(i,j)*pnpsi(nnn(i,j)) + ss(i,j)*xpnpsi(nnn(i,j))
    ENDDO
    pn1psi(i)  = 2.0d0* cnum -  pn1psi(i)
    xpn1psi(i) = 2.0d0*xcnum - xpn1psi(i)
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL SHARED(psi,xupsi,pnpsi,xpnpsi,pn1psi,xpn1psi,c) &
!$OMP PRIVATE(i,ctemp,xtemp)
!$OMP DO
  DO i = 1,natoms
    psi(i)     = psi(i)   + c(n)*pnpsi(i)
    xupsi(i)   = xupsi(i) + c(n)*xpnpsi(i)
    ctemp      = pnpsi(i)
    xtemp      = xpnpsi(i)
    pnpsi(i)   = pn1psi(i)
    xpnpsi(i)  = xpn1psi(i)
    pn1psi(i)  = ctemp
    xpn1psi(i) = xtemp
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

ENDDO



! To test the unitarity of U(T)
cnum = (0.0d0,0.0d0)
DO i = 1,natoms
  cnum = cnum + psi(i)*DCONJG(psi(i))
ENDDO
PRINT*
PRINT*,'  |U(t)|           = ',DSQRT(CDABS(cnum))



! Get final time and print out total time to evolve wavepacket
!!!CALL cpu_time(t2)
!$ t2 = omp_get_wtime()
PRINT*,'  t_evolution      = ',t2-t1,'seconds'




END SUBROUTINE evolution_x
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: evolution_spin()                                  !
! Description: evolves the wave packet forward one step in time !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE evolution_spin(natoms,maxnn,maxnnn,npol, e,s,ss,nn,nnn,near,nnear, c, psi)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

! Subroutine inputs
INTEGER          :: natoms, maxnn, maxnnn, npol
DOUBLE PRECISION :: e(natoms)
DOUBLE COMPLEX   :: s(natoms,maxnn), ss(natoms,maxnnn)
INTEGER          :: nn(natoms,maxnn), nnn(natoms,maxnnn),  near(natoms), nnear(natoms)
DOUBLE COMPLEX   :: c(npol)

! Subroutine outputs
DOUBLE COMPLEX   :: psi(natoms)

! Temporary storage for calculating wavepacket evolution
DOUBLE COMPLEX   :: pnpsi(natoms), pn1psi(natoms)
DOUBLE COMPLEX   :: ctemp
DOUBLE COMPLEX   :: cnum

! Loop index variables
INTEGER          :: i,j,n

! Timing variables
DOUBLE PRECISION :: t1,t2




! Get initial time
!!!CALL cpu_time(t1)
!$ t1 = omp_get_wtime()



! Calculate U(T)*|psi>
!
! NOTE: pnpsi  = alpha_n,
!       pn1psi = alpha_n+1 or alpha_n-1
!
! First step of the expansion
DO i = 1,natoms
  pn1psi(i) = psi(i)
ENDDO

DO i = 1,natoms
  cnum = e(i)*pn1psi(i)
  DO j = 1,near(i)
    cnum = cnum + s(i,j)*pn1psi(nn(i,j))
  ENDDO
  DO j = 1,nnear(i)
    cnum = cnum + ss(i,j)*pn1psi(nnn(i,j))
  ENDDO
  pnpsi(i) =  cnum
  psi(i) = c(1)*psi(i)
ENDDO


! Subsequent steps of the expansion
DO n = 2,npol

!$OMP PARALLEL SHARED(e,s,ss,nn,nnn,near,nnear,pnpsi,pn1psi) &
!$OMP PRIVATE(i,j,cnum)
!$OMP DO
  DO i = 1,natoms
    cnum = e(i)*pnpsi(i)
    DO j = 1,near(i)
      cnum = cnum + s(i,j)*pnpsi(nn(i,j))
    ENDDO
    DO j = 1,nnear(i)
      cnum = cnum + ss(i,j)*pnpsi(nnn(i,j))
    ENDDO
    pn1psi(i) = 2.0d0*cnum - pn1psi(i)
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL SHARED(psi,pnpsi,pn1psi,c) &
!$OMP PRIVATE(i,ctemp)
!$OMP DO
  DO i = 1,natoms
    psi(i)    = psi(i) + c(n)*pnpsi(i)
    ctemp     = pnpsi(i)
    pnpsi(i)  = pn1psi(i)
    pn1psi(i) = ctemp
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

ENDDO



! To test the unitarity of U(T)
cnum = (0.0d0,0.0d0)
DO i = 1,natoms
  cnum = cnum + psi(i)*DCONJG(psi(i))
ENDDO
PRINT*
PRINT*,'  |U(t)|           = ',DSQRT(CDABS(cnum))



! Get final time and print out total time to evolve wavepacket
!!!CALL cpu_time(t2)
!$ t2 = omp_get_wtime()
PRINT*,'  t_evolution_spin = ',t2-t1,'seconds'




END SUBROUTINE evolution_spin
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


END MODULE wavepacket
