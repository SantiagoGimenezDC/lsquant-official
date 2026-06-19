MODULE operators
CONTAINS



SUBROUTINE get_dos(natoms,maxnn,maxnnn,nE,psi,e,s,ss,nn,nnn,near,nnear,iRP,dos)

USE simulation, ONLY: nMom=>nrecurs,Emin,dE,eta,a

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn,nE
DOUBLE COMPLEX, INTENT(IN)    :: psi(natoms)
DOUBLE PRECISION, INTENT(IN)  :: e(natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(natoms,maxnn),ss(natoms,maxnnn)
INTEGER, INTENT(IN)           :: nn(natoms,maxnn),nnn(natoms,maxnnn)
INTEGER, INTENT(IN)           :: near(natoms),nnear(natoms)

INTEGER, INTENT(IN)           :: iRP
DOUBLE PRECISION, INTENT(OUT) :: dos(0:nE)

DOUBLE COMPLEX, ALLOCATABLE   :: mu(:)
DOUBLE PRECISION              :: t1,t2
DOUBLE PRECISION              :: Edos

INTEGER                       :: i
CHARACTER(2)                  :: cRP
CHARACTER(9)                  :: filename




! Calculate Chebyshev moments 
PRINT*
PRINT*,'Running Chebyshev recursion (DOS)...'



CALL cpu_time(t1)
!$ t1 = omp_get_wtime()

ALLOCATE(mu(nMom+1))
CALL MomentosDelta(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,psi,e,s,ss,mu)

CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_recursion = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*



WRITE(cRP,'(I0.2)')iRP
filename = 'mu_'//cRP//'.txt'
OPEN(101,file=filename,status='replace',action='write')
DO i = 1,nMom
  WRITE(101,*) i,dble(mu(i)),dimag(mu(i))
ENDDo
CLOSE(101)




DEALLOCATE(mu)


END SUBROUTINE get_dos

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE get_spin(natoms,maxnn,maxnnn, e,s,ss,nn,nnn,near,nnear, psi, nE, dos, it,axis)

USE simulation, ONLY: nMom=>nrecurs,Emin,dE,eta,a

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn
DOUBLE PRECISION, INTENT(IN)  :: e(2*natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(2*natoms,2*maxnn),ss(2*natoms,2*maxnnn)
INTEGER, INTENT(IN)           :: nn(2*natoms,2*maxnn),nnn(2*natoms,2*maxnnn)
INTEGER, INTENT(IN)           :: near(2*natoms),nnear(2*natoms)
DOUBLE COMPLEX, INTENT(IN)    :: psi(2*natoms)
INTEGER, INTENT(IN)           :: nE
DOUBLE PRECISION, INTENT(IN)  :: dos(0:nE)
INTEGER, INTENT(IN)           :: it
CHARACTER, INTENT(IN)         :: axis

DOUBLE COMPLEX, ALLOCATABLE   :: muspin(:)
DOUBLE COMPLEX, ALLOCATABLE   :: sx(:),sy(:),sz(:)

DOUBLE PRECISION              :: Edos
DOUBLE PRECISION              :: t1,t2

INTEGER                       :: i

! File output
CHARACTER(5)                  :: cit
CHARACTER(16)                 :: filename




PRINT*
PRINT*,'Running Chebyshev recursion (spin)...'



CALL cpu_time(t1)
!$ t1 = omp_get_wtime()

ALLOCATE(muspin(nMom))
CALL MomentosSpin(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,psi,e,s,ss,muspin,axis)

CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_recursion = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*



WRITE(cit,'(I0.5)')it
IF(axis.eq.'x') THEN
  filename = 'muspinx_'//cit//'.txt'
ELSEIF(axis.eq.'y') THEN
  filename = 'muspiny_'//cit//'.txt'
ELSE
  filename = 'muspinz_'//cit//'.txt'
ENDIF
OPEN(103,file=filename,status='replace',action='write')
DO i = 1,nMom
  WRITE(103,*) i,DBLE(muspin(i)),DIMAG(muspin(i))
ENDDO
CLOSE(103)



DEALLOCATE(muspin)

END SUBROUTINE get_spin

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE get_dXY2(natoms,maxnn,maxnnn, e,s,ss,nn,nnn,near,nnear, xupsi,yupsi, nE, dos, it)

USE simulation, ONLY: nMom=>nrecurs,Emin,dE,eta,a

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn,nE
DOUBLE COMPLEX, INTENT(INOUT) :: xupsi(natoms),yupsi(natoms)
DOUBLE PRECISION, INTENT(IN)  :: e(natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(natoms,maxnn),ss(natoms,maxnnn)
INTEGER, INTENT(IN)           :: nn(natoms,maxnn),nnn(natoms,maxnnn)
INTEGER, INTENT(IN)           :: near(natoms),nnear(natoms)
DOUBLE PRECISION, INTENT(IN)  :: dos(0:nE)
INTEGER, INTENT(IN)           :: it

DOUBLE COMPLEX, ALLOCATABLE   :: mux(:),muy(:)
DOUBLE COMPLEX                :: cnumx,cnumy
DOUBLE PRECISION              :: normdx,normdy
DOUBLE PRECISION              :: t1,t2
DOUBLE PRECISION              :: Edos
DOUBLE PRECISION, ALLOCATABLE :: dX2(:),dY2(:)

INTEGER                       :: i
CHARACTER(5)                  :: cit
CHARACTER(14)                 :: filename

DOUBLE COMPLEX, PARAMETER     :: zero = (0.d0,0.d0)




! Mean-square spreading along x
PRINT*
PRINT*,'Running Chebyshev recursion (dX2) ...'



! Normalize xupsi for Lanczos recursion
cnumx = zero
cnumy = zero
DO i = 1,natoms
  cnumx = cnumx + xupsi(i)*DCONJG(xupsi(i))
  cnumy = cnumy + yupsi(i)*DCONJG(yupsi(i))
ENDDO
normdx = DSQRT( CDABS(cnumx) )
normdy = DSQRT( CDABS(cnumy) )
xupsi = xupsi/normdx
yupsi = yupsi/normdy



CALL cpu_time(t1)
!$ t1 = omp_get_wtime()

ALLOCATE(mux(nMom+1))
CALL MomentosDelta(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,xupsi,e,s,ss,mux)
ALLOCATE(muy(nMom+1))
CALL MomentosDelta(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,yupsi,e,s,ss,muy)

CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_recursion = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*



xupsi = xupsi*normdx
yupsi = yupsi*normdy



WRITE(cit,'(I0.5)')it
filename = 'muxy_'//cit//'.txt'
OPEN(106,file=filename,status='replace',action='write')
WRITE(106,*) "#",normdx,normdy
DO i = 1,nMom
  WRITE(106,*) i,dble(mux(i)),dimag(mux(i)),dble(muy(i)),dimag(muy(i))
ENDDO
CLOSE(106)



DEALLOCATE(mux)
DEALLOCATE(muy)

END SUBROUTINE get_dXY2

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE get_dX2(natoms,maxnn,maxnnn, e,s,ss,nn,nnn,near,nnear, xupsi, nE, dos, it)

USE simulation, ONLY: nMom=>nrecurs,Emin,dE,eta,a

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn,nE
DOUBLE COMPLEX, INTENT(INOUT) :: xupsi(natoms)
DOUBLE PRECISION, INTENT(IN)  :: e(natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(natoms,maxnn),ss(natoms,maxnnn)
INTEGER, INTENT(IN)           :: nn(natoms,maxnn),nnn(natoms,maxnnn)
INTEGER, INTENT(IN)           :: near(natoms),nnear(natoms)
DOUBLE PRECISION, INTENT(IN)  :: dos(0:nE)
INTEGER, INTENT(IN)           :: it

DOUBLE COMPLEX, ALLOCATABLE   :: mux(:)
DOUBLE COMPLEX                :: cnumx
DOUBLE PRECISION              :: normdx
DOUBLE PRECISION              :: t1,t2

INTEGER                       :: i
CHARACTER(5)                  :: cit
CHARACTER(14)                 :: filename

DOUBLE COMPLEX, PARAMETER     :: zero = (0.d0,0.d0)




! Mean-square spreading along x
PRINT*
PRINT*,'Running Chebyshev recursion (dX2) ...'



! Normalize xupsi for Lanczos recursion
cnumx = zero
DO i = 1,natoms
  cnumx = cnumx + xupsi(i)*DCONJG(xupsi(i))
ENDDO
normdx = DSQRT( CDABS(cnumx) )
xupsi = xupsi/normdx



CALL cpu_time(t1)
!$ t1 = omp_get_wtime()

ALLOCATE(mux(nMom+1))
CALL MomentosDelta(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,xupsi,e,s,ss,mux)

CALL cpu_time(t2)
!$ t2 = omp_get_wtime()

PRINT*,'  t_recursion = ',t2-t1,'seconds'
PRINT*,'...done'
PRINT*
PRINT*



xupsi = xupsi*normdx



WRITE(cit,'(I0.5)')it
filename = 'mux_'//cit//'.txt'
OPEN(106,file=filename,status='replace',action='write')
WRITE(106,*) "#",normdx
DO i = 1,nMom
  WRITE(106,*) i,dble(mux(i)),dimag(mux(i))
ENDDO
CLOSE(106)



DEALLOCATE(mux)

END SUBROUTINE get_dX2

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE MomentosSpin(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,psi,e,s,ss,muspin,axis)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)          :: natoms,maxnn,maxnnn,nMom
INTEGER, INTENT(IN)          :: near(2*natoms),nnear(2*natoms)
INTEGER, INTENT(IN)          :: nn(2*natoms,2*maxnn),nnn(2*natoms,2*maxnnn)
DOUBLE COMPLEX, INTENT(IN)   :: psi(2*natoms)
DOUBLE PRECISION, INTENT(IN) :: e(2*natoms)
DOUBLE COMPLEX, INTENT(IN)   :: s(2*natoms,2*maxnn),ss(2*natoms,2*maxnnn)
DOUBLE COMPLEX, INTENT(OUT)  :: muspin(nMom)
CHARACTER, INTENT(IN)        :: axis

DOUBLE COMPLEX, ALLOCATABLE  :: rs_nm1(:),rs_n(:)
DOUBLE COMPLEX               :: sumas,sumbs
DOUBLE COMPLEX               :: sspsi
INTEGER                      :: i,j,m

DOUBLE PRECISION              :: t1,t2
DOUBLE PRECISION              :: ta,tb

DOUBLE COMPLEX, PARAMETER    :: ii   = (0.d0,1.d0)
DOUBLE COMPLEX, PARAMETER    :: zero = (0.d0,0.d0)




muspin = zero



! |rs_nm1> = \sigma |psi>
ALLOCATE(rs_nm1(2*natoms))
ALLOCATE(rs_n(2*natoms))

IF(axis.eq.'x') THEN
  rs_nm1(1:natoms)          =  psi(natoms+1:2*natoms)
  rs_nm1(natoms+1:2*natoms) =  psi(1:natoms)
ELSEIF(axis.eq.'y') THEN
  rs_nm1(1:natoms)          = -ii*psi(natoms+1:2*natoms)
  rs_nm1(natoms+1:2*natoms) =  ii*psi(1:natoms)
ELSE
  rs_nm1(1:natoms)          =  psi(1:natoms)
  rs_nm1(natoms+1:2*natoms) = -psi(natoms+1:2*natoms)
ENDIF



! |rx_n> = H \sigma_x |psi>
! |ry_n> = H \sigma_y |psi>
! |rz_n> = H \sigma_z |psi>
sumas  = zero
sumbs  = zero

DO i = 1,2*natoms

  sspsi = e(i)*rs_nm1(i)

  DO j = 1,near(i)
    sspsi = sspsi + s(i,j)*rs_nm1(nn(i,j))
  ENDDO

  DO j = 1,nnear(i)
    sspsi = sspsi + ss(i,j)*rs_nm1(nnn(i,j))
  ENDDO

  rs_n(i)  = sspsi

  sumas   = sumas + rs_nm1(i)*DCONJG(psi(i))
  sumbs   = sumbs + sspsi*DCONJG(psi(i))

ENDDO

muspin(1) = sumas
muspin(2) = sumbs



DO m = 3,nMom

  ! |rs_nm1> =  2*H|rs_n>   - |rs_nm1>
  !  muspin(n)  = <psi|rs_nm1>

  !!!call cpu_time(t1)
  !!!!$ t1 = omp_get_wtime()

  CALL mv_muspin(natoms,maxnn,maxnnn,nn,nnn,near,nnear,e,s,ss,rs_n,rs_nm1,psi,muspin(m))

  !!!call cpu_time(t2)
  !!!!$ t2 = omp_get_wtime()
  !!!ta = ta + (t2-t1)


  !!!call cpu_time(t1)
  !!!!$ t1 = omp_get_wtime()

  CALL Update(2*natoms,rs_n,rs_nm1)

  !!!call cpu_time(t2)
  !!!!$ t2 = omp_get_wtime()
  !!!tb = tb + (t2-t1)

ENDDO

!!!write(*,*)
!!!write(*,*) 'MomentosSpin'
!!!write(*,*) 'ta (recursion): ', ta
!!!write(*,*) 'tb    (update): ', tb

DEALLOCATE(rs_nm1)
DEALLOCATE(rs_n)

END SUBROUTINE MomentosSpin

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE mv_muspin(natoms,maxnn,maxnnn,nn,nnn,near,nnear,e,s,ss,rs_n,rs_nm1,psi,sumas)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn
INTEGER, INTENT(IN)           :: nn(2*natoms,2*maxnn),nnn(2*natoms,2*maxnnn)
INTEGER, INTENT(IN)           :: near(2*natoms),nnear(2*natoms)
DOUBLE PRECISION, INTENT(IN)  :: e(2*natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(2*natoms,2*maxnn),ss(2*natoms,2*maxnnn)
DOUBLE COMPLEX, INTENT(IN)    :: rs_n(2*natoms)
DOUBLE COMPLEX, INTENT(INOUT) :: rs_nm1(2*natoms)
DOUBLE COMPLEX, INTENT(IN)    :: psi(2*natoms)
DOUBLE COMPLEX, INTENT(OUT)   :: sumas

DOUBLE COMPLEX                :: sspsi
DOUBLE COMPLEX                :: tmps,tmpsi
INTEGER                       :: tmpnn
INTEGER                       :: i,j
DOUBLE COMPLEX, PARAMETER     :: zero = (0.d0,0.d0)


sumas = zero

!$OMP PARALLEL SHARED(rs_n,rs_nm1,nn,nnn,near,nnear,e,s,ss,psi) &
!$OMP PRIVATE(i,j,tmps,tmpsi,sspsi) &
!$OMP REDUCTION(+: sumas)
!$OMP DO
DO i = 1,2*natoms

  sspsi  = e(i)*rs_n(i) - 0.5d0*rs_nm1(i)

  DO j = 1,near(i)
    sspsi = sspsi + s(i,j)*rs_n(nn(i,j))
  ENDDO

  DO j = 1,nnear(i)
    sspsi = sspsi + ss(i,j)*rs_n(nnn(i,j))
  ENDDO

  sumas    = sumas + DCONJG(psi(i))*sspsi
  rs_nm1(i) = 2.0d0*sspsi

ENDDO
!$OMP END DO
!$OMP END PARALLEL

sumas = 2.0d0*sumas

END SUBROUTINE mv_muspin

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE MomentosDelta(natoms,maxnn,maxnnn,nMom,near,nnear,nn,nnn,psi,e,s,ss,mu)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)          :: natoms,maxnn,maxnnn,nMom
INTEGER, INTENT(IN)          :: near(natoms),nnear(natoms),nn(natoms,maxnn),nnn(natoms,maxnnn)
DOUBLE COMPLEX, INTENT(IN)   :: psi(natoms)
DOUBLE PRECISION, INTENT(IN) :: e(natoms)
DOUBLE COMPLEX, INTENT(IN)   :: s(natoms,maxnn),ss(natoms,maxnnn)
DOUBLE COMPLEX, INTENT(OUT)  :: mu(nMom+1)

INTEGER                      :: i,j,n
DOUBLE PRECISION             :: t1,t2,ta,tb
DOUBLE COMPLEX               :: suma,cpsi,tmpsi,sumb
DOUBLE COMPLEX, ALLOCATABLE  :: r_n(:),r_nm1(:)

DOUBLE COMPLEX, PARAMETER    :: zero = (0.d0,0.d0)
DOUBLE COMPLEX, PARAMETER    :: id   = (1.d0,0.d0)
DOUBLE COMPLEX, PARAMETER    :: ii   = (0.d0,0.d0)



mu = zero

ALLOCATE(r_n(natoms))
ALLOCATE(r_nm1(natoms))

! mu(1) = 1 = <psi|psi>
mu(1) = id 

! |r_nm1> =  |psi>
! |r_n>   = H|psi>
! mu(2)   = <r_nm1|r_n>
! mu(3)   = 2*<r_n|r_n> - mu(1)
 
suma  = zero
sumb  = zero

DO i = 1,natoms

  tmpsi    = psi(i)
  r_nm1(i) = tmpsi

  cpsi     = e(i)*tmpsi

  DO j = 1,near(i)
    cpsi = cpsi + s(i,j)*psi(nn(i,j))
  ENDDO

  DO j = 1,nnear(i)
    cpsi = cpsi + ss(i,j)*psi(nnn(i,j))
  ENDDO

  r_n(i)   = cpsi

  suma     = suma + cpsi*DCONJG(tmpsi)
  sumb     = sumb + 2.d0*cpsi*DCONJG(cpsi)

ENDDO

mu(2) = suma
mu(3) = sumb - mu(1)

DO n = 3,nMom/2+1

  ! |r_nm1>   =  2*H|r_n>       - |r_nm1>
  ! mu(2*n-1) = 2*<r_n|r_n>     - mu(1)
  ! mu(2*n-2) = 2*<r_n|r_(n-1)> - mu(2)

  !!!call cpu_time(t1)
  !!!!$ t1 = omp_get_wtime()

  CALL mv_2mu(natoms,maxnn,maxnnn,nn,nnn,near,nnear,e,s,ss,r_n,r_nm1,suma,sumb)

  !!!call cpu_time(t2)
  !!!!$ t2 = omp_get_wtime()
  !!!ta = ta + (t2-t1)

  mu(2*n-2) =  suma - mu(2)
  mu(2*n-1) =  sumb - mu(1)

  ! Update

  !!!call cpu_time(t1)
  !!!!$ t1 = omp_get_wtime()

  CALL Update(natoms,r_n,r_nm1)

  !!!call cpu_time(t2)
  !!!!$ t2 = omp_get_wtime()
  !!!tb = tb + (t2-t1)


ENDDO

!!!write(*,*)
!!!write(*,*) 'MomentosDelta'
!!!write(*,*) 'ta (recursion): ', ta
!!!write(*,*) 'tb    (update): ', tb


DEALLOCATE(r_n)
DEALLOCATE(r_nm1)


END SUBROUTINE MomentosDelta

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE mv_2mu(natoms,maxnn,maxnnn,nn,nnn,near,nnear,e,s,ss,r_n,r_nm1,suma,sumb)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms,maxnn,maxnnn
INTEGER, INTENT(IN)           :: nn(natoms,maxnn),nnn(natoms,maxnnn)
INTEGER, INTENT(IN)           :: near(natoms),nnear(natoms)
DOUBLE PRECISION, INTENT(IN)  :: e(natoms)
DOUBLE COMPLEX, INTENT(IN)    :: s(natoms,maxnn),ss(natoms,maxnnn)
DOUBLE COMPLEX, INTENT(IN)    :: r_n(natoms)
DOUBLE COMPLEX, INTENT(INOUT) :: r_nm1(natoms)
DOUBLE COMPLEX                :: suma,sumb

DOUBLE COMPLEX                :: tmpsi,cpsi
INTEGER                       :: i,j
DOUBLE COMPLEX, PARAMETER     :: zero = (0.d0,0.d0)



suma = zero
sumb = zero

!$OMP PARALLEL SHARED(r_n,r_nm1,nn,nnn,near,nnear,e,s,ss) &
!$OMP PRIVATE(i,j,tmpsi,cpsi) &
!$OMP REDUCTION(+: suma,sumb)
!$OMP DO
DO i = 1,natoms

  cpsi  = e(i)*r_n(i) - 0.5d0*r_nm1(i)

  DO j = 1,near(i)
    cpsi = cpsi + s(i,j)*r_n(nn(i,j))
  ENDDO

  DO j = 1,nnear(i)
    cpsi = cpsi + ss(i,j)*r_n(nnn(i,j))
  ENDDO

  suma = suma + dconjg(cpsi)*r_n(i)
  sumb = sumb + dconjg(cpsi)*cpsi

  r_nm1(i) = 2.0d0*cpsi

ENDDO
!$OMP END DO
!$OMP END PARALLEL

suma = 4.0d0*suma
sumb = 8.0d0*sumb

END SUBROUTINE mv_2mu

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SUBROUTINE Update(natoms,r_n,r_nm1)

IMPLICIT NONE

INCLUDE 'omp_lib.h'

INTEGER, INTENT(IN)           :: natoms
DOUBLE COMPLEX, INTENT(INOUT) :: r_n(natoms),r_nm1(natoms)

INTEGER                       :: i
DOUBLE COMPLEX                :: tmpsi


!$OMP PARALLEL SHARED(r_n,r_nm1) &
!$OMP PRIVATE(tmpsi,i)
!$OMP DO
DO i = 1,natoms
  tmpsi    = r_nm1(i)
  r_nm1(i) = r_n(i)
  r_n(i)   = tmpsi
ENDDO
!$OMP END DO
!$OMP END PARALLEL



END SUBROUTINE Update

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

  ctemp = (0.0d0,-1.0d0)**DBLE(i-1) * DBESJN(i-1,T)
  ctemp = ctemp * CDEXP((0.0d0,-1.0d0)*bc/ac*T)
  ctemp = ctemp * 2.0d0
  IF(CDABS(ctemp) .lt. 1.0d-20) flag = 1
  i = i+1

ENDDO

npol = i
ALLOCATE(c(npol))
PRINT*,'  npol = ',npol


! Calculation of the Chebyshev expansion coefficients using Bessel functions
c(1) = CDEXP((0.d0,-1.d0)*bc/ac*T) * DBESJ0(T)
DO i = 2,npol

  c(i) = (0.0d0,-1.0d0)**DBLE(i-1) * DBESJN(i-1,T)
  c(i) = c(i) * CDEXP((0.0d0,-1.0d0)*bc/ac*T)
  c(i) = c(i) * 2.0d0

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



END MODULE operators
