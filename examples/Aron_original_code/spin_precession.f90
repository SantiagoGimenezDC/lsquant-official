!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Program: spin_precession                                                                 !
! Description: calculate semiclassical spin precession through an energy landscape         !
!              of electron-hole puddles, using parameters extracted from a                 !
!              tight-binding model of graphene with Rashba + intrinsic spin-orbit coupling !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM spin_precession

IMPLICIT NONE

INTEGER                       :: nC,neh, nx
DOUBLE PRECISION              :: pi,hbar,q, acc,rho, peh,Weh,Leh, dx,xmin,xmax,ymin,ymax, &
                                 d2, randf
DOUBLE PRECISION, ALLOCATABLE :: x(:), xeh(:),yeh(:), V(:)

INTEGER                       :: i,j, n,clock
INTEGER, ALLOCATABLE          :: seed(:)



! Seed the random number generator
CALL RANDOM_SEED(SIZE=n)
ALLOCATE(seed(n))
CALL SYSTEM_CLOCK(COUNT=clock)
seed = clock + 37 * [(i, i = 0,n-1)]
CALL RANDOM_SEED(PUT=seed)


! Some fundamental constants
pi   = 4.0d0*DATAN(1.0d0)   ! Pie
hbar = 6.58211928d-16       ! Planck's constant [eV-s]
q    = 1.602176565d-19      ! Electron charge [C]


! Some graphene constants
acc = 0.145d0                               ! Carbon-carbon distance [nm]
rho = 4.0d0 / (3.0d0**1.5d0 * acc**2.0d0)   ! Carbon atomic density [1/nm²]


! Define e-h puddles
peh = 0.0004d0   ! Density of puddles
Weh = 0.05d0     ! Height of puddles [eV]
Leh = 10.0d0     ! Size of puddles [nm]


! Define grid
xmin = -1000.0d0              ! Grid minimum [nm]
xmax =  1000.0d0              ! Grid maximum [nm]
dx   =     1.0d0              ! Grid discretization [nm]
nx   = NINT((xmax-xmin)/dx)   ! Number of grid points

ALLOCATE(x(0:nx))
DO i = 0,nx
  x(i) = xmin + DBLE(i)*dx
ENDDO


! Rectangular grid over which to place puddles
xmin = xmin - 10.0d0*Leh
xmax = xmax + 10.0d0*Leh
ymin = -10.0d0*Leh
ymax =  10.0d0*Leh


! Number of puddles
nC  = NINT((xmax-xmin)*(ymax-ymin)*rho)
neh = NINT(peh*DBLE(nC))
PRINT*,'nC:  ',nC
PRINT*,'neh: ',neh


! Random positions and heights of puddles
ALLOCATE(xeh(neh))
ALLOCATE(yeh(neh))
OPEN(1,file='xyeh.txt',status='replace')
DO i = 1,neh
  CALL random_number(randf)
  xeh(i) = (xmax-xmin)*randf + xmin

  CALL random_number(randf)
  yeh(i) = (ymax-ymin)*randf + ymin

  WRITE(1,*)xeh(i),yeh(i)
ENDDO
CLOSE(1)


! Calculate potential at each grid point
ALLOCATE(V(0:nx))
V = 0.0d0
DO j = 1,neh
  CALL random_number(randf)
  DO i = 0,nx
    d2 = (x(i)-xeh(j))**2.0d0 + yeh(j)**2.0d0
    V(i) = V(i) + Weh*2.0d0*(randf-0.5d0) * DEXP(-0.5d0*d2/Leh**2.0d0)
  ENDDO
ENDDO


! Write potential to a file
OPEN(1,file='Veh.txt',status='replace')
DO i = 0,nx
  WRITE(1,*)x(i),V(i)
ENDDO
CLOSE(1)





! Deallocate yo mama
DEALLOCATE(V)
DEALLOCATE(yeh)
DEALLOCATE(xeh)
DEALLOCATE(x)
DEALLOCATE(seed)



END PROGRAM spin_precession
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
