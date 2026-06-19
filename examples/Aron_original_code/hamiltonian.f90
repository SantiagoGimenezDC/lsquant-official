MODULE hamiltonian
CONTAINS



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: make_H                                                   !
! Description: reads input files for simulation and sample information !
!              finds nearest neighbors                                 !
!              repeats supercell                                       !
!              calculates the system Hamiltonian                       !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE make_H(natoms,maxnn,maxnnn, nn,nnn,near,nnear, e,s,ss, dx,dy,ddx,ddy)

USE constants
USE simulation
USE utils

IMPLICIT NONE

! Subroutine outputs
INTEGER                       :: natoms,maxnn,maxnnn
INTEGER,          ALLOCATABLE :: nn(:,:),near(:), nnn(:,:),nnear(:),ncom(:,:)
DOUBLE PRECISION, ALLOCATABLE :: e(:), dx(:,:),dy(:,:), ddx(:,:),ddy(:,:)
DOUBLE COMPLEX,   ALLOCATABLE :: s(:,:), ss(:,:)

! Sample information
INTEGER                       :: nuc
DOUBLE PRECISION              :: A1x,A1y,lA1, A2x,A2y,lA2, acc, ncutoff

! Atomic position arrays
DOUBLE PRECISION, ALLOCATABLE :: x(:), y(:)
INTEGER,          ALLOCATABLE :: cells(:,:,:)

! Electron-hole puddles
INTEGER                       :: neh
INTEGER,          ALLOCATABLE :: ieh(:)

! Random numbers
DOUBLE PRECISION              :: randf

! Loop index variables
INTEGER                       :: i,iatom,irow,icol




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Read in unit cell's atomic positions and determine its nearest neighbors !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PRINT*
PRINT*,'Reading in atom positions and getting nearest neighbor data...'
PRINT*
OPEN(1,file='unitcell.txt',status='old')
READ(1,*)A1x,A1y   ! First lattice vector
READ(1,*)A2x,A2y   ! Second lattice vector
READ(1,*)acc
READ(1,*)ncutoff
READ(1,*)nuc
natoms = nuc*nrow*ncol
PRINT*,'  natoms:      ',natoms
PRINT*



! Allocate arrays that depend on the number of atoms
ALLOCATE(x(natoms))
ALLOCATE(y(natoms))
ALLOCATE(cells(nrow,ncol,nuc))
x = 0.0d0
y = 0.0d0
cells = 0



! Read in the atomic coordinates, and repeat the supercell
DO i = 1,nuc
  READ(1,*)x(i),y(i)

  DO irow = 1,nrow
  DO icol = 1,ncol

    iatom = i + nuc*((irow-1)*ncol + (icol-1))

    ! Positions of repeated unit cells
    x(iatom) =  x(i) + DBLE(icol-1)*A1x + DBLE(irow-1)*A2x
    y(iatom) =  y(i) + DBLE(icol-1)*A1y + DBLE(irow-1)*A2y

    ! Fill cells array for nn finding
    cells(irow,icol,i) = iatom

  ENDDO
  ENDDO

ENDDO
CLOSE(1)



! Now scale up the lattice vectors to the full supercell
A1x = DBLE(ncol)*A1x
A1y = DBLE(ncol)*A1y
A2x = DBLE(nrow)*A2x
A2y = DBLE(nrow)*A2y



! Rewrite the lattice vectors as a magnitude and a unit vector
lA1 = DSQRT(A1x**2.0d0 + A1y**2.0d0)
A1x = A1x/lA1
A1y = A1y/lA1
lA2 = DSQRT(A2x**2.0d0 + A2y**2.0d0)
A2x = A2x/lA2
A2y = A2y/lA2



! Find max number of nearest neighbors
CALL find_max_nn(natoms, x,y, acc,ncutoff, nrow,ncol,nuc, cells, &
                 A1x,A1y,A2x,A2y,lA1,lA2, &
                 maxnn)
PRINT*
PRINT*,'  maxnn:       ',maxnn

! Find nearest neighbors and fill the nn array
ALLOCATE(near(2*natoms))
ALLOCATE(nn(2*natoms,2*maxnn))
ALLOCATE(dx(2*natoms,2*maxnn))
ALLOCATE(dy(2*natoms,2*maxnn))
near = 0
nn = 0
dx = 0.0d0
dy = 0.0d0
CALL find_nn(natoms,maxnn, x,y, acc,ncutoff, nrow,ncol,nuc, cells, &
             A1x,A1y,A2x,A2y,lA1,lA2, &
             nn(1:natoms,1:maxnn),near(1:natoms), &
             dx(1:natoms,1:maxnn),dy(1:natoms,1:maxnn))


! Find max number of next-nearest neighbors
CALL find_max_nnn(natoms,maxnn, nn(1:natoms,1:maxnn),near(1:natoms), maxnnn)
PRINT*,'  maxnnn:      ',maxnnn
PRINT*

! Find next-nearest neighbors and fill the nnn array
ALLOCATE(nnear(2*natoms))
ALLOCATE(nnn(2*natoms,2*maxnnn))
ALLOCATE(ddx(2*natoms,2*maxnnn))
ALLOCATE(ddy(2*natoms,2*maxnnn))
ALLOCATE(ncom(natoms,maxnnn))
nnear = 0
nnn = 0
ddx = 0.0d0
ddy = 0.0d0
ncom = 0
CALL find_nnn(natoms,maxnn,maxnnn, nn(1:natoms,1:maxnn),near(1:natoms), &
              x,y, A1x,A1y,A2x,A2y,lA1,lA2, &
              nnn(1:natoms,1:maxnnn),nnear(1:natoms),ncom, ddx(1:natoms,1:maxnnn),ddy(1:natoms,1:maxnnn))
PRINT*,'...done'
PRINT*
PRINT*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copy spin-up to spin-down !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PRINT*
PRINT*,'Copying spin-up to spin-down...'


! Copy spin-up to spin-down and cross-spin terms
DO i = 1,natoms

  near(natoms+i)  = near(i)  ! near down = near up
  nnear(natoms+i) = nnear(i) ! nnear down = nnear up

  nn(       i , near(i)+1:2*near(i)) = nn(i,1:near(i)) + natoms        ! nn, up-down   = up-up + natoms
  nn(natoms+i ,         1:  near(i)) = nn(i,1:near(i)) + natoms        ! nn, down-down = up-up + natoms
  nn(natoms+i , near(i)+1:2*near(i)) = nn(i,1:near(i))                 ! nn, down-up   = up-up

  nnn(       i , nnear(i)+1:2*nnear(i)) = nnn(i,1:nnear(i)) + natoms   ! nnn, up-down   = up-up + natoms
  nnn(natoms+i ,          1:  nnear(i)) = nnn(i,1:nnear(i)) + natoms   ! nnn, down-down = up-up + natoms
  nnn(natoms+i , nnear(i)+1:2*nnear(i)) = nnn(i,1:nnear(i))            ! nnn, down-up   = up-up  

  dx(       i , near(i)+1:2*near(i)) = dx(i,1:near(i))                 ! dx, up-down   = up-up
  dx(natoms+i ,         1:  near(i)) = dx(i,1:near(i))                 ! dx, down-down = up-up
  dx(natoms+i , near(i)+1:2*near(i)) = dx(i,1:near(i))                 ! dx, down-up   = up-up

  ddx(       i , nnear(i)+1:2*nnear(i)) = ddx(i,1:nnear(i))            ! ddx, up-down   = up-up
  ddx(natoms+i ,          1:  nnear(i)) = ddx(i,1:nnear(i))            ! ddx, down-down = up-up
  ddx(natoms+i , nnear(i)+1:2*nnear(i)) = ddx(i,1:nnear(i))            ! ddx, down-up   = up-up
 
  dy(       i , near(i)+1:2*near(i)) = dy(i,1:near(i))                 ! dy, up-down   = up-up
  dy(natoms+i ,         1:  near(i)) = dy(i,1:near(i))                 ! dy, down-down = up-up
  dy(natoms+i , near(i)+1:2*near(i)) = dy(i,1:near(i))                 ! dy, down-up   = up-up

  ddy(       i , nnear(i)+1:2*nnear(i)) = ddy(i,1:nnear(i))            ! ddy, up-down   = up-up
  ddy(natoms+i ,          1:  nnear(i)) = ddy(i,1:nnear(i))            ! ddy, down-down = up-up
  ddy(natoms+i , nnear(i)+1:2*nnear(i)) = ddy(i,1:nnear(i))            ! ddy, down-up   = up-up

ENDDO



PRINT*,'...done'
PRINT*
PRINT*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 5: determine positions of the electron-hole puddles !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PRINT*
PRINT*,'Finding electron-hole puddles...'
neh = DBLE(natoms)*peh
ALLOCATE(ieh(2*neh))
ieh = 0

neh = 0
DO i = 1,natoms
  CALL random_number(randf)
  IF(randf .le. peh) THEN
    neh = neh + 1
    ieh(neh) = i
  ENDIF
ENDDO
PRINT*
PRINT*,'  Number of puddles: ',neh
PRINT*
PRINT*,'...done'
PRINT*
PRINT*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Step 6: Calculate the Hamiltonian based on the sample information !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PRINT*
PRINT*,'Building the Hamiltonian...'
PRINT*
ALLOCATE(e(2*natoms))
ALLOCATE(s(2*natoms,2*maxnn))
ALLOCATE(ss(2*natoms,2*maxnnn))
CALL geom_hamilt(natoms,maxnn,maxnnn, nn,nnn,near,nnear,ncom, &
                 acc, A1x,A1y,A2x,A2y,lA1,lA2, x,y, &
                 teV,teV2,taniso, W, neh,ieh,Weh,leh, &
                 Delta,VI_A,VI_B,VR,VRaniso,VRin,VPIA_A,VPIA_B, &
                 e,s,ss)
PRINT*
PRINT*,'...done'
PRINT*
PRINT*

! Increase size of nn and nnn count to account for off-diagonal terms
near  = 2*near
nnear = 2*nnear
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



! Deallocate yo mama
DEALLOCATE(x)
DEALLOCATE(y)
DEALLOCATE(cells)
DEALLOCATE(ncom)
DEALLOCATE(ieh)




RETURN
END SUBROUTINE make_h
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: find_max_nn                                                        !
! Description: calculate max number of nearest neighbors for an atom in a sample !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE find_max_nn(natoms, x,y, acc,ncutoff, nrow,ncol,nuc, cells, &
                       A1x,A1y,A2x,A2y,lA1,lA2, &
                       maxnn)

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms,nrow,ncol,nuc, cells(nrow,ncol,nuc)
DOUBLE PRECISION :: x(natoms),y(natoms), acc,ncutoff
DOUBLE PRECISION :: A1x,A1y,A2x,A2y,lA1,lA2

! Subroutine outputs
INTEGER :: maxnn

! For finding and counting nearest neighbors
DOUBLE PRECISION :: ncutoff2, distx,disty,dist2
INTEGER          :: nncount

! For looping over atoms
INTEGER :: irown,icoln, irowm,icolm, i,j,n,m, addrow,addcol




! Initialize maxnn, set cutoff to cutoff² to avoid using sqrt()
maxnn   = 0
ncutoff2 = ncutoff*ncutoff*acc*acc


! TEST: write out position of atoms without 3 neighbors
!OPEN(1,file='xy1.txt',status='replace')
!OPEN(2,file='xy2.txt',status='replace')


! Find nearest neighbors by searching atoms in current and adjacent cells
! The first three DO loops cover all atoms
DO irown = 1,nrow
DO icoln = 1,ncol
DO i = 1,nuc

  ! Current atom
  n = i + nuc*((irown-1)*ncol + (icoln-1))
  nncount = 0

  ! Search adjacent cells for neighbors
  DO addrow = -2,2
  DO addcol = -2,2
    irowm = irown + addrow
    icolm = icoln + addcol

    ! Periodic boundary conditions here
    IF(irowm .gt. nrow) irowm = irowm-nrow
    IF(irowm .lt. 1)    irowm = irowm+nrow
    IF(icolm .gt. ncol) icolm = icolm-ncol
    IF(icolm .lt. 1)    icolm = icolm+ncol

    ! Examine all atoms in current cell
    DO j = 1,nuc
      m = cells(irowm,icolm,j)
      IF (m .ne. n) THEN
        distx = x(n) - x(m)
        disty = y(n) - y(m)

        ! Account for periodic boundary conditions here as well
        CALL fix_dist_pbc(distx,disty, A1x,A1y,A2x,A2y,lA1,lA2)

        ! Get distance and count neighbor if it's close enough
        dist2 = distx**2.0d0 + disty**2.0d0
        IF (dist2 .lt. ncutoff2) nncount = nncount + 1
      ENDIF

    ENDDO

  ENDDO
  ENDDO
  IF(nncount .gt. maxnn) maxnn = nncount

  !IF(nncount .eq. 1) WRITE(1,*)x(n),y(n)
  !IF(nncount .eq. 2) WRITE(2,*)x(n),y(n)

ENDDO
ENDDO
ENDDO

!CLOSE(1)
!CLOSE(2)


RETURN
END SUBROUTINE find_max_nn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: find_max_nnn                                                       !
! Description: calculate max number of next-nearest neighbors                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE find_max_nnn(natoms,maxnn, nn,near, maxnnn)

IMPLICIT NONE

! Subroutine inputs
INTEGER :: natoms,maxnn, nn(natoms,maxnn),near(natoms)

! Subroutine outputs
INTEGER :: maxnnn

! For finding and counting nearest neighbors
INTEGER :: nnncount

! Loop index variables
INTEGER :: i,j,k,l,m




! Initialize maxnnn
maxnnn = 0


! Loop over each atom
DO i = 1,natoms

  nnncount = 0

  ! Loop over the atom's nearest neighbors
  DO j = 1,near(i)

    ! Loop over the neighbors of the atom's neighbors
    l = nn(i,j)
    DO k = 1,near(l)
      m = nn(l,k)

      ! We've found a second neighbor, as long as it's not itself
      IF(m .ne. i) nnncount = nnncount + 1
    ENDDO

  ENDDO

  IF (nnncount .gt. maxnnn) maxnnn = nnncount

ENDDO



RETURN
END SUBROUTINE find_max_nnn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: find_nn                                                                       !
! Description: calculate nearest neighbors of a sample with a fast method that uses binning !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE find_nn(natoms,maxnn, x,y, acc,ncutoff, nrow,ncol,nuc, cells, &
                   A1x,A1y,A2x,A2y,lA1,lA2, &
                   nn,near,dx,dy)

IMPLICIT NONE

! Subroutine inputs
INTEGER              :: natoms,maxnn, nrow,ncol,nuc, cells(nrow,ncol,nuc)
DOUBLE PRECISION     :: x(natoms),y(natoms), acc,ncutoff
DOUBLE PRECISION     :: A1x,A1y,A2x,A2y,lA1,lA2

! Subroutine outputs
INTEGER              :: nn(natoms,maxnn), near(natoms)
DOUBLE PRECISION     :: dx(natoms,maxnn), dy(natoms,maxnn)

! For finding and counting nearest neighbors
INTEGER              :: num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10
INTEGER              :: nncount, navg
DOUBLE PRECISION     :: distx,disty,dist2,dist_avg, ncutoff2

! For looping over atoms
INTEGER :: irown,icoln, irowm,icolm, i,j,n,m, addrow,addcol




! Initialize neighbor counts, set cutoff to cutoff² to avoid using sqrt()
num0    = 0
num1    = 0
num2    = 0
num3    = 0
num4    = 0
num5    = 0
num6    = 0
num7    = 0
num8    = 0
num9    = 0
num10   = 0

ncutoff2 = ncutoff*ncutoff*acc*acc



! Find nearest neighbors by searching atoms in current and adjacent cells
! The first three DO loops cover all atoms
dist_avg = 0.0d0
navg = 0
DO irown = 1,nrow
DO icoln = 1,ncol
DO i = 1,nuc

  ! Current atom
  n = i + nuc*((irown-1)*ncol + (icoln-1))
  nncount = 0

  ! Search adjacent cells for neighbors
  DO addrow = -2,2
  DO addcol = -2,2
    irowm = irown + addrow
    icolm = icoln + addcol

    ! Periodic boundary conditions here
    IF(irowm .gt. nrow) irowm = irowm-nrow
    IF(irowm .lt. 1)    irowm = irowm+nrow
    IF(icolm .gt. ncol) icolm = icolm-ncol
    IF(icolm .lt. 1)    icolm = icolm+ncol

    ! Examine all atoms in current cell
    DO j = 1,nuc
      m = cells(irowm,icolm,j)
      IF (m .ne. n) THEN
        distx = x(n) - x(m)
        disty = y(n) - y(m)

        ! Account for periodic boundary conditions here as well
        CALL fix_dist_pbc(distx,disty, A1x,A1y,A2x,A2y,lA1,lA2)

        ! Get distance and count neighbor if it's close enough
        dist2 = distx**2.0d0 + disty**2.0d0
        IF (dist2 .lt. ncutoff2) THEN
          nncount = nncount + 1
          near(n) = nncount
          nn(n,nncount) = m
          dx(n,nncount) = distx
          dy(n,nncount) = disty

          dist_avg = dist_avg + DSQRT(dist2)
          navg = navg+1
        ENDIF
      ENDIF

    ENDDO

  ENDDO
  ENDDO

! Count the number of atoms with 1, 2, 3, 4, 5 nearest neighbors
  IF(nncount .eq. 0) num0 = num0 + 1
  IF(nncount .eq. 1) num1 = num1 + 1
  IF(nncount .eq. 2) num2 = num2 + 1
  IF(nncount .eq. 3) num3 = num3 + 1
  IF(nncount .eq. 4) num4 = num4 + 1
  IF(nncount .eq. 5) num5 = num5 + 1
  IF(nncount .eq. 6) num6 = num6 + 1
  IF(nncount .eq. 7) num7 = num7 + 1
  IF(nncount .eq. 8) num8 = num8 + 1
  IF(nncount .eq. 9) num9 = num9 + 1
  IF(nncount .eq. 10) num10 = num10 + 1

ENDDO
ENDDO
ENDDO


PRINT*
PRINT*,'  NUMBER OF NEAREST NEIGHBOR ATOMS:'
PRINT*
PRINT*,'  0 neighbors: ',num0
PRINT*,'  1 neighbors: ',num1
PRINT*,'  2 neighbors: ',num2
PRINT*,'  3 neighbors: ',num3
PRINT*,'  4 neighbors: ',num4
PRINT*,'  5 neighbors: ',num5
PRINT*,'  6 neighbors: ',num6
PRINT*,'  7 neighbors: ',num7
PRINT*,'  8 neighbors: ',num8
PRINT*,'  9 neighbors: ',num9
PRINT*,' 10 neighbors: ',num10
PRINT*,'>10 neighbors: ',natoms-(num0+num1+num2+num3+num4+num5+num6+num7+num8+num9+num10)
PRINT*


PRINT*,'  Average nn distance: ',dist_avg/DBLE(navg)



RETURN
END SUBROUTINE find_nn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: find_nnn                                      !
! Description: find the next-nearest neighbors of each atom !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE find_nnn(natoms,maxnn,maxnnn, nn,near, &
                    x,y, A1x,A1y,A2x,A2y,lA1,lA2, &
                    nnn,nnear,ncom, ddx,ddy)

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms,maxnn,maxnnn, nn(natoms,maxnn),near(natoms)
DOUBLE PRECISION :: x(natoms),y(natoms), A1x,A1y,A2x,A2y,lA1,lA2

! Subroutine outputs
INTEGER          :: nnn(natoms,maxnnn),nnear(natoms),ncom(natoms,maxnnn)
DOUBLE PRECISION :: ddx(natoms,maxnnn),ddy(natoms,maxnnn)

! For finding and counting nearest neighbors
INTEGER          :: nnncount, flag, navg
INTEGER          :: num0,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10
DOUBLE PRECISION :: distx,disty,dist_avg

! Loop index variables
INTEGER :: i,j,k,l,m,n, ii,jj



! Initialize counting stats
num0    = 0
num1    = 0
num2    = 0
num3    = 0
num4    = 0
num5    = 0
num6    = 0
num7    = 0
num8    = 0
num9    = 0
num10   = 0



! Loop over each atom
dist_avg = 0.0d0
navg = 0
DO i = 1,natoms

  nnncount = 0

  ! Loop over the atom's nearest neighbors
  DO j = 1,near(i)

    ! Loop over the neighbors of the atom's neighbors
    l = nn(i,j)
    DO k = 1,near(l)

      ! We've found a next-nearest neighbor, as long as it's not itself
      m = nn(l,k)
      IF(m .ne. i) THEN

        ! Make sure the next-nearest neighbor wasn't already counted (this can happen near grain boundaries)
        flag = 0
        DO n = 1,nnncount
          IF(m .eq. nnn(i,n)) flag = 1
        ENDDO

        ! We found a new next-nearest neighbor
        IF(flag .eq. 0) THEN
          ! Store the index of the neighbor
          nnncount = nnncount + 1
          nnear(i) = nnncount
          nnn(i,nnncount) = m

          ! Find the nnn distance
          distx = x(i) - x(m)
          disty = y(i) - y(m)
          CALL fix_dist_pbc(distx,disty, A1x,A1y,A2x,A2y,lA1,lA2)
          ddx(i,nnncount) = distx
          ddy(i,nnncount) = disty

          dist_avg = dist_avg + DSQRT(distx*distx + disty*disty)
          navg = navg + 1
        ENDIF

      ENDIF

    ENDDO
  ENDDO

! Count the number of atoms with 1, 2, 3, 4, 5 nearest neighbors
  IF(nnncount .eq.  0) num0  = num0  + 1
  IF(nnncount .eq.  1) num1  = num1  + 1
  IF(nnncount .eq.  2) num2  = num2  + 1
  IF(nnncount .eq.  3) num3  = num3  + 1
  IF(nnncount .eq.  4) num4  = num4  + 1
  IF(nnncount .eq.  5) num5  = num5  + 1
  IF(nnncount .eq.  6) num6  = num6  + 1
  IF(nnncount .eq.  7) num7  = num7  + 1
  IF(nnncount .eq.  8) num8  = num8  + 1
  IF(nnncount .eq.  9) num9  = num9  + 1
  IF(nnncount .eq. 10) num10 = num10 + 1

ENDDO



PRINT*
PRINT*,'  NUMBER OF NEXT-NEAREST NEIGHBOR ATOMS:'
PRINT*
PRINT*,'  0 neighbors: ',num0
PRINT*,'  1 neighbors: ',num1
PRINT*,'  2 neighbors: ',num2
PRINT*,'  3 neighbors: ',num3
PRINT*,'  4 neighbors: ',num4
PRINT*,'  5 neighbors: ',num5
PRINT*,'  6 neighbors: ',num6
PRINT*,'  7 neighbors: ',num7
PRINT*,'  8 neighbors: ',num8
PRINT*,'  9 neighbors: ',num9
PRINT*,' 10 neighbors: ',num10
PRINT*,'>10 neighbors: ',natoms-(num0+num1+num2+num3+num4+num5+num6+num7+num8+num9+num10)
PRINT*

PRINT*,'  Average nnn distance: ',dist_avg/DBLE(navg)


! Calculate the common neighbor site of two second neighbors
DO i = 1,natoms
DO j = 1,nnear(i)

  k = nnn(i,j)   

  DO ii = 1,near(i)
  DO jj = 1,near(k)

    IF (nn(i,ii) .eq. nn(k,jj)) ncom(i,j) = nn(i,ii)

  ENDDO
  ENDDO

ENDDO
ENDDO




RETURN
END SUBROUTINE find_nnn
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: geom_hamilt()                                                      !
! Description: calculates the sample Hamiltonian based on the sample structure   !
!              also calculates the distance between nearest neighbors, dx and dy !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE geom_hamilt(natoms,maxnn,maxnnn, nn,nnn,near,nnear,ncom, &
                       acc, A1x,A1y,A2x,A2y,lA1,lA2, x,y, &
                       teV,teV2,taniso, W, neh,ieh,Weh,leh, &
                       Delta,VI_A,VI_B,VR,VRaniso,VRin,VPIA_A,VPIA_B, &
                       e,s,ss)

USE constants

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms,maxnn,maxnnn
INTEGER          :: nn(2*natoms,2*maxnn), near(2*natoms)
INTEGER          :: nnn(2*natoms,2*maxnnn), nnear(2*natoms), ncom(natoms,maxnnn)
DOUBLE PRECISION :: acc, A1x,A1y,A2x,A2y,lA1,lA2
DOUBLE PRECISION :: x(natoms), y(natoms) 
DOUBLE PRECISION :: teV,teV2,taniso
DOUBLE PRECISION :: W
INTEGER          :: neh,ieh(neh)
DOUBLE PRECISION :: Weh,leh
DOUBLE PRECISION :: Delta,VI_A,VI_B,VR,VRaniso,VRin,VPIA_A,VPIA_B

! Subroutine outputs
DOUBLE PRECISION :: e(2*natoms)
DOUBLE COMPLEX   :: s(2*natoms,2*maxnn),ss(2*natoms,2*maxnnn)

! Distance between neighbors
DOUBLE PRECISION              :: dx,dy
DOUBLE PRECISION              :: dist,dist2,dist_avg,dist2_avg

! Terms for SOC
DOUBLE PRECISION              :: VI,VPIA, nu(3)

! For including disorder
DOUBLE PRECISION              :: randf

! Loop index variables
INTEGER                       :: i,j,jj,k,l,ll




! Use the nearest-neighbor data to build the Hamiltonian
dist_avg = 0.0d0
dist2_avg = 0.0d0
l = 0
ll = 0
!OPEN(1,file='xy0.txt',status='replace')
DO i = 1,natoms

  ! Set the sublattice-dependent terms of the Hamiltonian
  IF(MOD(i,2).eq.1) THEN
    e(       i) =  Delta
    e(natoms+i) =  Delta
    VI          =  VI_A
    VPIA        =  VPIA_A
  ELSE
    e(       i) = -Delta
    e(natoms+i) = -Delta
    VI          =  VI_B
    VPIA        =  VPIA_B
  ENDIF


  ! Nearest neighbor terms in the Hamiltonian
  DO j = 1,near(i)

    ! TEST: look for specific pairs of atoms in the unit cell
    !IF(MOD(i,8).eq.3 .and. MOD(nn(i,j),8).eq.4) WRITE(1,*)x(i),y(i)
    !IF(MOD(i,8).eq.4 .and. MOD(nn(i,j),8).eq.3) WRITE(1,*)x(i),y(i)

    ! Get nearest neighbor distance
    dx = x(nn(i,j)) - x(i)
    dy = y(nn(i,j)) - y(i)
    CALL fix_dist_pbc(dx,dy, A1x,A1y,A2x,A2y,lA1,lA2)

    ! Hopping between nearest neighbors
    ! Include hopping anisotropy
    IF((MOD(i,8).eq.3 .and. MOD(nn(i,j),8).eq.4) .or. &
       (MOD(i,8).eq.4 .and. MOD(nn(i,j),8).eq.3)) THEN
      s(       i , j) = teV * (12.0d0-11.0d0*taniso)   ! up-up
      s(natoms+i , j) = teV * (12.0d0-11.0d0*taniso)   ! down-down
    ELSE
      s(       i , j) = teV * taniso   ! up-up
      s(natoms+i , j) = teV * taniso   ! down-down   
    ENDIF

    ! Rashba spin-orbit coupling -- involves nearest neighbors in off-diagonal blocks
    s(       i , near(i)+j) = -2.0d0/3.0d0*(0.0d0,1.0d0) * VR * (dy + (0.0d0,1.0d0)*dx)/acc   ! up-down 
    s(natoms+i , near(i)+j) = -2.0d0/3.0d0*(0.0d0,1.0d0) * VR * (dy - (0.0d0,1.0d0)*dx)/acc   ! down-up 

    ! Include anisotropy of Rashba along bond 3-4
    IF((MOD(i,8).eq.3 .and. MOD(nn(i,j),8).eq.4) .or. &
       (MOD(i,8).eq.4 .and. MOD(nn(i,j),8).eq.3)) THEN
      s(       i , near(i)+j) = s(       i , near(i)+j) * (12.0d0-11.0d0*VRaniso)
      s(natoms+i , near(i)+j) = s(natoms+i , near(i)+j) * (12.0d0-11.0d0*VRaniso)
    ELSE
      s(       i , near(i)+j) = s(       i , near(i)+j) * VRaniso
      s(natoms+i , near(i)+j) = s(natoms+i , near(i)+j) * VRaniso
    ENDIF

    ! Rashba from in-plane field across bond 3-4
    IF(MOD(i,8).eq.3 .and. MOD(nn(i,j),8).eq.4) THEN
      s(       i , j) = s(       i , j) + 4.0d0*(0.0d0,1.0d0) * VRin
      s(natoms+i , j) = s(natoms+i , j) - 4.0d0*(0.0d0,1.0d0) * VRin
    ENDIF
    IF(MOD(i,8).eq.4 .and. MOD(nn(i,j),8).eq.3) THEN
      s(       i , j) = s(       i , j) - 4.0d0*(0.0d0,1.0d0) * VRin
      s(natoms+i , j) = s(natoms+i , j) + 4.0d0*(0.0d0,1.0d0) * VRin
    ENDIF

  ENDDO



  ! Next nearest neighbor terms in the Hamiltonian
  DO jj = 1,nnear(i)

    ! Get nearest neighbor distance
    dx = x(nnn(i,jj)) - x(i)
    dy = y(nnn(i,jj)) - y(i)
    CALL fix_dist_pbc(dx,dy, A1x,A1y,A2x,A2y,lA1,lA2)

    ! Including Kane-Mele terms in the Hamiltonian
    CALL Kane_Mele(x,y,0.0d0, A1x,A1y,A2x,A2y,lA1,lA2, natoms,i,nnn(i,jj),ncom(i,jj),nu)
    IF (ncom(i,jj).eq.0) THEN
      PRINT*,' ERROR IN ncom:  ','i =  ',i,'  jj =  ',jj
    ENDIF

    ! Hopping between second neighbors
    ss(       i , jj) = teV2  ! Up-up
    ss(natoms+i , jj) = teV2  ! Down-down

    ! Kane-Mele spin-orbit coupling -- involves next nearest neighbors in diagonal blocks (Graphene)
    ss(       i , jj) = ss(       i , jj) + 2.0d0/9.0d0 * VI * nu(3)*(0.0d0,1.0d0)   ! up-up 
    ss(natoms+i , jj) = ss(natoms+i , jj) - 2.0d0/9.0d0 * VI * nu(3)*(0.0d0,1.0d0)   ! down-down
  
  ENDDO


ENDDO
!CLOSE(1)



! Include Anderson disorder everywhere
DO i = 1,natoms
  CALL random_number(randf)
  e(       i) = e(       i) + W*2.0d0*(randf-0.5d0)
  e(natoms+i) = e(natoms+i) + W*2.0d0*(randf-0.5d0)
ENDDO



! Include electron-hole puddles
DO j = 1,neh
  CALL random_number(randf)
  DO i = 1,natoms
    dx = x(i) - x(ieh(j))
    dy = y(i) - y(ieh(j))
    CALL fix_dist_pbc(dx,dy, A1x,A1y,A2x,A2y,lA1,lA2)
    dist2 = dx*dx + dy*dy

    e(       i) = e(       i) + Weh*2.0d0*(randf-0.5d0) * DEXP(-0.5d0*dist2/leh**2.0d0)
    e(natoms+i) = e(natoms+i) + Weh*2.0d0*(randf-0.5d0) * DEXP(-0.5d0*dist2/leh**2.0d0)
  ENDDO
ENDDO



! TEST: write out xy position and on-site energies
!OPEN(1,file='xy.txt',status='replace')
!DO i = 1,natoms
!  WRITE(1,*)x(i),y(i),e(i)
!ENDDO
!CLOSE(1)



!! TEST: write out part of the Hamiltonian to a file
!OPEN(1,file='nn.txt',status='replace')
!OPEN(2,file='nnn.txt',status='replace')
!OPEN(3,file='s.txt',status='replace')
!OPEN(4,file='ss.txt',status='replace')
!DO i = 1,10
!  WRITE(1,*)nn(i,:)
!  WRITE(2,*)nnn(i,:)
!  WRITE(3,*)s(i,:)
!  WRITE(4,*)ss(i,:)
!ENDDO
!CLOSE(1)
!CLOSE(2)
!CLOSE(3)
!CLOSE(4)



PRINT*
PRINT*,'  VI_A:    ',VI_A
PRINT*,'  VI_B:    ',VI_B
PRINT*,'  VR:      ',VR
PRINT*,'  VRaniso: ',VRaniso
PRINT*,'  VRin:    ',VRin
PRINT*,'  VPIA_A:  ',VPIA_A
PRINT*,'  VPIA_B:  ',VPIA_B



RETURN
END SUBROUTINE geom_hamilt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: Kane_Mele()                                                          !
! Description: calculates the instrinsic spin-orbit coupling using complex second  !
!              neighbor hoppings                                                   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE Kane_Mele(x,y,z, A1x,A1y,A2x,A2y,lA1,lA2, natoms,site,nnsite,nsite, nu)

IMPLICIT NONE

! Subroutine inputs
INTEGER          :: natoms,site,nnsite,nsite
DOUBLE PRECISION :: A1x,A1y,A2x,A2y,lA1,lA2
DOUBLE PRECISION :: x(natoms), y(natoms), z

! Subroutine outputs
DOUBLE PRECISION :: nu(3)

! Unitary vectors between neighbors
DOUBLE PRECISION :: v1(3),v2(3)


nu = 0.0d0

IF(nsite .ne. 0) THEN

  ! Computing vectors v_ik and v_kj
  v1(1) = x(nsite) - x(site)
  v1(2) = y(nsite) - y(site)
  v1(3) = 0.0d0
  CALL fix_dist_pbc(v1(1),v1(2), A1x,A1y,A2x,A2y,lA1,lA2)

  v2(1) = x(nnsite) - x(nsite)
  v2(2) = y(nnsite) - y(nsite)
  v2(3) = 0.0d0
  CALL fix_dist_pbc(v2(1),v2(2), A1x,A1y,A2x,A2y,lA1,lA2)

  ! Vector normalization
  v1 = v1 / DSQRT(v1(1)**2.0d0 + v1(2)**2.0d0 + v1(3)**2.0d0)
  v2 = v2 / DSQRT(v2(1)**2.0d0 + v2(2)**2.0d0 + v2(3)**2.0d0)

  ! Computing the cross-product to get vector nu_ij
  nu(1) = v1(2)*v2(3) - v2(2)*v1(3)
  nu(2) = v2(1)*v1(3) - v1(1)*v2(3)
  nu(3) = v1(1)*v2(2) - v2(1)*v1(2)

ENDIF


RETURN
END SUBROUTINE Kane_Mele
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: fix_dist_pbc()
! Description: when appropriate, shift distances by the lattice vectors to   !
!              account for neighboring atoms on opposite sides of the sample !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE fix_dist_pbc(distx,disty, A1x,A1y,A2x,A2y,lA1,lA2)

IMPLICIT NONE

DOUBLE PRECISION :: distx,disty, A1x,A1y,A2x,A2y,lA1,lA2
DOUBLE PRECISION :: projA1, projA2


! Project distance vector along lattice vectors:
! https://math.stackexchange.com/questions/148199/
! equation-for-non-orthogonal-projection-of-a-point-onto-two-vectors-representing
projA1 = -(A2x*disty - A2y*distx) / (A1x*A2y - A1y*A2x) / lA1
projA2 =  (A1x*disty - A1y*distx) / (A1x*A2y - A1y*A2x) / lA2

! If the projection length is large, then shift it by one lattice vector
IF(projA1 .gt.  0.5d0) THEN
  distx = distx - A1x*lA1
  disty = disty - A1y*lA1
ENDIF
IF(projA1 .lt. -0.5d0) THEN
  distx = distx + A1x*lA1
  disty = disty + A1y*lA1
ENDIF
IF(projA2 .gt.  0.5d0) THEN
  distx = distx - A2x*lA2
  disty = disty - A2y*lA2
ENDIF
IF(projA2 .lt. -0.5d0) THEN
  distx = distx + A2x*lA2
  disty = disty + A2y*lA2
ENDIF



END SUBROUTINE fix_dist_pbc
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




END MODULE hamiltonian
