MODULE utils
CONTAINS




!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Subroutine: init_random_seed()                                        !
! Description: create a random seed for generating a random phase state !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE init_random_seed(iseed)

IMPLICIT NONE

INTEGER              :: i, n, clock, iseed
INTEGER, ALLOCATABLE :: seed(:)




CALL RANDOM_SEED(size = n)
ALLOCATE(seed(n))

IF(iseed .eq. 0) THEN
  CALL SYSTEM_CLOCK(COUNT=clock)
ELSE
  clock = iseed
ENDIF
PRINT*,'clock: ',clock

OPEN(1,file='seed.txt',status='replace')
WRITE(1,*)clock
CLOSE(1)

seed = clock + 37 * (/ (i - 1, i = 1, n) /)
CALL RANDOM_SEED(PUT = seed)

DEALLOCATE(seed)




RETURN
END SUBROUTINE init_random_seed
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



subroutine checkingHermiticity(natoms, maxnn,nn,s, maxnnn,nnn,ss)
implicit none
integer, intent(in)        :: natoms,maxnn,maxnnn
integer, intent(in)        :: nn(natoms,maxnn),nnn(natoms,maxnnn)
double complex, intent(in) :: s(natoms,maxnn),ss(natoms,maxnnn)
integer :: i,j,k,it,jt,kt, count1,count2


count1 = 0
count2 = 0

write(*,*)'Checking hermiticity...'
do i=1,natoms
   it=i

   Vec1a: do j=1,maxnn
      jt=nn(i,j)
      if (jt/=0) then
        Vec1b: do k=1,maxnn
           if (nn(jt,k)==it) then
              kt=k; exit Vec1b
           end if
        end do Vec1b
        if ( s(it,j) /= dconjg(s(jt,kt)) ) then
           write(*,*) 'Error en Hermiticity 1', it,j,jt,kt
           write(*,*) 'sitio it:',it
           write(*,'(6i10)') nn(it,:)
           write(*,'(6f10.4)') s(it,:)
           write(*,*) 'sitio jt:',jt
           write(*,'(6i10)') nn(jt,:)
           write(*,'(6f10.4)') s(jt,:)
           write(*,*) 'ratio: ',DREAL(s(it,j))/DREAL(s(jt,kt))
           write(*,*)
           count1 = count1 + 1
        end if
      else
        cycle Vec1a
      end if
   end do Vec1a

   Vec2a: do j=1,maxnnn
      jt=nnn(i,j)
      if (jt/=0) then
        Vec2b: do k=1,maxnnn
           if (nnn(jt,k)==it) then
              kt=k; exit Vec2b
           end if
        end do Vec2b
        if ( ss(it,j) /= dconjg(ss(jt,kt)) ) then
           write(*,*) 'Error en Hermiticity 2', it,j,jt,kt
           write(*,*) 'sitio it:',it
           write(*,'(6i10)') nnn(it,:)
           write(*,'(6f10.4)') ss(it,:)
           write(*,*) 'sitio jt:',jt
           write(*,'(6i10)') nnn(jt,:)
           write(*,'(6f10.4)') ss(jt,:)
           write(*,*) 'ratio: ',DREAL(ss(it,j))/DREAL(ss(jt,kt))
           write(*,*)
           count2 = count2 + 1
        end if
      else
        cycle Vec2a
      end if
   end do Vec2a

end do
PRINT*,'# of 1-nn errors: ',count1
PRINT*,'# of 2-nn errors: ',count2
write(*,*)'...done'
end subroutine checkingHermiticity



END MODULE utils
