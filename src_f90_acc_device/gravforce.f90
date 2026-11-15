      subroutine GravForce
      use basicmod
      use fluxmod
      implicit none
      integer :: i,j,k,n

!$acc kernels
!$acc loop collapse(3) independent
      do k=ks,ke
      do j=js,je
      do i=is,ie+1
         gp1a(i  ,j,k) = gp(i,j,k) &
     & - 0.5d0*(gp(i  ,j,k)-gp(i-1,j,k))

         gp1a(i+1,j,k) = gp(i,j,k) &
     & + 0.5d0*(gp(i+1,j,k)-gp(i  ,j,k))

       grvsrc1(i,j,k) = (gp1a(i+1,j,k)-gp1a(i,j,k))/(x1a(i+1)-x1a(i))*d(i,j,k)

      enddo
      enddo
      enddo
!$acc end kernels
      
!$acc kernels
!$acc loop collapse(3) independent
      do k=ks,ke
      do i=is,ie
      do j=js,je+1
       grvsrc2(i,j,k) = 0.0d0
      enddo
      enddo
      enddo
!$acc end kernels
      
!$acc kernels
!$acc loop collapse(3) independent
      do j=js,je
      do i=is,ie
      do k=ks,ke+1
         grvsrc3(i,j,k) = 0.0d0
      enddo
      enddo
      enddo
!$acc end kernels
      return
      end subroutine  GravForce
