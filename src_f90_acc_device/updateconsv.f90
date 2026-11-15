      subroutine UpdateConsv
      use basicmod
      use fluxmod
#ifdef USE_CUDA_UC
      use cuda_kernels_f90
#endif
      implicit none
      integer::i,j,k

#ifdef USE_CUDA_UC
      call cuda_updateconsv()
#else
!$acc kernels
!$acc loop collapse(3) independent
      do k=ks,ke
      do j=js,je
      do i=is,ie
         
         d(i,j,k) = d(i,j,k)  &
     & +dt*(                  &
     &  (- nflux1(i+1,j,k,mden) &
     &   + nflux1(i,j,k,mden))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mden) &
     &   + nflux2(i,j,k,mden))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mden) &
     &   + nflux3(i,j,k,mden))/(x3a(k+1)-x3a(k)) &
     &      )

         mv1(i,j,k) = mv1(i,j,k) &
     & +dt*( &
     &      +  grvsrc1(i,j,k)                      &
     & +(- nflux1(i+1,j,k,mrv1) &
     &   + nflux1(i,j,k,mrv1))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mrv1) &
     &   + nflux2(i,j,k,mrv1))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mrv1) &
     &   + nflux3(i,j,k,mrv1))/(x3a(k+1)-x3a(k)) & 
     &      )
         mv2(i,j,k) = mv2(i,j,k) &
     & +dt*( &
     &      +  grvsrc2(i,j,k)                      &
     & +(- nflux1(i+1,j,k,mrv2) &
     &   + nflux1(i,j,k,mrv2))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mrv2) &
     &   + nflux2(i,j,k,mrv2))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mrv2) &
     &   + nflux3(i,j,k,mrv2))/(x3a(k+1)-x3a(k)) &
     &      )

         mv3(i,j,k) = mv3(i,j,k)  &
     & +dt*( &
     &      +  grvsrc3(i,j,k)                      &
     & +(- nflux1(i+1,j,k,mrv3) &
     &   + nflux1(i,j,k,mrv3))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mrv3) &
     &   + nflux2(i,j,k,mrv3))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mrv3) &
     &   + nflux3(i,j,k,mrv3))/(x3a(k+1)-x3a(k)) &
     &      )

          et(i,j,k) = et(i,j,k) &
     & +dt*( &
     &  (- nflux1(i+1,j,k,meto) &
     &   + nflux1(i,j,k,meto))/(x1a(i+1)-x1a(i)) &
     & +(- nflux2(i,j+1,k,meto) &
     &   + nflux2(i,j,k,meto))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,meto) &
     &   + nflux3(i,j,k,meto))/(x3a(k+1)-x3a(k)) &
     &      )

          b1(i,j,k) = b1(i,j,k) &
     & +dt*( &
     &  (- nflux1(i+1,j,k,mbm1) &
     &   + nflux1(i,j,k,mbm1))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mbm1) &
     &   + nflux2(i,j,k,mbm1))/(x2a(j+1)-x2a(j)) &  
     & +(- nflux3(i,j,k+1,mbm1) &
     &   + nflux3(i,j,k,mbm1))/(x3a(k+1)-x3a(k)) &
     &      )

          b2(i,j,k) = b2(i,j,k) &
     & +dt*( &
     &  (- nflux1(i+1,j,k,mbm2) &
     &   + nflux1(i,j,k,mbm2))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mbm2) &
     &   + nflux2(i,j,k,mbm2))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mbm2) &
     &   + nflux3(i,j,k,mbm2))/(x3a(k+1)-x3a(k)) &
     &      )

          b3(i,j,k) = b3(i,j,k) &
     & +dt*( &
     &  (- nflux1(i+1,j,k,mbm3) &
     &   + nflux1(i,j,k,mbm3))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mbm3) &
     &   + nflux2(i,j,k,mbm3))/(x2a(j+1)-x2a(j)) &  
     & +(- nflux3(i,j,k+1,mbm3) &
     &   + nflux3(i,j,k,mbm3))/(x3a(k+1)-x3a(k)) &
     &      )

          bp(i,j,k) = bp(i,j,k) &
     & +dt*( &
     &  (- nflux1(i+1,j,k,mbps) &
     &   + nflux1(i,j,k,mbps))/(x1a(i+1)-x1a(i)) & 
     & +(- nflux2(i,j+1,k,mbps) &
     &   + nflux2(i,j,k,mbps))/(x2a(j+1)-x2a(j)) & 
     & +(- nflux3(i,j,k+1,mbps) &
     &   + nflux3(i,j,k,mbps))/(x3a(k+1)-x3a(k)) &
     &      )

!          print *, i,j,k,bp(i,j,k)
         DXcomp(1:ncomp,i,j,k) = DXcomp(1:ncomp,i,j,k) &
     & +dt*( &
     & +(- nflux1(i+1,j,k,mst:med) &
     &   + nflux1(i,j,k,mst:med))/(x1a(i+1)-x1a(i)) &
     & +(- nflux2(i,j+1,k,mst:med) &
     &   + nflux2(i,j,k,mst:med))/(x2a(j+1)-x2a(j)) &
     & +(- nflux3(i,j,k+1,mst:med) &
     &   + nflux3(i,j,k,mst:med))/(x3a(k+1)-x3a(k)) &
     &      )
      enddo
      enddo
      enddo
!$acc end kernels
#endif

      return
    end subroutine UpdateConsv
