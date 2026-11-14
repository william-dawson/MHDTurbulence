module basicmod
  use config, only: nhymax, nhydis, timemax, dtout & 
       & , ngridtotal1, ngridtotal2, ngridtotal3, ntiles & 
       & , x1min, x1max, x2min, x2max, x3min, x3max &
       & , ncomp
  implicit none
      integer::nhy
      real(8)::time,dt
      data time / 0.0d0 /

      integer,parameter:: ngrid1 = ngridtotal1/ntiles(1)
      integer,parameter:: ngrid2 = ngridtotal2/ntiles(2)
      integer,parameter:: ngrid3 = ngridtotal3/ntiles(3)
      integer,parameter:: mgn=2
      ! pad `in` to odd: nvfortran 26.5 emits unsound LDG.E.128 at some even
      ! leading dims (in=80 -> CUDA_ERROR_MISALIGNED_ADDRESS; absent at 81)
      integer,parameter:: in=ngrid1+2*mgn+1+mod(ngrid1+2*mgn+2,2) &
     &                  , jn=ngrid2+2*mgn+1 &
     &                  , kn=ngrid3+2*mgn+1
      integer,parameter:: is=mgn+1 &
     &                  , js=mgn+1 &
     &                  , ks=mgn+1
      integer,parameter:: ie=ngrid1+mgn &
     &                  , je=ngrid2+mgn &
     &                  , ke=ngrid3+mgn

      real(8),dimension(in)::x1a,x1b
      real(8),dimension(jn)::x2a,x2b
      real(8),dimension(kn)::x3a,x3b

      real(8),dimension(in,jn,kn)::d,et,mv1,mv2,mv3
      real(8),dimension(in,jn,kn)::p,ei,v1,v2,v3,cs
      real(8),dimension(in,jn,kn)::b1,b2,b3,bp

      real(8),dimension(in,jn,kn)::gp,gp1a,gp2a,gp3a
      
      real(8),dimension(ncomp,in,jn,kn):: DXcomp
      real(8),dimension(ncomp,in,jn,kn)::  Xcomp
            
!$acc declare create(ngrid1,ngrid2,ngrid3)
!$acc declare create(mgn)
!$acc declare create(in,jn,kn)
!$acc declare create(is,js,ks)
!$acc declare create(ie,je,ke)
      
!$acc declare create(dt)
!$acc declare create(x1a,x1b)
!$acc declare create(x2a,x2b)
!$acc declare create(x3a,x3b)
      
!$acc declare create(d,et,mv1,mv2,mv3)
!$acc declare create(p,ei,v1,v2,v3,cs)
!$acc declare create(b1,b2,b3,bp)
!$acc declare create(gp,gp1a,gp2a,gp3a)

!$acc declare create(ncomp)
!$acc declare create(Dxcomp,Xcomp)
      
      end module basicmod
      
      module eosmod
      implicit none
! adiabatic
      real(8),parameter::gam=5.0d0/3.0d0 ! adiabatic index
!$acc declare create(gam)
! isothermal
!      real(8)::csiso  !! isothemal sound speed
!!$acc declare create(csiso)
      end module eosmod
    
      module fluxmod
      use basicmod, only : in,jn,kn,ncomp
      implicit none
      real(8):: chg
      integer,parameter::nhyd=11+ncomp
      integer,parameter::nden=1,nve1=2,nve2=3,nve3=4,nene=5,npre=6,ncsp=7 &
           &                   ,nbm1=8,nbm2=9,nbm3=10,nbps=11&
           &                   ,nst=nhyd-ncomp+1,ned=nhyd ! composition
      real(8),dimension(in,jn,kn,nhyd):: svc
      
      integer,parameter:: mflx=9+ncomp,madd=3
      integer,parameter::mudn=     1,muvu=     2,muvv=     3,muvw=     4,muet=     5 &
     &                              ,mubu=     6,mubv=     7,mubw=     8,mubp=     9 &
     &                  ,mfdn=mflx+1,mfvu=mflx+2,mfvv=mflx+3,mfvw=mflx+4,mfet=mflx+5 &
     &                              ,mfbu=mflx+6,mfbv=mflx+7,mfbw=mflx+8,mfbp=mflx+9 &
     &                          ,must=  mflx-ncomp+1,mued=  mflx &! composition
     &                          ,mfst=2*mflx-ncomp+1,mfed=2*mflx &! composition
     &                          ,mcsp=2*mflx+1,mvel=2*mflx+2,mpre=2*mflx+3 

      integer,parameter:: mden=1,mrv1=2,mrv2=3,mrv3=4,meto=5   &
     &                          ,mrvu=muvu,mrvv=muvv,mrvw=muvw &
     &                          ,mbm1=6,mbm2=7,mbm3=8,mbps=9   &
     &                          ,mbmu=mubu,mbmv=mubv,mbmw=mubw &
     &                          ,mst=mflx-ncomp+1,med=mflx
      real(8),dimension(in,jn,kn,mflx):: nflux1,nflux2,nflux3
      real(8),dimension(in,jn,kn):: grvsrc1,grvsrc2,grvsrc3

!$acc declare create(chg)
!$acc declare create(svc,nflux1,nflux2,nflux3)
!$acc declare create(grvsrc1,grvsrc2,grvsrc3)
      end module fluxmod

      module cuda_kernels_f90
! Fortran glue for the CUDA C++ kernels in mhd_cuda.cu. Fetches device
! pointers for the module arrays (allocated by the OpenACC runtime via
! the module-level `!$acc declare create`) and launches on the OpenACC
! synchronous stream so CUDA kernels, OpenACC kernels and the per-step
! MPI exchanges stay ordered. Lives in mhd.f90 (not its own file) so
! the module's use-associations resolve without a circular .mod
! dependency; updateconsv.f90 compiles after mhd.o and may use it.
      use iso_c_binding, only: c_devptr, c_int, c_double
      use openacc, only: acc_deviceptr, acc_get_cuda_stream, acc_async_sync
      use basicmod, only: in, jn, kn, is, js, ks, ie, je, ke, dt &
           & , x1a, x2a, x3a &
           & , d, et, mv1, mv2, mv3, b1, b2, b3, bp, p &
           & , v1, v2, v3, ei, Xcomp, DXcomp
      use fluxmod, only: nhyd, ncomp, mflx, svc, nflux1, nflux2, nflux3 &
           & , chg, grvsrc1, grvsrc2, grvsrc3
      use eosmod, only: gam
      implicit none

      interface
      function statevevtor_cuda(d_p, v1_p, v2_p, v3_p, ei_p, b1_p, b2_p &
           & , b3_p, bp_p, xc_p, p_p, svc_p, gam, in_, jn_, kn_ &
           & , i1, i2, j1, j2, k1, k2, stream_p) bind(C) result(rc)
        import :: c_devptr, c_int, c_double
        implicit none
        type(c_devptr), value :: d_p, v1_p, v2_p, v3_p, ei_p, b1_p, b2_p
        type(c_devptr), value :: b3_p, bp_p, xc_p, p_p, svc_p, stream_p
        real(c_double), value :: gam
        integer(c_int), value :: in_, jn_, kn_, i1, i2, j1, j2, k1, k2
        integer(c_int) :: rc
      end function statevevtor_cuda

      function flux1_cuda(svc_p, nfx_p, chg_p, in_, jn_, kn_ &
           & , i1, i2, j1, j2, k1, k2, stream_p) bind(C) result(rc)
        import :: c_devptr, c_int
        implicit none
        type(c_devptr), value :: svc_p, nfx_p, chg_p, stream_p
        integer(c_int), value :: in_, jn_, kn_, i1, i2, j1, j2, k1, k2
        integer(c_int) :: rc
      end function flux1_cuda

      function flux2_cuda(svc_p, nfx_p, chg_p, in_, jn_, kn_ &
           & , i1, i2, j1, j2, k1, k2, stream_p) bind(C) result(rc)
        import :: c_devptr, c_int
        implicit none
        type(c_devptr), value :: svc_p, nfx_p, chg_p, stream_p
        integer(c_int), value :: in_, jn_, kn_, i1, i2, j1, j2, k1, k2
        integer(c_int) :: rc
      end function flux2_cuda

      function flux3_cuda(svc_p, nfx_p, chg_p, in_, jn_, kn_ &
           & , i1, i2, j1, j2, k1, k2, stream_p) bind(C) result(rc)
        import :: c_devptr, c_int
        implicit none
        type(c_devptr), value :: svc_p, nfx_p, chg_p, stream_p
        integer(c_int), value :: in_, jn_, kn_, i1, i2, j1, j2, k1, k2
        integer(c_int) :: rc
      end function flux3_cuda

      function updateconsv_cuda(nf1_p, nf2_p, nf3_p, d_p, et_p, mv1_p &
           & , mv2_p, mv3_p, b1_p, b2_p, b3_p, bp_p, dxc_p, g1_p, g2_p &
           & , g3_p, x1_p, x2_p, x3_p, dtv, in_, jn_, kn_ &
           & , i1, i2, j1, j2, k1, k2, stream_p) bind(C) result(rc)
        import :: c_devptr, c_int, c_double
        implicit none
        type(c_devptr), value :: nf1_p, nf2_p, nf3_p, d_p, et_p, mv1_p
        type(c_devptr), value :: mv2_p, mv3_p, b1_p, b2_p, b3_p, bp_p
        type(c_devptr), value :: dxc_p, g1_p, g2_p, g3_p, x1_p, x2_p, x3_p
        type(c_devptr), value :: stream_p
        real(c_double), value :: dtv
        integer(c_int), value :: in_, jn_, kn_, i1, i2, j1, j2, k1, k2
        integer(c_int) :: rc
      end function updateconsv_cuda
      end interface

      contains

      subroutine cuda_check(rc, where)
        integer(c_int), intent(in) :: rc
        character(len=*), intent(in) :: where
        if (rc /= 0_c_int) then
           write(*,'(a,i0,2a)') 'CUDA launch failed (', rc, ') in ', where
           error stop 'cuda_kernels_f90'
        end if
      end subroutine cuda_check

      subroutine cuda_statevevtor()
        type(c_devptr) :: sptr
        if (nhyd /= 12 .or. ncomp /= 1) then
           error stop 'cuda_statevevtor: requires ncomp=1 (nhyd=12)'
        end if
        sptr = acc_get_cuda_stream(acc_async_sync)
        call cuda_check(statevevtor_cuda( &
             acc_deviceptr(d), acc_deviceptr(v1), acc_deviceptr(v2) &
           , acc_deviceptr(v3), acc_deviceptr(ei), acc_deviceptr(b1) &
           , acc_deviceptr(b2), acc_deviceptr(b3), acc_deviceptr(bp) &
           , acc_deviceptr(Xcomp), acc_deviceptr(p), acc_deviceptr(svc) &
           , real(gam, c_double) &
           , int(in, c_int), int(jn, c_int), int(kn, c_int) &
           , 1_c_int, int(in-1, c_int) &
           , 1_c_int, int(jn-1, c_int) &
           , 1_c_int, int(kn-1, c_int) &
           , sptr), 'statevevtor')
      end subroutine cuda_statevevtor

      subroutine cuda_flux1()
        type(c_devptr) :: sptr
        if (mflx /= 10) error stop 'cuda_flux1: requires ncomp=1'
        sptr = acc_get_cuda_stream(acc_async_sync)
        call cuda_check(flux1_cuda( &
             acc_deviceptr(svc), acc_deviceptr(nflux1), acc_deviceptr(chg) &
           , int(in, c_int), int(jn, c_int), int(kn, c_int) &
           , int(is, c_int), int(ie+1, c_int) &
           , int(js, c_int), int(je, c_int) &
           , int(ks, c_int), int(ke, c_int) &
           , sptr), 'flux1')
      end subroutine cuda_flux1

      subroutine cuda_flux2()
        type(c_devptr) :: sptr
        if (mflx /= 10) error stop 'cuda_flux2: requires ncomp=1'
        sptr = acc_get_cuda_stream(acc_async_sync)
        call cuda_check(flux2_cuda( &
             acc_deviceptr(svc), acc_deviceptr(nflux2), acc_deviceptr(chg) &
           , int(in, c_int), int(jn, c_int), int(kn, c_int) &
           , int(is, c_int), int(ie, c_int) &
           , int(js, c_int), int(je+1, c_int) &
           , int(ks, c_int), int(ke, c_int) &
           , sptr), 'flux2')
      end subroutine cuda_flux2

      subroutine cuda_flux3()
        type(c_devptr) :: sptr
        if (mflx /= 10) error stop 'cuda_flux3: requires ncomp=1'
        sptr = acc_get_cuda_stream(acc_async_sync)
        call cuda_check(flux3_cuda( &
             acc_deviceptr(svc), acc_deviceptr(nflux3), acc_deviceptr(chg) &
           , int(in, c_int), int(jn, c_int), int(kn, c_int) &
           , int(is, c_int), int(ie, c_int) &
           , int(js, c_int), int(je, c_int) &
           , int(ks, c_int), int(ke+1, c_int) &
           , sptr), 'flux3')
      end subroutine cuda_flux3

      subroutine cuda_updateconsv()
        type(c_devptr) :: sptr
        if (ncomp /= 1) error stop 'cuda_updateconsv: requires ncomp=1'
        sptr = acc_get_cuda_stream(acc_async_sync)
        call cuda_check(updateconsv_cuda( &
             acc_deviceptr(nflux1), acc_deviceptr(nflux2) &
           , acc_deviceptr(nflux3) &
           , acc_deviceptr(d), acc_deviceptr(et), acc_deviceptr(mv1) &
           , acc_deviceptr(mv2), acc_deviceptr(mv3), acc_deviceptr(b1) &
           , acc_deviceptr(b2), acc_deviceptr(b3), acc_deviceptr(bp) &
           , acc_deviceptr(DXcomp) &
           , acc_deviceptr(grvsrc1), acc_deviceptr(grvsrc2) &
           , acc_deviceptr(grvsrc3) &
           , acc_deviceptr(x1a), acc_deviceptr(x2a), acc_deviceptr(x3a) &
           , real(dt, c_double) &
           , int(in, c_int), int(jn, c_int), int(kn, c_int) &
           , int(is, c_int), int(ie, c_int) &
           , int(js, c_int), int(je, c_int) &
           , int(ks, c_int), int(ke, c_int) &
           , sptr), 'updateconsv')
      end subroutine cuda_updateconsv

      end module cuda_kernels_f90

      subroutine ConsvVariable
      use basicmod
      implicit none
      integer::i,j,k
      
!$acc kernels      
!$acc loop collapse(3) independent 
      do k=ks,ke
      do j=js,je
      do i=is,ie
          et(i,j,k) = 0.5d0*d(i,j,k)*(   &
     &                    +v1(i,j,k)**2  &
     &                    +v2(i,j,k)**2  &
     &                    +v3(i,j,k)**2) &
     &               +0.5d0*(            &
     &                    +b1(i,j,k)**2  &
     &                    +b2(i,j,k)**2  &
     &                    +b3(i,j,k)**2) &
     &                    +ei(i,j,k)
          mv1(i,j,k) =d(i,j,k)*v1(i,j,k)
          mv2(i,j,k) =d(i,j,k)*v2(i,j,k)
          mv3(i,j,k) =d(i,j,k)*v3(i,j,k)
          DXcomp(1:ncomp,i,j,k) = d(i,j,k)*Xcomp(1:ncomp,i,j,k)
      enddo
      enddo
      enddo
!$acc end kernels
      
      return
      end subroutine Consvvariable

      subroutine PrimVariable
      use basicmod
      use eosmod  
      implicit none
      integer::i,j,k,n
!$acc kernels      
!$acc loop collapse(3) independent private(n)
      do k=ks,ke
      do j=js,je
      do i=is,ie
          v1(i,j,k) = mv1(i,j,k)/d(i,j,k)
          v2(i,j,k) = mv2(i,j,k)/d(i,j,k)
          v3(i,j,k) = mv3(i,j,k)/d(i,j,k)

          ei(i,j,k) =  et(i,j,k)          &
     &          -0.5d0*d(i,j,k)*(         &
     &                    +v1(i,j,k)**2   &
     &                    +v2(i,j,k)**2   &
     &                    +v3(i,j,k)**2)  &
     &          -0.5d0*(                  &
     &                    +b1(i,j,k)**2   &
     &                    +b2(i,j,k)**2   &
     &                    +b3(i,j,k)**2)
! adiabatic
           p(i,j,k) =  ei(i,j,k)*(gam-1.0d0)
           cs(i,j,k) =  sqrt(gam*p(i,j,k)/d(i,j,k))
! isotermal
!           p(i,j,k) =  d(i,j,k)*csiso**2
!          cs(i,j,k) =  csiso
           
          do n=1,ncomp
             DXcomp(n,i,j,k) = max(0.0d0,DXcomp(n,i,j,k))
          enddo
          Xcomp(1:ncomp,i,j,k) = DXcomp(1:ncomp,i,j,k)/d(i,j,k)
      enddo
      enddo
      enddo
!$acc end kernels

      return
      end subroutine PrimVariable

subroutine TimestepControl
  use basicmod
  use mpimod      
  implicit none
  real(8)::dtl1
  real(8)::dtl2
  real(8)::dtl3
  real(8)::dtlocal
  real(8)::dtmin
  integer::theid
  real(8)::ctot
  integer::i,j,k

!$acc kernels
  dtmin=1.0d90
!$acc loop collapse(3) reduction(min:dtmin)  
  do k=ks,ke
  do j=js,je
  do i=is,ie
         ctot = sqrt(cs(i,j,k)**2 &
     &            +( b1(i,j,k)**2 &
     &              +b2(i,j,k)**2 &
     &              +b3(i,j,k)**2 &
     &                           )/d(i,j,k))
         dtl1 =(x1a(i+1)-x1a(i))/(abs(v1(i,j,k)) + ctot)
         dtl2 =(x2a(j+1)-x2a(j))/(abs(v2(i,j,k)) + ctot)
         dtl3 =(x3a(k+1)-x3a(k))/(abs(v3(i,j,k)) + ctot)
         dtlocal = min (dtl1,dtl2,dtl3)
         if(dtlocal .lt. dtmin) dtmin = dtlocal
  enddo
  enddo
  enddo
  bufinpmin(1) = dtmin
  bufinpmin(2) = dble(myid_w)
!$acc end kernels
  call MPIminfind
!$acc kernels
  dtmin =     bufoutmin(1)
  theid = int(bufoutmin(2))
  dt = 0.05d0 * dtmin
!$acc end kernels
!$acc update host (dt)
  
  return
end subroutine TimestepControl

      subroutine StateVevtor
      use basicmod
      use fluxmod
      use eosmod
#ifdef USE_CUDA_SV
      use cuda_kernels_f90
#endif
      implicit none
      integer::i,j,k

#ifdef USE_CUDA_SV
      call cuda_statevevtor()
#else
!$acc kernels
!$acc loop collapse(3) independent
      do k=1,kn-1
      do j=1,jn-1
      do i=1,in-1
         svc(i,j,k,nden) =  d(i,j,k)
         svc(i,j,k,nve1) = v1(i,j,k)
         svc(i,j,k,nve2) = v2(i,j,k)
         svc(i,j,k,nve3) = v3(i,j,k)

         svc(i,j,k,nene) = ei(i,j,k)/d(i,j,k)
         svc(i,j,k,npre) = ei(i,j,k)*(gam-1.0d0)
         svc(i,j,k,ncsp) = sqrt(gam*(gam-1.0d0)*ei(i,j,k)/d(i,j,k))
         p(i,j,k) = svc(i,j,k,npre)  ! for output boundary

         svc(i,j,k,nbm1) = b1(i,j,k)
         svc(i,j,k,nbm2) = b2(i,j,k)
         svc(i,j,k,nbm3) = b3(i,j,k)
         svc(i,j,k,nbps) = bp(i,j,k)
         svc(i,j,k,nst:ned) = Xcomp(1:ncomp,i,j,k)

      enddo
      enddo
      enddo
!$acc end kernels
#endif

      return
      end subroutine StateVevtor

      subroutine minmod(a,b,d)
!$acc routine seq
      use fluxmod, only : nhyd
      implicit none
      real(8),dimension(nhyd),intent(in)::a,b
      real(8),dimension(nhyd),intent(out)::d
      integer:: n

      do n=1,nhyd
         d(n) = sign(1.0d0,a(n))*max(0.0d0,min(abs(a(n)) &
     &                                        ,sign(1.0d0,a(n))*b(n)))
      enddo

      return
      end subroutine minmod


      subroutine vanLeer(dvp,dvm,dv)
!$acc routine seq
      use fluxmod, only : nhyd
      implicit none
      real(8),dimension(nhyd),intent(in)::dvp,dvm
      real(8),dimension(nhyd),intent(out)::dv
      integer:: n

      do n=1,nhyd
         if(dvp(n)*dvm(n) .gt. 0.0d0)then
            dv(n) =2.0d0*dvp(n)*dvm(n)/(dvp(n)+dvm(n))
         else
            dv(n) = 0.0d0
         endif

      enddo

      return
      end subroutine vanLeer



      subroutine MClimiter(a,b,c,d)
!$acc routine seq
      use fluxmod, only : nhyd
      implicit none
      real(8),dimension(nhyd),intent(in)::a,b,c
      real(8),dimension(nhyd),intent(out)::d
      integer:: n

      do n=1,nhyd
         d(n) = sign(1.0d0,a(n))*max(0.0d0,min(abs(a(n))          &
     &                                  ,sign(1.0d0,a(n))*b(n)    &
     &                                  ,sign(1.0d0,a(n))*c(n)))
      enddo

      return
      end subroutine MClimiter



      subroutine NumericalFlux1
      use basicmod, only: is,ie,in,js,je,jn,ks,ke,kn
      use fluxmod
#ifdef USE_CUDA_FLUX
      use cuda_kernels_f90
#endif
      implicit none
      integer::i,j,k
      real(8),dimension(nhyd):: dsvp,dsvm,dsvc,dsv
      real(8),dimension(nhyd):: Pleftc1, Pleftc2, Plefte
      real(8),dimension(nhyd):: Prigtc1, Prigtc2, Prigte
      real(8),dimension(2*mflx+madd):: leftco,rigtco
      real(8),dimension(mflx):: nflux
      real(8):: ptl,css,cts,etot

#ifdef USE_CUDA_FLUX
      call cuda_flux1()
#else
!$acc kernels
!$acc loop collapse(3) independent private(i, Pleftc1, Pleftc2, Plefte,Prigtc1, Prigtc2, Prigte,dsv,dsvp,dsvm,etot,ptl,css,cts,leftco,rigtco,nflux)
      do k=ks,ke
      do j=js,je
      do i=is,ie+1
         Pleftc1(:) = svc(i-2,j,k,:)
         Pleftc2(:) = svc(i-1,j,k,:)
         Prigtc1(:) = svc(i,j,k,:)
         Prigtc2(:) = svc(i+1,j,k,:)
         
! | Pleftc1   | Pleftc2 =>| Prigtc1   | Prigtc2   |        
!                     You are here               
!====================
! Left
!====================
         dsvp(:) = Prigtc1(:) - Pleftc2(:) 
         dsvm(:) =              Pleftc2(:) - Pleftc1(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Plefte(:) = Pleftc2(:) + 0.5d0*dsv(:)

! Consvative variables
         leftco(mudn)=Plefte(nden) ! rho
         leftco(muvu)=Plefte(nve1)*Plefte(nden)   ! rho v_x
         leftco(muvv)=Plefte(nve2)*Plefte(nden)   ! rho v_y
         leftco(muvw)=Plefte(nve3)*Plefte(nden)   ! rho v_z
         etot =Plefte(nene)*Plefte(nden) & ! e_i
     &               +0.5d0*Plefte(nden)*(                  &
     &                     +Plefte(nve1)**2                 &
     &                     +Plefte(nve2)**2                 &
     &                     +Plefte(nve3)**2)                & ! + rho v^2/2
     &               +0.5d0*                    (                 &
     &                     +Plefte(nbm1)**2                 &
     &                     +Plefte(nbm2)**2                 &
     &                     +Plefte(nbm3)**2)                  ! + B^2/2

         leftco(muet) = etot
         leftco(mubu)=Plefte(nbm1)  ! b_x
         leftco(mubv)=Plefte(nbm2)  ! b_y
         leftco(mubw)=Plefte(nbm3)  ! b_z
         leftco(mubp)=Plefte(nbps)  ! psi
         leftco(must:mued)=Plefte(nst:ned)*Plefte(nden) ! rho X
         
! Flux
         ptl = Plefte(npre) + ( Plefte(nbm1)**2        &
     &                         +Plefte(nbm2)**2        &
     &                         +Plefte(nbm3)**2)/2.0d0 

         leftco(mfdn)=Plefte(nden)             *Plefte(nve1)
         leftco(mfvu)=Plefte(nden)*Plefte(nve1)*Plefte(nve1) &
     &                        +ptl-Plefte(nbm1)*Plefte(nbm1)
         leftco(mfvv)=Plefte(nden)*Plefte(nve2)*Plefte(nve1) &
     &                            -Plefte(nbm2)*Plefte(nbm1)
         leftco(mfvw)=Plefte(nden)*Plefte(nve3)*Plefte(nve1) &
     &                            -Plefte(nbm3)*Plefte(nbm1)
         leftco(mfet)= (etot +ptl   )*Plefte(nve1) &
     &                -( Plefte(nbm1)*Plefte(nve1) &
     &                  +Plefte(nbm2)*Plefte(nve2) &
     &                  +Plefte(nbm3)*Plefte(nve3))*Plefte(nbm1)

         leftco(mfbu) =  0.0d0
         leftco(mfbv) =  Plefte(nbm2)*Plefte(nve1) &
     &                  -Plefte(nve2)*Plefte(nbm1)
         leftco(mfbw) =  Plefte(nbm3)*Plefte(nve1) &
     &                  -Plefte(nve3)*Plefte(nbm1)
         leftco(mfbp) = 0.0d0  ! psi
         leftco(mfst:mfed)=Plefte(nden)*Plefte(nst:ned)*Plefte(nve1) ! rho X v
         
         css =Plefte(ncsp)**2
         cts =  css  & !c_s^2*c_a^2
     &                       +( Plefte(nbm1)**2  &
     &                         +Plefte(nbm2)**2  &
     &                         +Plefte(nbm3)**2)/Plefte(nden) 

         leftco(mcsp)= sqrt((cts +sqrt(cts**2                 &
     &                             -4.0d0*css*Plefte(nbm1)**2 &
     &                                       /Plefte(nden))   &
     &                            )/2.0d0)
         leftco(mvel)= Plefte(nve1)
         leftco(mpre)= ptl

! | Pleftc1   | Pleftc2 |<= Prigtc1   | Prigtc2   |        
!                     You are here               
!====================
! Right
!====================
         dsvp = Prigtc2(:) - Prigtc1(:) 
         dsvm =              Prigtc1(:) - Pleftc2(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Prigte(:) = Prigtc1(:) - 0.5d0*dsv(:)
         
! Consvative variables
         rigtco(mudn)=Prigte(nden) ! rho
         rigtco(muvu)=Prigte(nve1)*Prigte(nden)   ! rho v_x
         rigtco(muvv)=Prigte(nve2)*Prigte(nden)   ! rho v_y
         rigtco(muvw)=Prigte(nve3)*Prigte(nden)   ! rho v_z
         etot =Prigte(nene)*Prigte(nden) & ! e_i
     &               +0.5d0*Prigte(nden)*(                  &
     &                     +Prigte(nve1)**2                 &
     &                     +Prigte(nve2)**2                 &
     &                     +Prigte(nve3)**2)                & ! + rho v^2/2
     &               +0.5d0*                    (                 &
     &                     +Prigte(nbm1)**2                 &
     &                     +Prigte(nbm2)**2                 &
     &                     +Prigte(nbm3)**2)                 ! + B^2/2
         rigtco(muet) = etot
         rigtco(mubu)=Prigte(nbm1)  ! b_x
         rigtco(mubv)=Prigte(nbm2)  ! b_y
         rigtco(mubw)=Prigte(nbm3)  ! b_z
         rigtco(mubp)=Prigte(nbps)  ! psi
         rigtco(must:mued)=Prigte(nden)*Prigte(nst:ned) ! rho X
 
! Flux
         ptl = Prigte(npre) + ( Prigte(nbm1)**2        &
     &                         +Prigte(nbm2)**2        &
     &                         +Prigte(nbm3)**2)/2.0d0

         rigtco(mfdn)=Prigte(nden)             *Prigte(nve1)
         rigtco(mfvu)=Prigte(nden)*Prigte(nve1)*Prigte(nve1) &
     &                        +ptl-Prigte(nbm1)*Prigte(nbm1)
         rigtco(mfvv)=Prigte(nden)*Prigte(nve2)*Prigte(nve1) &
     &                            -Prigte(nbm2)*Prigte(nbm1)
         rigtco(mfvw)=Prigte(nden)*Prigte(nve3)*Prigte(nve1) &
     &                            -Prigte(nbm3)*Prigte(nbm1)
         rigtco(mfet)=(etot+ptl)*Prigte(nve1) &
     &                     -( Prigte(nbm1)*Prigte(nve1)     &
     &                       +Prigte(nbm2)*Prigte(nve2)     &
     &                       +Prigte(nbm3)*Prigte(nve3))*Prigte(nbm1)
     
         rigtco(mfbu) =  0.0d0
         rigtco(mfbv) =  Prigte(nbm2)*Prigte(nve1) &
     &                  -Prigte(nve2)*Prigte(nbm1)
         rigtco(mfbw) =  Prigte(nbm3)*Prigte(nve1) &
     &                  -Prigte(nve3)*Prigte(nbm1)
         rigtco(mfbp) = 0.0d0  ! b_z
         rigtco(mfst:mfed)=Prigte(nden)*Prigte(nst:ned)*Prigte(nve1) ! rho X v
         
         css = Prigte(ncsp)**2
         cts =  css   &!c_s^2*c_a^2
     &                       +( Prigte(nbm1)**2 &
     &                         +Prigte(nbm2)**2 &
     &                         +Prigte(nbm3)**2)/Prigte(nden) 


         rigtco(mcsp)= sqrt((cts +sqrt(cts**2                 &
     &                             -4.0d0*css*Prigte(nbm1)**2 &
     &                                       /Prigte(nden))   &
     &                            )/2.0d0)
         rigtco(mvel)= Prigte(nve1)
         rigtco(mpre)= ptl

!         call HLLE(leftco,rigtco,nflux)
!         call HLLC(leftco,rigtco,nflux)
         call HLLD(leftco,rigtco,nflux)
         nflux1(i,j,k,mden)=nflux(mden)
         nflux1(i,j,k,mrv1)=nflux(mrvu)
         nflux1(i,j,k,mrv2)=nflux(mrvv)
         nflux1(i,j,k,mrv3)=nflux(mrvw)
         nflux1(i,j,k,meto)=nflux(meto)
         nflux1(i,j,k,mbm1)=nflux(mbmu)
         nflux1(i,j,k,mbm2)=nflux(mbmv)
         nflux1(i,j,k,mbm3)=nflux(mbmw)
         nflux1(i,j,k,mst:med)=nflux(mst:med)

         nflux1(i,j,k,mbm1) =  0.5d0*(leftco(mubp)+rigtco(mubp)) &
     &                    -0.5d0*chg*(rigtco(mubu)-leftco(mubu))        ! finite volume
         nflux1(i,j,k,mbps) = (0.5d0*(leftco(mubu)+rigtco(mubu)) &
     &                    -0.5d0/chg*(rigtco(mubp)-leftco(mubp)))*chg**2 ! finite volume

!         print *, "bpf1",nflux1(i,j,k,mbps)

      enddo
      enddo
      enddo
!$acc end kernels
#endif

      return
      end subroutine Numericalflux1

      subroutine NumericalFlux2
      use basicmod, only: is,ie,in,js,je,jn,ks,ke,kn
      use fluxmod
#ifdef USE_CUDA_FLUX
      use cuda_kernels_f90
#endif
      implicit none
! | Pleftc1   | Pleftc2 | Prigtc1   | Prigtc2   |        
!                     You are here               
      integer::i,j,k
      real(8),dimension(nhyd):: dsvp,dsvm,dsvc,dsv
      real(8),dimension(nhyd):: Pleftc1, Pleftc2, Plefte
      real(8),dimension(nhyd):: Prigtc1, Prigtc2, Prigte
      real(8),dimension(2*mflx+madd):: leftco,rigtco
      real(8),dimension(mflx):: nflux
      real(8):: ptl,css,cts,etot

#ifdef USE_CUDA_FLUX
      call cuda_flux2()
#else
!$acc kernels
!$acc loop collapse(3) independent private(j, Pleftc1, Pleftc2, Plefte,Prigtc1, Prigtc2, Prigte,dsv,dsvp,dsvm,etot,ptl,css,cts,leftco,rigtco,nflux)
      do k=ks,ke
      do i=is,ie
      do j=js,je+1
         Pleftc1(:) = svc(i,j-2,k,:)
         Pleftc2(:) = svc(i,j-1,k,:)
         Prigtc1(:) = svc(i,j,k,:)
         Prigtc2(:) = svc(i,j+1,k,:)     
! | Pleftc1   | Pleftc2 |<= Prigtc1   | Prigtc2   |        
!                     You are here   
!====================
! Left
!====================
         dsvp(:) = Prigtc1(:) - Pleftc2(:) 
         dsvm(:) = Pleftc2(:) - Pleftc1(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Plefte(:) = Pleftc2(:) + 0.5d0*dsv(:)
         
! Consvative variables
         leftco(mudn)=Plefte(nden)
         leftco(muvw)=Plefte(nve1)*Plefte(nden)
         leftco(muvu)=Plefte(nve2)*Plefte(nden)   ! rho v
         leftco(muvv)=Plefte(nve3)*Plefte(nden)
         etot        =Plefte(nene)*Plefte(nden) & ! internal
     &               +0.5d0*Plefte(nden)*(                  &
     &                     +Plefte(nve1)**2                 &
     &                     +Plefte(nve2)**2                 &
     &                     +Plefte(nve3)**2)                & ! kinetic
     &               +0.5d0*                    (                 &
     &                     +Plefte(nbm1)**2                 &
     &                     +Plefte(nbm2)**2                 &
     &                     +Plefte(nbm3)**2) ! magnetic
         leftco(muet)=etot
         leftco(mubw)=Plefte(nbm1)  ! b_x
         leftco(mubu)=Plefte(nbm2)  ! b_y
         leftco(mubv)=Plefte(nbm3)  ! b_z
         leftco(mubp)=Plefte(nbps)  ! psi
         leftco(must:mued)=Plefte(nst:ned)*Plefte(nden) ! rho X

         ptl = Plefte(npre) + ( Plefte(nbm1)**2        &
     &                         +Plefte(nbm2)**2        &
     &                         +Plefte(nbm3)**2)/2.0d0 

         leftco(mfdn)=Plefte(nden)             *Plefte(nve2)   ! rho v
         leftco(mfvw)=Plefte(nden)*Plefte(nve1)*Plefte(nve2) &
     &                            -Plefte(nbm1)*Plefte(nbm2)
         leftco(mfvu)=Plefte(nden)*Plefte(nve2)*Plefte(nve2) &
     &                        +ptl-Plefte(nbm2)*Plefte(nbm2)
         leftco(mfvv)=Plefte(nden)*Plefte(nve3)*Plefte(nve2) &
     &                            -Plefte(nbm3)*Plefte(nbm2)
         leftco(mfet)=(etot+ptl)*Plefte(nve2) &
     &                     -( Plefte(nbm1)*Plefte(nve1)     &
     &                       +Plefte(nbm2)*Plefte(nve2)     &
     &                       +Plefte(nbm3)*Plefte(nve3))*Plefte(nbm2)

         leftco(mfbw) =  Plefte(nbm1)*Plefte(nve2) &
     &                  -Plefte(nve1)*Plefte(nbm2)
         leftco(mfbu) =  0.0d0
         leftco(mfbv) =  Plefte(nbm3)*Plefte(nve2) &
     &                  -Plefte(nve3)*Plefte(nbm2)
         leftco(mfbp) = 0.0d0  ! psi
         leftco(mfst:mfed)=Plefte(nst:ned)*Plefte(nden)*Plefte(nve2) ! rho X v
     
         css = Plefte(ncsp)**2
         cts =  css  & !c_s^2*c_a^2
     &                       +( Plefte(nbm1)**2  &
     &                         +Plefte(nbm2)**2  &
     &                         +Plefte(nbm3)**2)/Plefte(nden) 

         leftco(mcsp)= sqrt((cts +sqrt(cts**2                  &
     &                             -4.0d0*css*Plefte(nbm2)**2  &
     &                                          /Plefte(nden)) &
     &                            )/2.0d0)
         leftco(mvel)= Plefte(nve2)
         leftco(mpre)= ptl
     
! | Pleftc1   | Pleftc2 =>| Prigtc1   | Prigtc2   |        
!                     You are here  
!====================
! Right
!====================
         dsvp(:) = Prigtc2(:) - Prigtc1(:) 
         dsvm(:) = Prigtc1(:) - Pleftc2(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Prigte(:) = Prigtc1(:) - 0.5d0*dsv(:)

         rigtco(mudn)=Prigte(nden)
         rigtco(muvw)=Prigte(nve1)*Prigte(nden)
         rigtco(muvu)=Prigte(nve2)*Prigte(nden)   ! rho v
         rigtco(muvv)=Prigte(nve3)*Prigte(nden)
         etot =Prigte(nene)*Prigte(nden) & ! internal
     &               +0.5d0*Prigte(nden)*(                  &
     &                     +Prigte(nve1)**2                 &
     &                     +Prigte(nve2)**2                 &
     &                     +Prigte(nve3)**2)                & ! kinetic
     &               +0.5d0*                    (                 &
     &                     +Prigte(nbm1)**2                 &
     &                     +Prigte(nbm2)**2                 &
     &                     +Prigte(nbm3)**2) ! magnetic
         rigtco(muet) = etot
         rigtco(mubw)=Prigte(nbm1)  ! b_x
         rigtco(mubu)=Prigte(nbm2)  ! b_y
         rigtco(mubv)=Prigte(nbm3)  ! b_z
         rigtco(mubp)=Prigte(nbps)  ! psi
         rigtco(must:mued)=Prigte(nden)*Prigte(nst:ned) ! rho X v
         
         ptl = Prigte(npre) + ( Prigte(nbm1)**2 &
     &                         +Prigte(nbm2)**2 &
     &                         +Prigte(nbm3)**2)/2.0d0 

         rigtco(mfdn)=Prigte(nden)             *Prigte(nve2) ! rho v
         rigtco(mfvw)=Prigte(nden)*Prigte(nve1)*Prigte(nve2) &
     &                            -Prigte(nbm1)*Prigte(nbm2)
         rigtco(mfvu)=Prigte(nden)*Prigte(nve2)*Prigte(nve2) &
     &                        +ptl-Prigte(nbm2)*Prigte(nbm2)
         rigtco(mfvv)=Prigte(nden)*Prigte(nve3)*Prigte(nve2) &
     &                            -Prigte(nbm3)*Prigte(nbm2)
         rigtco(mfet)=(etot+ptl)*Prigte(nve2) &
     &                     -( Prigte(nbm1)*Prigte(nve1)     &
     &                       +Prigte(nbm2)*Prigte(nve2)     &
     &                       +Prigte(nbm3)*Prigte(nve3))*Prigte(nbm2)
         rigtco(mfbw) =  Prigte(nbm1)*Prigte(nve2) &
     &                  -Prigte(nve1)*Prigte(nbm2)
         rigtco(mfbu) =  0.0d0
         rigtco(mfbv) =  Prigte(nbm3)*Prigte(nve2) &
     &                        -Prigte(nve3)*Prigte(nbm2)
         rigtco(mfbp) = 0.0d0  ! psi
         rigtco(mfst:mfed)=Prigte(nden)*Prigte(nst:ned)*Prigte(nve2) ! rho X v
         
         css = Prigte(ncsp)**2
         cts =  css  & !c_s^2*c_a^2
     &                       +( Prigte(nbm1)**2 &
     &                         +Prigte(nbm2)**2 &
     &                         +Prigte(nbm3)**2)/Prigte(nden) 

         rigtco(mcsp)= sqrt((cts +sqrt(cts**2                  &
     &                             -4.0d0*css*Prigte(nbm2)**2  &
     &                                          /Prigte(nden)) &
     &                            )/2.0d0)
         rigtco(mvel)= Prigte(nve2)
         rigtco(mpre)= ptl

!         call HLLE(leftco,rigtco,nflux)
!         call HLLC(leftco,rigtco,nflux)
         call HLLD(leftco,rigtco,nflux)

         nflux2(i,j,k,mden)=nflux(mden)
         nflux2(i,j,k,mrv1)=nflux(mrvw)
         nflux2(i,j,k,mrv2)=nflux(mrvu) ! mrv2=3, mrvu=2
         nflux2(i,j,k,mrv3)=nflux(mrvv)
         nflux2(i,j,k,meto)=nflux(meto)
         nflux2(i,j,k,mbm1)=nflux(mbmw)
         nflux2(i,j,k,mbm2)=nflux(mbmu)
         nflux2(i,j,k,mbm3)=nflux(mbmv)
         nflux2(i,j,k,mst:med)=nflux(mst:med)

         nflux2(i,j,k,mbm2) =  0.5d0*(leftco(mubp)+rigtco(mubp)) &
     &                    -0.5d0*chg*(rigtco(mubu)-leftco(mubu))        ! finite volume
         nflux2(i,j,k,mbps) = (0.5d0*(leftco(mubu)+rigtco(mubu)) &
     &                    -0.5d0/chg*(rigtco(mubp)-leftco(mubp)))*chg**2 ! finite volume
!         print *, "bpf2",nflux2(i,j,k,mbps)

      enddo
      enddo
      enddo
!$acc end kernels
#endif
      
      return
      end subroutine Numericalflux2

      subroutine NumericalFlux3
      use basicmod, only: is,ie,in,js,je,jn,ks,ke,kn
      use fluxmod
#ifdef USE_CUDA_FLUX
      use cuda_kernels_f90
#endif
      implicit none
      integer::i,j,k
      real(8),dimension(nhyd):: dsvp,dsvm,dsvc,dsv
      real(8),dimension(nhyd):: Pleftc1, Pleftc2, Plefte
      real(8),dimension(nhyd):: Prigtc1, Prigtc2, Prigte
      real(8),dimension(2*mflx+madd):: leftco,rigtco
      real(8),dimension(mflx):: nflux
      real(8):: ptl,css,cts,etot

#ifdef USE_CUDA_FLUX
      call cuda_flux3()
#else
!$acc kernels
!$acc loop collapse(2) independent private(k, Pleftc1, Pleftc2, Plefte,Prigtc1, Prigtc2, Prigte,dsv,dsvp,dsvm,etot,ptl,css,cts,leftco,rigtco,nflux)
      do j=js,je
      do i=is,ie
      do k=ks,ke+1
            
         Pleftc1(:) = svc(i,j,k-2,:)
         Pleftc2(:) = svc(i,j,k-1,:)
         Prigtc1(:) = svc(i,j,k,:)
         Prigtc2(:) = svc(i,j,k+1,:)
                       
! | Pleftc1   | Pleftc2 =>| Prigtc1   | Prigtc2   |        
!                     You are here  
!====================
! Left
!====================
         dsvp(:) = Prigtc1(:) - Pleftc2(:) 
         dsvm(:) = Pleftc2(:) - Pleftc1(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Plefte(:) = Pleftc2(:) + 0.5d0*dsv(:)
         
! Consvative variables
         leftco(mudn)=Plefte(nden)
         leftco(muvv)=Plefte(nve1)*Plefte(nden)
         leftco(muvw)=Plefte(nve2)*Plefte(nden) ! rho v
         leftco(muvu)=Plefte(nve3)*Plefte(nden)
         etot =Plefte(nene)*Plefte(nden) & ! thermal
     &               +0.5d0*Plefte(nden)*(                  & ! kinetic
     &                     +Plefte(nve1)**2                 &
     &                     +Plefte(nve2)**2                 &
     &                     +Plefte(nve3)**2)                &
     &               +0.5d0*                    (                 & ! magnetic 
     &                     +Plefte(nbm1)**2                 &
     &                     +Plefte(nbm2)**2                 &
     &                     +Plefte(nbm3)**2)                 
         leftco(muet) = etot 
         leftco(mubv)=Plefte(nbm1)  ! b_x
         leftco(mubw)=Plefte(nbm2)  ! b_y
         leftco(mubu)=Plefte(nbm3)  ! b_z
         leftco(mubp)=Plefte(nbps)  ! psi
         leftco(must:mued)=Plefte(nden)*Plefte(nst:ned)
         
         ptl = Plefte(npre) + ( Plefte(nbm1)**2        &
     &                         +Plefte(nbm2)**2        &
     &                         +Plefte(nbm3)**2)/2.0d0 

         leftco(mfdn)=Plefte(nden)             *Plefte(nve3) ! rho v
         leftco(mfvv)=Plefte(nden)*Plefte(nve1)*Plefte(nve3) &
     &                            -Plefte(nbm1)*Plefte(nbm3)
         leftco(mfvw)=Plefte(nden)*Plefte(nve2)*Plefte(nve3) &
     &                            -Plefte(nbm2)*Plefte(nbm3)
         leftco(mfvu)=Plefte(nden)*Plefte(nve3)*Plefte(nve3) &
     &                     +ptl   -Plefte(nbm3)*Plefte(nbm3)
         leftco(mfet)=(etot+ptl                )*Plefte(nve3) &
     &                     -( Plefte(nbm1)*Plefte(nve1)     &
     &                       +Plefte(nbm2)*Plefte(nve2)     &
     &                       +Plefte(nbm3)*Plefte(nve3))*Plefte(nbm3)

         leftco(mfbv) =  Plefte(nbm1)*Plefte(nve3) &
     &                  -Plefte(nve1)*Plefte(nbm3)
         leftco(mfbw) =  Plefte(nbm2)*Plefte(nve3) &
     &                  -Plefte(nve2)*Plefte(nbm3)
         leftco(mfbu) = 0.d0
         leftco(mfbp) = 0.0d0  ! psi
         leftco(mfst:mfed)=Plefte(nden)*Plefte(nst:ned)*Plefte(nve3)

         css = Plefte(ncsp)**2
         cts =  css  & !c_s^2*c_a^2
     &                       +( Plefte(nbm1)**2  &
     &                         +Plefte(nbm2)**2  &
     &                         +Plefte(nbm3)**2)/Plefte(nden) 

         leftco(mcsp)= sqrt((cts +sqrt(cts**2                  &
     &                             -4.0d0*css*Plefte(nbm3)**2  &
     &                                          /Plefte(nden)) &
     &                            )/2.0d0)
         leftco(mvel)= Plefte(nve3)
         leftco(mpre)= ptl
     
! | Pleftc1   | Pleftc2 |<= Prigtc1   | Prigtc2   |        
!                     You are here  
!==============================
! Right 
!==============================
         dsvp = Prigtc2(:) - Prigtc1(:) 
         dsvm = Prigtc1(:) - Pleftc2(:)
         call vanLeer(dsvp,dsvm,dsv)
         !         call minmod(dsvp,dsvm,dsv)
         Prigte(:) = Prigtc1(:) - 0.5d0*dsv(:)
         
         rigtco(mudn)=Prigte(nden)
         rigtco(muvv)=Prigte(nve1)*Prigte(nden)
         rigtco(muvw)=Prigte(nve2)*Prigte(nden)
         rigtco(muvu)=Prigte(nve3)*Prigte(nden)
         etot =Prigte(nene)*Prigte(nden) & ! internal
     &               +0.5d0*Prigte(nden)*(                  & ! kinetic
     &                     +Prigte(nve1)**2                 &
     &                     +Prigte(nve2)**2                 &
     &                     +Prigte(nve3)**2)                & 
     &               +0.5d0*                    (                 & ! magnetic
     &                     +Prigte(nbm1)**2                 &
     &                     +Prigte(nbm2)**2                 &
     &                     +Prigte(nbm3)**2)
         rigtco(muet) = etot
         rigtco(mubv)=Prigte(nbm1)  ! b_x
         rigtco(mubw)=Prigte(nbm2)  ! b_y
         rigtco(mubu)=Prigte(nbm3)  ! b_z
         rigtco(mubp)=Prigte(nbps)  ! psi
         rigtco(must:mued)=Prigte(nden)*Prigte(nst:ned)

         ptl = Prigte(npre) + ( Prigte(nbm1)**2 &
     &                         +Prigte(nbm2)**2 &
     &                         +Prigte(nbm3)**2)/2.0d0 

         rigtco(mfdn)=Prigte(nden)             *Prigte(nve3)
         rigtco(mfvv)=Prigte(nden)*Prigte(nve1)*Prigte(nve3) &
     &                            -Prigte(nbm1)*Prigte(nbm3)
         rigtco(mfvw)=Prigte(nden)*Prigte(nve2)*Prigte(nve3) &
     &                            -Prigte(nbm2)*Prigte(nbm3)
         rigtco(mfvu)=Prigte(nden)*Prigte(nve3)*Prigte(nve3) &
     &                     +ptl   -Prigte(nbm3)*Prigte(nbm3)
         rigtco(mfet)=(etot+ptl                )*Prigte(nve3) &
     &                     -( Prigte(nbm1)*Prigte(nve1)      &
     &                       +Prigte(nbm2)*Prigte(nve2)      &
     &                       +Prigte(nbm3)*Prigte(nve3))*Prigte(nbm3)
         rigtco(mfbv) =  Prigte(nbm1)*Prigte(nve3) &
     &                  -Prigte(nve1)*Prigte(nbm3)
         rigtco(mfbw) =  Prigte(nbm2)*Prigte(nve3) &
     &                  -Prigte(nve2)*Prigte(nbm3)
         rigtco(mfbu) = 0.0d0
         rigtco(mfbp) = 0.0d0  ! psi
         rigtco(mfst:mfed)=Prigte(nden)*Prigte(nst:ned)*Prigte(nve3)
     
         css = Prigte(ncsp)**2
         cts =  css  & !c_s^2*c_a^2
     &                       +( Prigte(nbm1)**2 &
     &                         +Prigte(nbm2)**2 &
     &                         +Prigte(nbm3)**2)/Prigte(nden) 

         rigtco(mcsp)= sqrt((cts +sqrt(cts**2                  &
     &                             -4.0d0*css*Prigte(nbm3)**2  &
     &                                          /Prigte(nden)) &
     &                            )/2.0d0)
         rigtco(mvel)= Prigte(nve3)
         rigtco(mpre)= ptl

!         call HLLE(leftco,rigtco,nflux)
!         call HLLC(leftco,rigtco,nflux)
         call HLLD(leftco,rigtco,nflux)
         nflux3(i,j,k,mden)=nflux(mden)
         nflux3(i,j,k,mrv1)=nflux(mrvv)
         nflux3(i,j,k,mrv2)=nflux(mrvw)
         nflux3(i,j,k,mrv3)=nflux(mrvu)
         nflux3(i,j,k,meto)=nflux(meto)
         nflux3(i,j,k,mbm1)=nflux(mbmv)
         nflux3(i,j,k,mbm2)=nflux(mbmw)
         nflux3(i,j,k,mbm3)=nflux(mbmu)
         nflux3(i,j,k,mst:med)=nflux(mst:med)

         nflux3(i,j,k,mbm3) =  0.5d0*(leftco(mubp)+rigtco(mubp)) &
     &                    -0.5d0*chg*(rigtco(mubu)-leftco(mubu))        ! finite volume
         nflux3(i,j,k,mbps) = (0.5d0*(leftco(mubu)+rigtco(mubu)) &
     &                    -0.5d0/chg*(rigtco(mubp)-leftco(mubp)))*chg**2 ! finite volume

      enddo
      enddo
      enddo
!$acc end kernels
#endif

      return
      end subroutine Numericalflux3
      
      subroutine HLLE(leftst,rigtst,nflux)
!$acc routine seq
      use fluxmod
      implicit none
      real(8),dimension(2*mflx+madd),intent(in)::leftst,rigtst
      real(8),dimension(mflx),intent(out)::nflux
      real(8),dimension(mflx)::ul,ur,fl,fr
      real(8)::csl,csr
      real(8):: vl, vr
      real(8):: sl, sr

      ul(1:mflx) = leftst(1:mflx)
      fl(1:mflx) = leftst(mflx+1:2*mflx)
      ur(1:mflx) = rigtst(1:mflx)
      fr(1:mflx) = rigtst(mflx+1:2*mflx)
      csl=leftst(mcsp)
      csr=rigtst(mcsp)
       vl=leftst(mvel)
       vr=rigtst(mvel)

       sl = min(vl,vr) - max(csl,csr)
       sl = min(0.0d0,sl)
       sr = max(vl,vr) + max(csl,csr)
       sr = max(0.0d0,sr)

       nflux(:) = (sr*fl(:)-sl*fr(:) +sl*sr*(ur(:)-ul(:)))/(sr-sl)

      return
      end subroutine HLLE

      subroutine HLLC(leftst,rigtst,nflux)
!$acc routine seq
!=====================================================================
!
! HLLC Scheme
!
! Purpose
! Calculation of Numerical Flux by HLLC method
!
! Reference
!  Toro EF, Spruce M, Speares W. (1992,1994)
!
! Input
! Output
!=====================================================================
      use fluxmod, only: mflx,madd                &
     &                 , mudn,muvu,muvv,muvw,muet &
     &                 , mfdn,mfvu,mfvv,mfvw,mfet &
     &                 , mcsp,mvel,mpre           &
     &                 , mden,mrvu,mrvv,mrvw,meto &
     &                 , mubu,mubv,mubw,mubp      &
     &                 , mfbu,mfbv,mfbw,mfbp      &
     &                 , mbmu,mbmv,mbmw &
     &                 , ncomp, mst, med

      implicit none
      real(8),dimension(2*mflx+madd),intent(in)::leftst,rigtst
      real(8),dimension(mflx),intent(out)::nflux

!----- U -----
! qql :: left state
! qqr :: right state
      real(8) :: rol,vxl,vyl,vzl,ptl,eel
      real(8) :: ror,vxr,vyr,vzr,ptr,eer
      real(8) :: rxl,ryl,rzl
      real(8) :: rxr,ryr,rzr
      real(8) ::     byl,bzl
      real(8) ::     byr,bzr
      real(8) :: ptst
      real(8),dimension(ncomp) :: scl,scr

!----- U* ----
! qqlst ::  left state
! qqrst :: right state
      real(8) :: rolst,vxlst,vylst,vzlst,eelst
      real(8) :: rorst,vxrst,vyrst,vzrst,eerst
      real(8) :: rxlst,rylst,rzlst
      real(8) :: rxrst,ryrst,rzrst
      real(8) ::       bylst,bzlst
      real(8) ::       byrst,bzrst
      real(8),dimension(ncomp) :: sclst,scrst

!----- flux ---
! fqql ::  left physical flux
! fqqr :: right physical flux
      real(8) :: frol,frxl,fryl,frzl,feel
      real(8) ::           fbyl,fbzl
      real(8) :: fror,frxr,fryr,frzr,feer
      real(8) ::           fbyr,fbzr
      real(8),dimension(ncomp) :: fscl,fscr

!----- wave speed ---
! sl ::  left-going fastest signal velocity
! sr :: right-going fastest signal velocity
! sm :: contact discontinuity velocity
! slst ::  left-going alfven velocity
! srst :: right-going alfven velocity
      real(8) :: sm,sl,sr

! cfl :: left-state Fast wave velocity
! cfr :: right-sate Fast wave velocity
      real(8) :: cfl,cfr

!--------------------
! temporary variables
      real(8) :: sdl,sdr,sdml,sdmr,isdml,isdmr,rosdl,rosdr
      real(8) :: temp
      real(8),dimension(ncomp) :: scsdl,scsdr
! no if
      real(8) :: sign1,maxs1,mins1
      real(8) :: msl,msr

!----- Step 0. ----------------------------------------------------------|

!---- Left state
        
        rol = leftst(mudn)
        eel = leftst(muet)
        rxl = leftst(muvu)
        ryl = leftst(muvv)
        rzl = leftst(muvw)
        vxl = leftst(muvu)/leftst(mudn)
        vyl = leftst(muvv)/leftst(mudn)
        vzl = leftst(muvw)/leftst(mudn)
        scl(1:ncomp) = leftst(mst:med)
        byl = leftst(mubv)
        bzl = leftst(mubw)
        ptl = leftst(mpre)

!---- Right state
        
        ror = rigtst(mudn)
        eer = rigtst(muet)
        rxr = rigtst(muvu)
        ryr = rigtst(muvv)
        rzr = rigtst(muvw)
        vxr = rigtst(muvu)/rigtst(mudn)
        vyr = rigtst(muvv)/rigtst(mudn)
        vzr = rigtst(muvw)/rigtst(mudn)
        scr(1:ncomp) = rigtst(mst:med)
        byr = rigtst(mubv)
        bzr = rigtst(mubw)
        ptr = rigtst(mpre)
!----- Step 1. ----------------------------------------------------------|
! Compute wave left & right wave speed
!
         
        cfl = leftst(mcsp)
        cfr = rigtst(mcsp)

        sl = min(vxl,vxr)-max(cfl,cfr) ! note sl is negative
        sr = max(vxl,vxr)+max(cfl,cfr)
!----- Step 2. ----------------------------------------------------------|
! compute L/R fluxs
!
! Left value
        frol = leftst(mfdn)
        feel = leftst(mfet)
        frxl = leftst(mfvu)
        fryl = leftst(mfvv)
        frzl = leftst(mfvw)
        fbyl = leftst(mfbv)
        fbzl = leftst(mfbw)    
        fscl(1:ncomp) = leftst(mflx+mst:mflx+med)

! Right value
! Left value
        fror = rigtst(mfdn)
        feer = rigtst(mfet)
        frxr = rigtst(mfvu)
        fryr = rigtst(mfvv) 
        frzr = rigtst(mfvw)
        fbyr = rigtst(mfbv)
        fbzr = rigtst(mfbw)
        fscr(1:ncomp) = rigtst(mflx+mst:mflx+med)

!----- Step 4. ----------------------------------------------------------|
! compute middle and alfven wave
!
        sdl = sl - vxl
        sdr = sr - vxr
        rosdl = rol*sdl
        rosdr = ror*sdr
        scsdl(1:ncomp) = scl(1:ncomp)*sdl
        scsdr(1:ncomp) = scr(1:ncomp)*sdr

        temp = 1.0d0/(rosdr - rosdl)
! Eq. 45
        sm = (rosdr*vxr - rosdl*vxl - ptr + ptl)*temp
           
        sdml = sl - sm; isdml = 1.0d0/sdml
        sdmr = sr - sm; isdmr = 1.0d0/sdmr
        
!----- Step 5. ----------------------------------------------------------|
! compute intermediate states
!
! Eq. 49
        ptst = (rosdr*ptl-rosdl*ptr+rosdl*rosdr*(vxr-vxl))*temp

!----- Step 5A. ----------------------------------------------------------|
! compute Ul*
!

        rolst = rol*sdl   *isdml
        vxlst = sm
        rxlst = rolst*vxlst
           
        vylst = vyl
        rylst = rolst*vylst
        vzlst = vzl
        rzlst = rolst*vzlst

        bylst = rolst/rol * byl
        bzlst = rolst/rol * bzl

        eelst =(sdl*eel - ptl*vxl + ptst*sm  )*isdml

!----- Step 5B. ----------------------------------------------------------|
! compute Ur*
!

        rorst   = rosdr   *isdmr
        vxrst = sm
        rxrst = rorst*vxrst
        vyrst = vyr
        ryrst = rorst*vyrst
        vzrst = vzr
        rzrst = rorst*vzrst

        byrst = rorst/ror * byr
        bzrst = rorst/ror * bzr
           
        eerst = (sdr*eer - ptr*vxr  + ptst*sm  )*isdmr
              
!----- Step 6. ----------------------------------------------------------|
! compute flux
        sign1 = sign(1.0d0,sm)    ! 1 for sm>0, -1 for sm<0
        maxs1 =  max(0.0d0,sign1) ! 1 sm>0, 0 for sm<0
        mins1 = -min(0.0d0,sign1) ! 0 sm>0,-1 for sm<0

        msl   = min(sl  ,0.0d0)   ! 0 for sl > 0, sl for sl < 0
        msr   = max(sr  ,0.0d0)   ! S_R > 0

        nflux(mden) = (frol+msl*(rolst-rol))*maxs1 &
     &               +(fror+msr*(rorst-ror))*mins1
        nflux(meto) = (feel+msl*(eelst-eel))*maxs1 &
     &               +(feer+msr*(eerst-eer))*mins1
        nflux(mrvu) = (frxl+msl*(rxlst-rxl))*maxs1 &
     &               +(frxr+msr*(rxrst-rxr))*mins1
        nflux(mrvv) = (fryl+msl*(rylst-ryl))*maxs1 &
     &               +(fryr+msr*(ryrst-ryr))*mins1
        nflux(mrvw) = (frzl+msl*(rzlst-rzl))*maxs1 &
     &               +(frzr+msr*(rzrst-rzr))*mins1
        nflux(mbmu) = 0.0d0
        nflux(mbmv) = (fbyl+msl*(bylst-byl))*maxs1 &
     &               +(fbyr+msr*(byrst-byr))*mins1
        nflux(mbmw) = (fbzl+msl*(bzlst-bzl))*maxs1 &
     &               +(fbzr+msr*(bzrst-bzr))*mins1

        nflux(mst:med) = &
     &     (fscl(1:ncomp)+msl*(sclst(1:ncomp)-scl(1:ncomp)))*maxs1 &
     &    +(fscr(1:ncomp)+msr*(scrst(1:ncomp)-scr(1:ncomp)))*mins1
      return
      end subroutine HLLC

      subroutine HLLD(leftst,rigtst,nflux)
!$acc routine seq
!=====================================================================
!
! HLLD Scheme -- if/else variant. Replaces the branchless
! maxs1*(A)-mins1*(B) blends with real branches that keep only the
! taken path's state live. Expressions on the taken path are
! textually identical to the originals, so results should be
! bit-identical modulo sign-of-zero. The degenerate guards are
! preserved: the itf denominator guard and the 1e-30 eps tests.
!
!=====================================================================
      use fluxmod, only: mflx,madd                &
     &                 , mudn,muvu,muvv,muvw,muet &
     &                 , mfdn,mfvu,mfvv,mfvw,mfet &
     &                 , mcsp,mvel,mpre           &
     &                 , mden,mrvu,mrvv,mrvw,meto &
     &                 , mubu,mubv,mubw,mubp      &
     &                 , mfbu,mfbv,mfbw,mfbp      &
     &                 , mbmu,mbmv,mbmw           &
     &                 , ncomp,mst,med

      implicit none
      real(8),dimension(2*mflx+madd),intent(in)::leftst,rigtst
      real(8),dimension(mflx),intent(out)::nflux

!----- U -----
! qql :: left state
! qqr :: right state
      real(8) :: rol,vxl,vyl,vzl,ptl,eel
      real(8) :: ror,vxr,vyr,vzr,ptr,eer
      real(8) :: rxl,ryl,rzl
      real(8) :: rxr,ryr,rzr
      real(8) :: bxs,byl,bzl
      real(8) ::     byr,bzr
      real(8) :: ptst
      real(8),dimension(ncomp) :: scl,scr

!----- U* ----
! qqlst ::  left state
! qqrst :: right state
      real(8) :: rolst,vxlst,vylst,vzlst,eelst
      real(8) :: rorst,vxrst,vyrst,vzrst,eerst
      real(8) :: rxlst,rylst,rzlst
      real(8) :: rxrst,ryrst,rzrst
      real(8) ::       bylst,bzlst
      real(8) ::       byrst,bzrst
      real(8),dimension(ncomp) :: sclst,scrst

!----- U** ----
! qqlst ::  left state
! qqrst :: right state
      real(8) :: vyldst,vzldst,eeldst
      real(8) :: vyrdst,vzrdst,eerdst
      real(8) :: ryldst,rzldst
      real(8) :: ryrdst,rzrdst
      real(8) ::       byldst,bzldst
      real(8) ::       byrdst,bzrdst

!----- flux ---
! fqql ::  left physical flux
! fqqr :: right physical flux
      real(8) :: frol,frxl,fryl,frzl,feel
      real(8) ::           fbyl,fbzl
      real(8) :: fror,frxr,fryr,frzr,feer
      real(8) ::           fbyr,fbzr
      real(8),dimension(ncomp) :: fscl,fscr

!----- wave speed ---
      real(8) :: sm,sl,sr,slst,srst

! cfl :: left-state Fast wave velocity
! cfr :: right-sate Fast wave velocity
      real(8) :: cfl,cfr

!--------------------
! temporary variables
      real(8) :: sdl,sdr,sdml,sdmr,isdml,isdmr,rosdl,rosdr
      real(8) :: temp

! if/else variant: guard denominator, selectors, sqrt block
      real(8) :: msl,msr,mslst,msrst,invsumro,sqrtror,sqrtrol,abbx
      real(8) :: bxsq,temp_fst,eps,itf,vdbstl,vdbstr,signbx

!----- Step 0. ----------------------------------------------------------|
      eps = 1d-30
!---- Left state

        rol = leftst(mudn)
        eel = leftst(muet)
        rxl = leftst(muvu)
        ryl = leftst(muvv)
        rzl = leftst(muvw)
        vxl = leftst(muvu)/leftst(mudn)
        vyl = leftst(muvv)/leftst(mudn)
        vzl = leftst(muvw)/leftst(mudn)
        byl = leftst(mubv)
        bzl = leftst(mubw)
        ptl = leftst(mpre)
        scl(1:ncomp) = leftst(mst:med)

!---- Right state

        ror = rigtst(mudn)
        eer = rigtst(muet)
        rxr = rigtst(muvu)
        ryr = rigtst(muvv)
        rzr = rigtst(muvw)
        vxr = rigtst(muvu)/rigtst(mudn)
        vyr = rigtst(muvv)/rigtst(mudn)
        vzr = rigtst(muvw)/rigtst(mudn)
        byr = rigtst(mubv)
        bzr = rigtst(mubw)
        ptr = rigtst(mpre)
        scr(1:ncomp) = rigtst(mst:med)

!----- Step 1. ----------------------------------------------------------|
! Compute wave left & right wave speed
!
        cfl = leftst(mcsp)
        cfr = rigtst(mcsp)

        sl = min(vxl,vxr)-max(cfl,cfr) ! note sl is negative
        sr = max(vxl,vxr)+max(cfl,cfr)
!----- Step 2. ----------------------------------------------------------|
! compute L/R fluxs
!
! Left value
        frol = leftst(mfdn)
        feel = leftst(mfet)
        frxl = leftst(mfvu)
        fryl = leftst(mfvv)
        frzl = leftst(mfvw)
        fbyl = leftst(mfbv)
        fbzl = leftst(mfbw)
        fscl(1:ncomp) = leftst(mflx+mst:mflx+med)

! Right value
        fror = rigtst(mfdn)
        feer = rigtst(mfet)
        frxr = rigtst(mfvu)
        fryr = rigtst(mfvv)
        frzr = rigtst(mfvw)
        fbyr = rigtst(mfbv)
        fbzr = rigtst(mfbw)
        fscr(1:ncomp) = rigtst(mflx+mst:mflx+med)


!----- Step 4. ----------------------------------------------------------|
! compute middle and alfven wave
!
        sdl = min(-1d-20,sl - vxl)
        sdr = max(1d-20,sr - vxr)
        rosdl = rol*sdl
        rosdr = ror*sdr

        temp = 1.0d0/(rosdr - rosdl)
! Eq. 45
        sm = (rosdr*vxr - rosdl*vxl - ptr + ptl)*temp

        sdml = min(-1d-20,sl - sm); isdml = 1.0d0/sdml
        sdmr = max(1d-20,sr - sm); isdmr = 1.0d0/sdmr

!----- Step 5. ----------------------------------------------------------|
! compute intermediate states
!
! Eq. 49
        ptst = (rosdr*ptl-rosdl*ptr+rosdl*rosdr*(vxr-vxl))*temp

!----- Step 5A. ----------------------------------------------------------|
! compute Ul*
!
        bxs = 0.5d0*(leftst(mubu)+rigtst(mubu))
        bxsq = bxs*bxs
        temp_fst = rosdl*sdml - bxsq
        if (abs(temp_fst) >= eps) then
           itf = 1.0d0/temp_fst
           rolst = rosdl*isdml
           vxlst = sm
           temp = bxs*(sdl-sdml)*itf
           vylst = vyl - byl*temp
           vzlst = vzl - bzl*temp
           temp = (rosdl*sdl-bxsq)*itf
           bylst = byl*temp
           bzlst = bzl*temp
           vdbstl = vxlst*bxs+vylst*bylst+vzlst*bzlst
           eelst = (sdl*eel - ptl*vxl + ptst*sm +      &
     &          bxs*(vxl*bxs+vyl*byl+vzl*bzl-vdbstl))*isdml
        else
           rolst = rol; vxlst = vxl; vylst = vyl; vzlst = vzl
           bylst = byl; bzlst = bzl
           vdbstl = vxlst*bxs+vylst*bylst+vzlst*bzlst
           eelst = eel
        end if
        sclst(1:ncomp) = rolst/rol*scl(1:ncomp)
        rxlst = rolst*vxlst
        rylst = rolst*vylst
        rzlst = rolst*vzlst

!----- Step 5B. ----------------------------------------------------------|
! compute Ur*
!
        temp_fst = rosdr*sdmr - bxsq
        if (abs(temp_fst) >= eps) then
           itf = 1.0d0/temp_fst
           rorst = rosdr*isdmr
           vxrst = sm
           temp = bxs*(sdr-sdmr)*itf
           vyrst = vyr - byr*temp
           vzrst = vzr - bzr*temp
           temp = (rosdr*sdr-bxsq)*itf
           byrst = byr*temp
           bzrst = bzr*temp
           vdbstr = vxrst*bxs+vyrst*byrst+vzrst*bzrst
           eerst = ((sdr*eer - ptr*vxr  + ptst*sm)*isdmr +   &
     &          bxs*(vxr*bxs+vyr*byr+vzr*bzr-vdbstr)*isdmr)
        else
           rorst = ror; vxrst = vxr; vyrst = vyr; vzrst = vzr
           byrst = byr; bzrst = bzr
           vdbstr = vxrst*bxs+vyrst*byrst+vzrst*bzrst
           eerst = eer
        end if
        scrst(1:ncomp) = rorst/ror*scr(1:ncomp)
        rxrst = rorst*vxrst
        ryrst = rorst*vyrst
        rzrst = rorst*vzrst

!----- Step 5C. ----------------------------------------------------------|
! compute Ul** and Ur**
!
        abbx = abs(bxs)
        if (abbx >= eps) then
           sqrtrol = sqrt(rolst)
           sqrtror = sqrt(rorst)
           signbx = sign(1.0d0,bxs)
           invsumro = 1.0d0/(sqrtrol + sqrtror)

           temp = invsumro*(sqrtrol*vylst + sqrtror*vyrst  &
     &          + signbx*(byrst-bylst))
           vyldst = temp
           vyrdst = temp
           ryldst = rolst*temp
           ryrdst = rorst*temp

           temp = invsumro*(sqrtrol*vzlst + sqrtror*vzrst  &
     &          + signbx*(bzrst-bzlst))
           vzldst = temp
           vzrdst = temp
           rzldst = rolst*temp
           rzrdst = rorst*temp

           temp = invsumro*(sqrtrol*byrst + sqrtror*bylst  &
     &          + signbx*sqrtrol*sqrtror*(vyrst-vylst))
           byldst = temp
           byrdst = temp

           temp = invsumro*(sqrtrol*bzrst + sqrtror*bzlst  &
     &           + signbx*sqrtrol*sqrtror*(vzrst-vzlst))
           bzldst = temp
           bzrdst = temp

           temp = sm*bxs + vyldst*byldst + vzldst*bzldst
           eeldst = eelst - sqrtrol*signbx*(vdbstl - temp)
           eerdst = eerst + sqrtror*signbx*(vdbstr - temp)
           slst = sm - abbx/sqrtrol
           srst = sm + abbx/sqrtror
        else
           vyldst = vylst; vyrdst = vyrst; ryldst = rylst; ryrdst = ryrst
           vzldst = vzlst; vzrdst = vzrst; rzldst = rzlst; rzrdst = rzrst
           byldst = bylst; byrdst = byrst; bzldst = bzlst; bzrdst = bzrst
           eeldst = eelst; eerdst = eerst
           slst = 0.0d0
           srst = 0.0d0
        end if

!----- Step 6. ----------------------------------------------------------|
! compute flux
        msl = min(sl,0.0d0)
        msr = max(sr,0.0d0)
        mslst = min(slst,0.0d0)
        msrst = max(srst,0.0d0)

! sign(1.0,0.0)=+1 in Fortran, so sm==0 belongs to the left branch
        if (sm >= 0.0d0) then
           nflux(mden) = frol+(rolst-rol)*msl
           nflux(meto) = feel+(eelst-eel)*msl+(eeldst-eelst)*mslst
           nflux(mrvu) = frxl+(rxlst-rxl)*msl
           nflux(mrvv) = fryl+(rylst-ryl)*msl+(ryldst-rylst)*mslst
           nflux(mrvw) = frzl+(rzlst-rzl)*msl+(rzldst-rzlst)*mslst
           nflux(mbmu) = 0d0
           nflux(mbmv) = fbyl+(bylst-byl)*msl+(byldst-bylst)*mslst
           nflux(mbmw) = fbzl+(bzlst-bzl)*msl+(bzldst-bzlst)*mslst
           nflux(mst:med) = fscl(1:ncomp)+msl*(sclst(1:ncomp)-scl(1:ncomp))
        else if (sm < 0.0d0) then
           nflux(mden) = fror+(rorst-ror)*msr
           nflux(meto) = feer+(eerst-eer)*msr+(eerdst-eerst)*msrst
           nflux(mrvu) = frxr+(rxrst-rxr)*msr
           nflux(mrvv) = fryr+(ryrst-ryr)*msr+(ryrdst-ryrst)*msrst
           nflux(mrvw) = frzr+(rzrst-rzr)*msr+(rzrdst-rzrst)*msrst
           nflux(mbmu) = 0d0
           nflux(mbmv) = fbyr+(byrst-byr)*msr+(byrdst-byrst)*msrst
           nflux(mbmw) = fbzr+(bzrst-bzr)*msr+(bzrdst-bzrst)*msrst
           nflux(mst:med) = fscr(1:ncomp)+msr*(scrst(1:ncomp)-scr(1:ncomp))
        end if

      return
      end subroutine HLLD

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
    
! UpdateConsv moved to updateconsv.f90 (separate compilation unit,
! independent -gpu=maxregcount) -- see mhdturbulence-optimization-ideas.
subroutine EvaulateCh
  use basicmod
  use fluxmod
  use mpimod
  implicit none
  integer :: i,j,k,n
  real(8),parameter:: drate=0.1d0 ! 
! local variable
  real(8):: dh1l,dh2l,dh3l,dhl,dhd
  real(8):: ch1l,ch2l,ch3l,chl,chd
  real(8):: cts,css,cms
  real(8),parameter:: huge=1.0d90
  integer::theid

!$acc kernels
  chd = 0.0d0
  ch1l = 0.0d0; ch2l = 0.0d0; ch3l = 0.0d0
  dhd = huge
  dh1l =  huge; dh2l =  huge; dh3l =  huge
!$acc loop collapse(3) reduction(max:chd)
  do k=ks,ke
  do j=js,je
  do i=is,ie
            css  = svc(i,j,k,ncsp)**2
            cts  = css  &! cs^2+c_a^2
     &          + (svc(i,j,k,nbm1)**2+svc(i,j,k,nbm2)**2+svc(i,j,k,nbm3)**2)/svc(i,j,k,nden)
            cms  = sqrt((cts +sqrt(cts**2 &
     &      -4.0d0*css*svc(i,j,k,nbm1)**2/svc(i,j,k,nden)))/2.0d0)
            ch1l = ( abs(svc(i,j,k,nve1)) + cms )
            dh1l =  (x1a(i+1)-x1a(i))

            cms  = sqrt((cts +sqrt(cts**2 &
     &      -4.0d0*css*svc(i,j,k,nbm2)**2/svc(i,j,k,nden)))/2.0d0)
            ch2l = ( abs(svc(i,j,k,nve2)) + cms )
            dh2l = (x2a(j+1)-x2a(j)) 

            cms  = sqrt((cts +sqrt(cts**2 &
     &      -4.0d0*css*svc(i,j,k,nbm3)**2/svc(i,j,k,nden)))/2.0d0)
            ch3l = ( abs(svc(i,j,k,nve3)) + cms )
            dh3l = (x3a(k+1)-x3a(k))

         chl     = max(ch1l,ch2l,ch3l)
         dhl     = min(dh1l,dh2l,dh3l)
         chd     = max(chl,chd)
         dhd     = min(dhl,dhd)
  enddo
  enddo
  enddo
  bufinpmax(1) = chd
  bufinpmax(2) = dble(myid_w)
!$acc end kernels
  call MPImaxfind
!$acc kernels
  chd = bufoutmax(1)
  theid = int(bufoutmax(2)) 
  chg      =      chd
!$acc end kernels
  
  return
end subroutine  EvaulateCh

      subroutine DampPsi
      use basicmod
      use fluxmod
      implicit none
      integer :: i,j,k,n
      real(8),parameter:: alphabp=0.1d0 !
      real(8):: taui
      real(8):: dhl,dh1l,dh2l,dh3l
      real(8),parameter:: huge=1.0d90 

!$acc kernels  
      dh1l=huge
      dh2l=huge
      dh3l=huge
!$acc loop collapse(3) independent
      do k=ks,ke
      do j=js,je
      do i=is,ie
            dh1l = x1a(i+1)-x1a(i)
            dh2l = x2a(j+1)-x2a(j)
            dh3l = x3a(k+1)-x3a(k)

         dhl = min(dh1l,dh2l,dh3l)
         taui = alphabp * chg /dhl ! cm/s /cm => 1/s
         bp(i,j,k) = bp(i,j,k)*(1.0d0 - dt*taui) ! if dt = dtloc, damping by factor of (1.0-drate)
      enddo
      enddo
      enddo
!$acc end kernels

      return
end subroutine  DampPsi
