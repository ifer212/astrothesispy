cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      
c     This routine will convolve model results for continuum with the beam
c
      implicit real*8 (a-h,o-z)
      parameter (pi=3.14159265359d0,npts_par_max=10000,npix_max=300,
     &     parsec=3.086e18,hp=6.626196d-27,cluz=2.997925d10,
     &               bolt=1.38064852d-16)
      dimension flux_cont_235ghz(npts_par_max),
     &     flux_cont_345ghz(npts_par_max),
     &     flux_ff_235ghz(npts_par_max),
     &     flux_ff_345ghz(npts_par_max),
     &     param(npts_par_max),offx(npix_max),offy(npix_max),
     &     flux_grid_235ghz(npix_max,npix_max),
     &     flux_grid_345ghz(npix_max,npix_max),
     &     ff_grid_235ghz(npix_max,npix_max),
     &     ff_grid_345ghz(npix_max,npix_max)
      character*100 folderroot,fileroot
      character*200 folderfileroot,fileinp,filepar,fileparconv,fileffpar

      do i=1,npix_max
         do j=1,npix_max
            flux_grid_235ghz(i,j)=0.0
            ff_grid_235ghz(i,j)=0.0
            flux_grid_345ghz(i,j)=0.0
            ff_grid_345ghz(i,j)=0.0
         enddo
      enddo
            
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Change this:
c

c External/diffuse emission?
      if_emis_ext=0
      tauext235_0=0.08
      temp_ext=120.0
      r_int_pc=0.
      r_ext_pc=0.8
      alpha=1.0
      beta=2.0
      
c Radius      
      source_radius_pc=s_rad      !in pc
      
c File      
      folderroot='~/Documents/Ed/transf/program/models/mymods/'
      fileroot=
     &     'dust_model'       !4700
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dist_pc=3.5e6   !Distance to the source in pc
c      arcsec2_beam=pi*0.022*0.026/4.0/log(2.0)         !arcsec^2/beam
c      arcsec2_beam_345ghz=pi*0.027*0.033/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam=pi*0.022*0.020/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam_345ghz=pi*0.028*0.034/4.0/log(2.0)         !arcsec^2/beam
      pixel_size_pc=0.02           ! 0.02 in pc
      pc2_beam=arcsec2_beam*(dist_pc*pi/(180.0*3600.0))**2 !pc^2/beam
      sigma_beam_pc2=pc2_beam/(2.0*pi)                     !beam~exp{-x^2/(2 sigma_beam_pc2)}
      pc2_beam_345ghz=arcsec2_beam_345ghz*(dist_pc*pi/(180.0*3600.0))**2 !pc^2/beam
      sigma_beam_pc2_345ghz=pc2_beam_345ghz/(2.0*pi)                     !beam~exp{-x^2/(2 sigma_beam_pc2)}
      write(6,*)'Area of the beam in parsec^2=',pc2_beam
      write(6,*)'Area of the pixel in parsec^2=',pixel_size_pc**2

      if(if_emis_ext.eq.1)then
         freq=235.0e9
         bnuext235=2.0*hp*freq*freq/cluz*freq/cluz/
     &        (dexp(hp*freq/bolt/temp_ext)-1.0)
         freq=345.0e9
         bnuext345=2.0*hp*freq*freq/cluz*freq/cluz/
     &        (dexp(hp*freq/bolt/temp_ext)-1.0)
      endif
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Building input files
c
      nch=0
      do i=1,100
         if(folderroot(i:i).ne.' ')then
            nch=nch+1
         else
            goto 5
         endif
      enddo
 5    nchroot=nch   
      nch=0
      do i=1,100
         if(fileroot(i:i).ne.' ')then
            nch=nch+1
         else
            goto 6
         endif
      enddo
 6    nchfile=nch

      folderfileroot=folderroot
      do i=1,nchfile
         write(folderfileroot(nchroot+i:nchroot+i),"(a1)")
     &        fileroot(i:i)
      enddo
      
      write(6,*)folderfileroot
      nch_folderfileroot=nchroot+nchfile

      fileinp=folderfileroot
      write(fileinp(nch_folderfileroot+1:nch_folderfileroot+4),"(a4)")
     &     '.inp'
      filepar=folderfileroot
      write(filepar(nch_folderfileroot+1:nch_folderfileroot+5),"(a5)")
     &     '_.par'
      fileffpar=folderfileroot
      write(fileffpar(nch_folderfileroot+1:nch_folderfileroot+7),"(a7)")
     &     '_.ffpar'


      write(6,*)fileinp
      write(6,*)filepar
      write(6,*)fileffpar
      

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Jy emitted by 1 pixel in the channels
c
      open (unit=1,file=fileinp,status='old')
      open (unit=2,file=filepar,status='old')
      open (unit=3,file=fileffpar,status='old')

      do i=1,2
         read(1,*)
      enddo
      read(1,*)nsh,rout_model   !Rout in cm
      do i=1,13
         read(1,*)
      enddo
      rout_model=rout_model/parsec         !Rout in pc
      read(1,*)nn,s1,s2,s3,s4,dist_model   !Dist in pc
      
      do i=1,13
         read(2,*)
         read(3,*)
      enddo
      npts_par=0
      do i=1,100000
         read(2,*,end=15)pnorm,t1,t2,t3,t4,t5,t6,flux_345ghz,t8,flux !  
         read(3,*)pnormff,t1,t2,t3,t4,t5,t6,ff_345ghz,t8,ff !  
         npts_par=npts_par+1
         if(npts_par.gt.npts_par_max)then
            write(6,*)'WARNING: increase npts_par_max=',npts_par_max
            stop
         endif
         flux=flux/(2.0*pi)*dist_model*dist_model/
     &        (pnorm*rout_model*parsec)*
     &        parsec*parsec !Jy/sr
         flux=flux*(pixel_size_pc/dist_pc)**2*1.0e3 !mJy de 1 pixel
         flux_345ghz=flux_345ghz/(2.0*pi)*dist_model*dist_model/
     &        (pnorm*rout_model*parsec)*
     &        parsec*parsec !Jy/sr
         flux_345ghz=flux_345ghz*(pixel_size_pc/dist_pc)**2*1.0e3 !mJy de 1 pixel
         ff=ff/(2.0*pi)*dist_model*dist_model/
     &        (pnormff*rout_model*parsec)*
     &        parsec*parsec !Jy/sr
         ff=ff*(pixel_size_pc/dist_pc)**2*1.0e3 !mJy de 1 pixel

         ff_345ghz=ff_345ghz/(2.0*pi)*dist_model*dist_model/
     &        (pnormff*rout_model*parsec)*
     &        parsec*parsec !Jy/sr
         ff_345ghz=ff_345ghz*(pixel_size_pc/dist_pc)**2*1.0e3 !mJy de 1 pixel
         
         param(npts_par)=pnorm*source_radius_pc
         
         flux_cont_235ghz(npts_par)=flux !mJy de 1 pixel
         flux_cont_345ghz(npts_par)=flux_345ghz !mJy de 1 pixel
         flux_ff_235ghz(npts_par)=ff !mJy de 1 pixel
         flux_ff_345ghz(npts_par)=ff_345ghz !mJy de 1 pixel

         
c         write(6,*)'par=',param(npts_par),' flux_cont=',
c     &        flux_cont_235ghz(npts_par)
      enddo
 15   continue
      
      close (1)
      close (2)
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Building a square of 2.1xR_source x 2.1xR_source
c

      npix=4.0*source_radius_pc/pixel_size_pc

      npix2=npix/2
      npix2=npix2*2
      if(npix2.eq.npix)then     !all ok, npix even
         write(6,*)'OK, npix is even:',npix
      else
         npix=npix+1
         write(6,*)'Increasing npix by 1 to be even:',npix
      endif
      if(npix.gt.npix_max)then
         write(6,*)'WARNING: increase npix_max=',npix_max,
     &        ' to npix=',npix
         stop
      endif      
      do i=1,npix
         offx(i)=-0.5*pixel_size_pc-npix/2*pixel_size_pc+i*pixel_size_pc
         offy(i)=-0.5*pixel_size_pc-npix/2*pixel_size_pc+i*pixel_size_pc
c         write(6,*)offx(i)
      enddo

      
      npix_ff=0
      do i=1,npix
         do j=1,npix
            r_pc=dsqrt(offx(i)**2+offy(j)**2)
            if(r_pc.gt.param(npts_par))then
               flux_grid_235ghz(i,j)=0.0
               flux_grid_345ghz(i,j)=0.0
               ff_grid_235ghz(i,j)=0.0
               ff_grid_345ghz(i,j)=0.0
            elseif(r_pc.lt.param(1))then
               flux_grid_235ghz(i,j)=flux_cont_235ghz(1)
               flux_grid_345ghz(i,j)=flux_cont_345ghz(1)
               ff_grid_235ghz(i,j)=flux_ff_235ghz(1)
               ff_grid_345ghz(i,j)=flux_ff_345ghz(1)
            else
               do k=1,npts_par-1
                  if(r_pc.gt.param(k).and.r_pc.le.param(k+1))then
                     slope=(flux_cont_235ghz(k+1)-flux_cont_235ghz(k))/
     &                    (param(k+1)-param(k))
                     slope_345ghz=(flux_cont_345ghz(k+1)-
     &                    flux_cont_345ghz(k))/(param(k+1)-param(k))
                     flux_grid_235ghz(i,j)=flux_cont_235ghz(k)+slope*
     &                    (r_pc-param(k))
                     flux_grid_345ghz(i,j)=flux_cont_345ghz(k)+
     &                    slope_345ghz*(r_pc-param(k))
                     
                     slope=(flux_ff_235ghz(k+1)-flux_ff_235ghz(k))/
     &                    (param(k+1)-param(k))
                     slope_345ghz=(flux_ff_345ghz(k+1)-
     &                    flux_ff_345ghz(k))/(param(k+1)-param(k))
                     ff_grid_235ghz(i,j)=flux_ff_235ghz(k)+slope*
     &                    (r_pc-param(k))
                     ff_grid_345ghz(i,j)=flux_ff_345ghz(k)+
     &                    slope_345ghz*(r_pc-param(k))
                  endif
               enddo
            endif
c            if(if_source_add_ff.eq.1)then
c               if(if_ff_punctual.eq.1)then
c                  if((i.eq.npix/2.or.i.eq.npix/2+1).and.
c     &                 (j.eq.npix/2.or.j.eq.npix/2+1))then
c                     npix_ff=npix_ff+1
c                     flux_grid_235ghz_ff(i,j)=
c     &                    flux_ff_235ghz_mjy
c                     flux_grid_345ghz_ff(i,j)=
c     &                    flux_ff_235ghz_mjy*(235.0/345.0)**0.1
c                  endif
c               else             ! extended free-free source
c                  if(r_pc.gt.source_radius_pc_ff)then
c                     flux_grid_235ghz_ff(i,j)=0.0
c                     flux_grid_345ghz_ff(i,j)=0.0
c                  else
c                     npix_ff=npix_ff+1
c                     flux_grid_235ghz_ff(i,j)=flux_ff_235ghz_mjy
c                     flux_grid_345ghz_ff(i,j)=                 
c     &                    flux_ff_235ghz_mjy*(235.0/345.0)**0.1
c                  endif
c               endif
c            endif
c               
c            write(29,290)offx(i),offy(j),
c     &           flux_grid_235ghz(i,j)
         enddo
      enddo

      
c      if(if_source_add_ff.eq.1)then
c         do i=1,npix
c            do j=1,npix
c               flux_grid_235ghz_ff(i,j)=flux_grid_235ghz_ff(i,j)/npix_ff
c               flux_grid_345ghz_ff(i,j)=flux_grid_345ghz_ff(i,j)/npix_ff
c            enddo
c         enddo
c      endif

      if(if_emis_ext.eq.1)then
         do i=1,npix
            do j=1,npix
               r_pc=dsqrt(offx(i)**2+offy(j)**2)         
               if(r_pc.lt.r_ext_pc.and.r_pc.gt.r_int_pc)then
                  tauext235=tauext235_0*(param(npts_par)/r_pc)**alpha
                  tauext345=tauext235*(345.0/235.0)**beta
                  flux_grid_235ghz(i,j)=flux_grid_235ghz(i,j)+bnuext235*
     &                 (1.0-dexp(-tauext235))*1.0e26*
     &                 (pixel_size_pc/dist_pc)**2
                  flux_grid_345ghz(i,j)=flux_grid_345ghz(i,j)+bnuext345*
     &                 (1.0-dexp(-tauext345))*1.0e26*
     &                 (pixel_size_pc/dist_pc)**2
c                  write(6,*)flux_grid_345ghz(i,j)
               endif
            enddo
         enddo
      endif
      
      
 290  format(f10.6,3x,f10.6,3x,es14.6)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Convolving!
c

      do k=1,npix/2+15
         offsetx=-0.5*pixel_size_pc+k*pixel_size_pc
c         write(6,*)offsetx
         flux_conv_235ghz=0.0
         flux_conv_235ghz_ff=0.0
         flux_conv_235ghz_smooth=0.0
         flux_conv_235ghz_smooth_ff=0.0
         flux_conv_345ghz=0.0
         flux_conv_345ghz_ff=0.0
         do i=1,npix
            do j=1,npix
               dist2=(offx(i)-offsetx)**2+offy(j)**2 !in pc
               flux_conv_235ghz=flux_conv_235ghz+flux_grid_235ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               flux_conv_235ghz_ff=flux_conv_235ghz_ff+
     &              ff_grid_235ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               flux_conv_235ghz_smooth=flux_conv_235ghz_smooth+
     &              flux_grid_235ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)               
               flux_conv_235ghz_smooth_ff=flux_conv_235ghz_smooth_ff+
     &              ff_grid_235ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)               
               flux_conv_345ghz=flux_conv_345ghz+flux_grid_345ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               flux_conv_345ghz_ff=flux_conv_345ghz_ff+
     &              ff_grid_345ghz(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
            enddo
         enddo
         flux_conv_235ghz=flux_conv_235ghz/
     &        (2.0*pi*sigma_beam_pc2)*pc2_beam !mJy /pc^2 x pc^2/beam
         flux_conv_235ghz_ff=flux_conv_235ghz_ff/
     &        (2.0*pi*sigma_beam_pc2)*pc2_beam !mJy /pc^2 x pc^2/beam
         flux_conv_235ghz_smooth=flux_conv_235ghz_smooth/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !mJy /pc^2 x pc^2/beam
         flux_conv_235ghz_smooth_ff=flux_conv_235ghz_smooth_ff/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !mJy /pc^2 x pc^2/beam
         flux_conv_345ghz=flux_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !mJy /pc^2 x pc^2/beam
         flux_conv_345ghz_ff=flux_conv_345ghz_ff/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !mJy /pc^2 x pc^2/beam

         
         write(31,300)offsetx,flux_conv_235ghz_smooth,flux_conv_345ghz,
     &        flux_conv_235ghz_smooth_ff,flux_conv_345ghz_ff,
     &        flux_conv_235ghz,flux_conv_235ghz_ff
         
      enddo
 300  format(f10.6,6(3x,es14.6))
         
      stop
      end
      
      
