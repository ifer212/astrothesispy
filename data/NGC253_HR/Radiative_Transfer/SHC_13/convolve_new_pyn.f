cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      
c     This routine will convolve model results for HC3N with the beam
c
      implicit real*8 (a-h,o-z)
      parameter (pi=3.14159265359d0,npts_par_max=10000,npix_max=300)
      dimension area_v71(npts_par_max),area_v61(npts_par_max),
     &     area_v712(npts_par_max),area_v612(npts_par_max),
     &     area_v72(npts_par_max),area_v0(npts_par_max),
     &     area_v722(npts_par_max),
     &     area_v41(npts_par_max),
     &     area_v41v71(npts_par_max),
     &     area_v51v73(npts_par_max),
     &     area_v6v71(npts_par_max),
     &     area_v62(npts_par_max),
     &     cont_v6(npts_par_max),
     &     param(npts_par_max),offx(npix_max),offy(npix_max),
     &     area71_grid(npix_max,npix_max),
     &     area61_grid(npix_max,npix_max),
     &     area712_grid(npix_max,npix_max),
     &     area612_grid(npix_max,npix_max),
     &     area41_grid(npix_max,npix_max),
     &     area47_grid(npix_max,npix_max),
     &     area57_grid(npix_max,npix_max),
     &     area671_grid(npix_max,npix_max),
     &     area62_grid(npix_max,npix_max),
     &     area72_grid(npix_max,npix_max),
     &     area722_grid(npix_max,npix_max),
     &     areav0_grid(npix_max,npix_max),
     &     contv6_grid(npix_max,npix_max)
      character*500 folderroot,fileroot
      character*500 folderfileroot,fileinp,filepar,fileparconv,
     &     fileparcont

c Olvida esto
      if_abs_central=0
      par_min_abs=0.0           !pc
      par_max_abs=0.35          !pc
      hp=6.616e-27
      freq=236.51e9
      cluz=2.997925e10
      bolt=1.38e-16
      tex=50.0
      tau_v0=2.0
      delta_v_abs=7.5 !10.0
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Change this:
c
      source_radius_pc=1.5      !in pc
      
      folderroot='~/Documents/Ed/transf/program/models/mymods/'
      fileroot=
     &     'm13_LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5vt5_b3'       !Sel 4721

ccccccccccccc AGN
c     &     'NTHC3Nagnsig1.0E+25cd1.0E+25q1.5nsh30rad0.0col2.5E+17vt10'     !Sel 38
c	 ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dist_pc=3.5e6   !Distance to the source in pc
c      arcsec2_beam=pi*0.022*0.026/4.0/log(2.0)         !arcsec^2/beam
c      arcsec2_beam_345ghz=pi*0.027*0.033/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam=pi*0.022*0.020/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam_345ghz=pi*0.028*0.034/4.0/log(2.0)         !arcsec^2/beam
      pixel_size_pc=0.02           !0.02 in pc
      pc2_beam=arcsec2_beam*(dist_pc*pi/(180.0*3600.0))**2 !pc^2/beam
      sigma_beam_pc2=pc2_beam/(2.0*pi)                     !beam~exp{-x^2/(2 sigma_beam_pc2)}
      pc2_beam_345ghz=arcsec2_beam_345ghz*(dist_pc*pi/(180.0*3600.0))**2 !pc^2/beam
      sigma_beam_pc2_345ghz=pc2_beam_345ghz/(2.0*pi)                     !beam~exp{-x^2/(2 sigma_beam_pc2)}
      
      write(6,*)'Area of the beam in parsec^2=',pc2_beam
      write(6,*)'Area of the pixel in parsec^2=',pixel_size_pc**2
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
      write(filepar(nch_folderfileroot+1:nch_folderfileroot+9),"(a9)")
     &     '_1rec.par'
      fileparcont=folderfileroot
      write(fileparcont(nch_folderfileroot+1:nch_folderfileroot+13),
     &     "(a13)")'_1rec.parcont'


      write(6,*)fileinp
      write(6,*)filepar
      write(6,*)fileparcont
      

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Jy km/s emitted by 1 pixel in the lines
c
      open (unit=1,file=fileinp,status='old')
      open (unit=2,file=filepar,status='old')
      open (unit=3,file=fileparcont,status='old')
      
      npts_par=0
      do i=1,100000
         read(2,*,end=15)pnorm,areav0,areav71,areav61,areav72,areav41,
     & areav712,areav51v73,areav6v71,areav62,areav612,areav722 ! areas in Jy kms sr^-1
         read(3,*)pnormb,t2,t3,t35,t4,contv6   ! areas in Jy sr^-1
         npts_par=npts_par+1
         if(npts_par.gt.npts_par_max)then
            write(6,*)'WARNING: increase npts_par_max=',npts_par_max
            stop
         endif      
         param(npts_par)=pnorm*source_radius_pc
         area_v71(npts_par)=areav71*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v712(npts_par)=areav712*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v41(npts_par)=areav41*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v62(npts_par)=areav62*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v6v71(npts_par)=areav6v71*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v51v73(npts_par)=areav51v73*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v61(npts_par)=areav61*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v612(npts_par)=areav612*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v72(npts_par)=areav72*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v722(npts_par)=areav722*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v0(npts_par)=areav0*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         cont_v6(npts_par)=contv6*(pixel_size_pc/dist_pc)**2 !Jy de 1 pixel
		 
         if(if_abs_central.eq.1)then
            if(param(npts_par).gt.par_min_abs.and.
     &           param(npts_par).lt.par_max_abs)then
               emis_abs=2.0*hp*freq*freq/cluz/cluz*freq/
     &              (dexp(hp*freq/bolt/tex)-1.0)*1.0e23*
     &              (1.0-dexp(-tau_v0))*delta_v_abs*
     &              (pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
               area_v0(npts_par)=area_v0(npts_par)*dexp(-tau_v0)+
     &              emis_abs
            endif
         endif
         
c         write(6,*)'par=',param(npts_par),' area_v71=',
c     &        area_v71(npts_par)
      enddo
 15   continue
      
      close (1)
      close (2)
      close (3)
      
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Building a square of 4xR_source x 4xR_source
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

      do i=1,npix
         do j=1,npix
            r_pc=dsqrt(offx(i)**2+offy(j)**2)
            if(r_pc.gt.param(npts_par))then
               area71_grid(i,j)=0.0
               area712_grid(i,j)=0.0
               area41_grid(i,j)=0.0
               area57_grid(i,j)=0.0
               area62_grid(i,j)=0.0
               area671_grid(i,j)=0.0
               area61_grid(i,j)=0.0
               area62_grid(i,j)=0.0
               area72_grid(i,j)=0.0
               area722_grid(i,j)=0.0
               areav0_grid(i,j)=0.0
               contv6_grid(i,j)=0.0
            elseif(r_pc.lt.param(1))then
               area41_grid(i,j)=area_v41(1)
               area57_grid(i,j)=area_v51v73(1)
               area62_grid(i,j)=area_v62(1)
               area671_grid(i,j)=area_v6v71(1)
               area71_grid(i,j)=area_v71(1)
               area712_grid(i,j)=area_v712(1)
               area61_grid(i,j)=area_v61(1)
               area612_grid(i,j)=area_v612(1)
               area72_grid(i,j)=area_v72(1)
               area722_grid(i,j)=area_v722(1)
               areav0_grid(i,j)=area_v0(1)
               contv6_grid(i,j)=cont_v6(1)			 
            else
               do k=1,npts_par-1
                  if(r_pc.gt.param(k).and.r_pc.le.param(k+1))then
                     slope=(area_v41(k+1)-area_v41(k))/
     &                    (param(k+1)-param(k))
                     area41_grid(i,j)=area_v41(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v51v73(k+1)-area_v51v73(k))/
     &                    (param(k+1)-param(k))
                     area57_grid(i,j)=area_v51v73(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v62(k+1)-area_v62(k))/
     &                    (param(k+1)-param(k))
                     area62_grid(i,j)=area_v62(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v6v71(k+1)-area_v6v71(k))/
     &                    (param(k+1)-param(k))
                     area671_grid(i,j)=area_v6v71(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v71(k+1)-area_v71(k))/
     &                    (param(k+1)-param(k))
                     area71_grid(i,j)=area_v71(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v712(k+1)-area_v712(k))/
     &                    (param(k+1)-param(k))
                     area712_grid(i,j)=area_v712(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v61(k+1)-area_v61(k))/
     &                    (param(k+1)-param(k))
                     area61_grid(i,j)=area_v61(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v612(k+1)-area_v612(k))/
     &                    (param(k+1)-param(k))
                     area612_grid(i,j)=area_v612(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v72(k+1)-area_v72(k))/
     &                    (param(k+1)-param(k))
                     area72_grid(i,j)=area_v72(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v722(k+1)-area_v722(k))/
     &                    (param(k+1)-param(k))
                     area722_grid(i,j)=area_v722(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v0(k+1)-area_v0(k))/
     &                    (param(k+1)-param(k))
                     areav0_grid(i,j)=area_v0(k)+slope*
     &                    (r_pc-param(k))
                     slope=(cont_v6(k+1)-cont_v6(k))/
     &                    (param(k+1)-param(k))
                     contv6_grid(i,j)=cont_v6(k)+slope*
     &                    (r_pc-param(k))
                  endif
               enddo
            endif
c            write(29,290)offx(i),offy(j),
c     &           area71_grid(i,j),area61_grid(i,j)
         enddo
      enddo
 290  format(f10.6,3x,f10.6,3x,es14.6,3x,es14.6)

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Convolving!
c

      do k=1,npix/2+15
         offsetx=-0.5*pixel_size_pc+k*pixel_size_pc
c         write(6,*)offsetx
         area41_conv=0.0
         area671_conv=0.0
         area5173_conv=0.0
         area62_conv=0.0
         area71_conv=0.0
         area61_conv=0.0
         area72_conv=0.0
         area77_conv=0.0
         area66_conv=0.0
         area88_conv=0.0
         areav0_conv=0.0
         contv6_conv=0.0
         area41_conv_345ghz=0.0
         area67_conv_345ghz=0.0
         area57_conv_345ghz=0.0
         area62_conv_345ghz=0.0
         area71_conv_345ghz=0.0
         area61_conv_345ghz=0.0
         area72_conv_345ghz=0.0
         area77_conv_345ghz=0.0
         area66_conv_345ghz=0.0
         area88_conv_345ghz=0.0
         areav0_conv_345ghz=0.0
         contv6_conv_345ghz=0.0
         do i=1,npix
            do j=1,npix
               dist2=(offx(i)-offsetx)**2+offy(j)**2 !in pc
               area41_conv=area41_conv+area41_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area671_conv=area671_conv+area671_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area5173_conv=area5173_conv+area57_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area62_conv=area62_conv+area62_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area71_conv=area71_conv+area71_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area61_conv=area61_conv+area61_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area72_conv=area72_conv+area72_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area77_conv=area77_conv+area712_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area66_conv=area66_conv+area612_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area88_conv=area88_conv+area722_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               areav0_conv=areav0_conv+areav0_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               contv6_conv=contv6_conv+contv6_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area41_conv_345ghz=area41_conv_345ghz+area41_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area67_conv_345ghz=area67_conv_345ghz+area671_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area57_conv_345ghz=area57_conv_345ghz+area57_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area62_conv_345ghz=area62_conv_345ghz+area62_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area71_conv_345ghz=area71_conv_345ghz+area71_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area61_conv_345ghz=area61_conv_345ghz+area61_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area72_conv_345ghz=area72_conv_345ghz+area72_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area77_conv_345ghz=area77_conv_345ghz+area712_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area66_conv_345ghz=area66_conv_345ghz+area612_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area88_conv_345ghz=area88_conv_345ghz+area722_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               areav0_conv_345ghz=areav0_conv_345ghz+areav0_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               contv6_conv_345ghz=contv6_conv_345ghz+contv6_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
            enddo
         enddo
         area41_conv=area41_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam !Jy km/s /pc^2 x pc^2/beam
         area671_conv=area671_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area5173_conv=area5173_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area62_conv=area62_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area71_conv=area71_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area61_conv=area61_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area72_conv=area72_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         area77_conv=area77_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area66_conv=area66_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area88_conv=area88_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         areav0_conv=areav0_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         contv6_conv=contv6_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         area41_conv_345ghz=area41_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !Jy km/s /pc^2 x pc^2/beam
         area67_conv_345ghz=area67_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area57_conv_345ghz=area57_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area62_conv_345ghz=area62_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area71_conv_345ghz=area71_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area61_conv_345ghz=area61_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area72_conv_345ghz=area72_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         area77_conv_345ghz=area77_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area66_conv_345ghz=area66_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area88_conv_345ghz=area88_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         areav0_conv_345ghz=areav0_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         contv6_conv_345ghz=contv6_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         write(30,300)offsetx,areav0_conv,area71_conv,area61_conv,
     &        area72_conv,area41_conv,
     &        area77_conv,area5173_conv,
     &        area671_conv,area62_conv,area66_conv,area88_conv,
     &        contv6_conv
         write(305,300)offsetx,areav0_conv_345ghz,area71_conv_345ghz,
     &        area61_conv_345ghz,area72_conv_345ghz,
     &        area41_conv_345ghz,area77_conv_345ghz,
     &        area57_conv_345ghz,area67_conv_345ghz,
     &        area62_conv_345ghz,area66_conv_345ghz,area88_conv_345ghz,
     &        contv6_conv_345ghz
      enddo
 300  format(f10.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,
     &     3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,
     &     3x,es14.6,3x,es14.6)
         
      stop
      end
      
      
