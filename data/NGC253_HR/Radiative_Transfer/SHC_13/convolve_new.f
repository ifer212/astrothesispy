cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      
c     This routine will convolve model results for HC3N with the beam
c
      implicit real*8 (a-h,o-z)
      parameter (pi=3.14159265359d0,npts_par_max=10000,npix_max=300)
      dimension area_v71(npts_par_max),area_v61(npts_par_max),
     &     area_v72(npts_par_max),area_v0(npts_par_max),
     &     area_v41(npts_par_max),
     &     area_v41v71(npts_par_max),
     &     area_v51v73(npts_par_max),
     &     area_v62a(npts_par_max),
     &     area_v62b(npts_par_max),
     &     cont_v6(npts_par_max),
     &     param(npts_par_max),offx(npix_max),offy(npix_max),
     &     area71_grid(npix_max,npix_max),
     &     area61_grid(npix_max,npix_max),
     &     area41_grid(npix_max,npix_max),
     &     area47_grid(npix_max,npix_max),
     &     area57_grid(npix_max,npix_max),
     &     area62a_grid(npix_max,npix_max),
     &     area62b_grid(npix_max,npix_max),
     &     area72_grid(npix_max,npix_max),
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
c     &     'HC3Nsbsig1.1E+8cd5.6E+24q1.0nsh30rad1.5col4.0E16vt10'
c     &     'HC3Nsbsig1.1E+8cd5.6E+24q1.0nsh30rad1.5col4.0E16vt10full'
c     &     'HC3Nsbsig1.1E+8cd1.0E+25q1.5nsh30rad1.5col4.0E16vt10full'
c     &     'HC3Nsbsig1.1E+8cd6.5E+24q1.0nsh30rad1.5col4.0E16vt10full'   !Sel 1
c     &     'HC3Nsbsig1.1E+8cd2.0E+25q1.5nsh30rad1.5col2.0E17vt7.5full'  !Sel 2
c     &     'HC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.75col2E17vt7.5'      !Sel 3
c     &     'HC3Nsbsig1.1E+8cd6.5E+24q1.0nsh30rad1.5col4.0E16vt10'       !Sel 4
c     &     'HC3Nsbsig1.1E+8cd1.0E+25q1.0nsh30rad1.5col4.0E16vt10'       !Sel 5
c     &     'HC3Nsbsig5.5E+07cd1.0E+25q1.0nsh30rad1.5col1E17vt7.5'       !Sel 6
c     &     'HC3Nsbsig2.75E+07cd1.0E+25q1.0nsh30rad1.5col1E17vt7.5'      !Sel 7
c     &     'HC3Nsbsig1.4E+07cd1.2E+25q1.0nsh30rad1.5col1E17vt7.5'       !Sel 8
c     &     'HC3Nsbsig2.7E+07cd1.8E+25q1.0nsh30rad0.75col4E17vt7.5'      !Sel 9
c     &     'HC3Nsbsig7.0E+06cd1.5E+25q1.0nsh30rad1.5col1E17vt7.5'       !Sel 10
ccccccccccccc LTE MODELS 
cccccccccccc r=1.3
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad1.3col2.0E+17vt10_d8'       !Sel 999
cccccccccccc r=1.5
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad1.5col2.0E+17vt10'       !Sel 100
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad1.5col2.0E+17vt10_a1'       !Sel 101 modelo14
cccccccccccc r=0.8 
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10'       !Sel 200
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a1'       !Sel 201
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a2'       !Sel 202
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a3'       !Sel 203
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a4'       !Sel 204
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a5'       !Sel 205
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a6'       !Sel 206
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a7'       !Sel 207
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a8'       !Sel 208
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_a9'       !Sel 209
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b1'       !Sel 211
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b2'       !Sel 212
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b3'       !Sel 213
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b4'       !Sel 214
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b5'       !Sel 215
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b6'       !Sel 216
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b7'       !Sel 217
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b8'       !Sel 218
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_b9'       !Sel 219
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_c1'       !Sel 221
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_c2'       !Sel 222
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt10_c3'       !Sel 223
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7'       !Sel 300
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_a1'       !Sel 301
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b1'       !Sel 401
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b2'       !Sel 402
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b3'       !Sel 403
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b4'       !Sel 404
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b5'       !Sel 405
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b6'       !Sel 406
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b7'       !Sel 407
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b8'       !Sel 408
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_b9'       !Sel 409
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c1'       !Sel 501
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c2'       !Sel 502
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c3'       !Sel 503
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c4'       !Sel 504
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c5'       !Sel 505
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c6'       !Sel 506
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c7'       !Sel 507
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c8'       !Sel 508
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_c9'       !Sel 509
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_d1'       !Sel 601
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt7_d2'       !Sel 602
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d1'       !Sel 701
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d2'       !Sel 702
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d3'       !Sel 703
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d4'       !Sel 704
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d5'       !Sel 705
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col2.0E+17vt7_d6'       !Sel 706
cccccc NH2 = 1e25 q=1.5 R=1.5
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7'       !Sel 1000
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a1'       !Sel 1001
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a2'       !Sel 1002
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a3'       !Sel 1003
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a4'       !Sel 1004
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a5'       !Sel 1005
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a6'       !Sel 1006
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a7'       !Sel 1007
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a8'       !Sel 1008
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_a9'       !Sel 1009
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b1'       !Sel 1011
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b2'       !Sel 1012
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b3'       !Sel 1013
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b4'       !Sel 1014
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b5'       !Sel 1015
C     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+17vt7_b6'       !Sel 1016
cccccc NH2 = 1e25 q=1.5 R=0.8
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7'       !Sel 1100
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a1'       !Sel 1101	
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a2'       !Sel 1102
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a3'       !Sel 1103
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a4'       !Sel 1104
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a5'       !Sel 1105
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a6'       !Sel 1106
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a7'       !Sel 1107
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a8'       !Sel 1108
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_a9'       !Sel 1109
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b1'       !Sel 1111
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b2'       !Sel 1112
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b3'       !Sel 1113
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b4'       !Sel 1114
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b5'       !Sel 1115
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b6'       !Sel 1116
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b7'       !Sel 1117
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b8'       !Sel 1118
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_b9'       !Sel 1119
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c1'       !Sel 1121
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c2'       !Sel 1122
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c3'       !Sel 1123
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c4'       !Sel 1124
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c5'       !Sel 1125
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c6'       !Sel 1126
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c7'       !Sel 1127
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c8'       !Sel 1128
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_c9'       !Sel 1129
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_d1'       !Sel 1131
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad0.8col1.0E+17vt7_d2'       !Sel 1132
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7'       !Sel 2000
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a1'       !Sel 2001
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a2'       !Sel 2002
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a3'       !Sel 2003
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a4'       !Sel 2004
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a5'       !Sel 2005
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a6'       !Sel 2006
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a7'       !Sel 2007
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a8'       !Sel 2008
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_a9'       !Sel 2009
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b1'       !Sel 2011
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b2'       !Sel 2012
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b3'       !Sel 2013
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b4'       !Sel 2014
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b5'       !Sel 2015
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b6'       !Sel 2016
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b7'       !Sel 2017
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.0nsh30rad0.8col1.0E+17vt7_b8'       !Sel 2018
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10'       !Sel 3000
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a1'       !Sel 3001
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a2'       !Sel 3002
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a3'       !Sel 3003
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a4'       !Sel 3004
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a5'       !Sel 3005
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt10_a6'       !Sel 3006
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5'       !Sel 3010
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a1'       !Sel 3011
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a2'       !Sel 3012
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a3'       !Sel 3013
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a4'       !Sel 3014
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a5'       !Sel 3015
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a6'       !Sel 3016
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad1.5col2.0E+17vt5_a7'       !Sel 3017
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10'       !Sel 4000
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m1'       !Sel 4001
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m2'       !Sel 4002
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m3'       !Sel 4003
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m4'       !Sel 4004
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m5'       !Sel 4005
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m6'       !Sel 4006
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m7'       !Sel 4007
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m8'       !Sel 4008
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_m9'       !Sel 4009
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_n1'       !Sel 4011
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_n2'       !Sel 4012
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_n3'       !Sel 4013
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l3'       !Sel 4023
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l4'       !Sel 4024
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l5'       !Sel 4025
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l6'       !Sel 4026
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l7'       !Sel 4027
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad0.8col5.0E+17vt10_l8'       !Sel 4028
c     &     'LTHC3Nsbsig1.1E+08cd2.0E+25q1.5nsh30rad1.5col2.0E+17vt5_p1'       !Sel 4031
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5'       !Sel 5000
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a1'       !Sel 5001
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a2'       !Sel 5002
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a3'       !Sel 5003
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a4'       !Sel 5004
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a5'       !Sel 5005
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a6'       !Sel 5006
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a7'       !Sel 5007
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a8'       !Sel 5008
c     &     'LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5col1.0E+17vt5_a9'       !Sel 5009
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5'       !Sel 6000
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a1'       !Sel 6001
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a2'       !Sel 6002
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a3'       !Sel 6003
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a4'       !Sel 6004
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a5'       !Sel 6005
c     &     'LTHC3Nsbsig5.5E+07cd2.0E+25q1.0nsh30rad0.8col2.0E+17vt5_a6'       !Sel 6006
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_n3'       !Sel 4500
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a1'       !Sel 4501
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a2'       !Sel 4502
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a3'       !Sel 4503
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a4'       !Sel 4504
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a5'       !Sel 4505
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a6'       !Sel 4506
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a7'       !Sel 4507
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a8'       !Sel 4508
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad0.8col4.0E+17vt5_a9'       !Sel 4509
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad1.5col4.0E+17vt5'       !Sel 4600
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad1.5col4.0E+17vt5_a1'       !Sel 4601
c     &     'LTHC3Nsbsig1.1E+08cd4.0E+25q1.5nsh30rad1.5col4.0E+17vt5_a2'       !Sel 4602
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5'       !Sel 4700
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a1'       !Sel 4701
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a2'       !Sel 4702
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a3'       !Sel 4703
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a4'       !Sel 4704
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a5'       !Sel 4705
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a6'       !Sel 4706
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a7'       !Sel 4707
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a8'       !Sel 4708
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_a9'       !Sel 4709
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b1'       !Sel 4711
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b2'       !Sel 4712
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b3'       !Sel 4713
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b4'       !Sel 4714
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b5'       !Sel 4715
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_b6'       !Sel 4716
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c1'       !Sel 4721
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c2'       !Sel 4722
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c3'       !Sel 4723
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c4'       !Sel 4724
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c5'       !Sel 4725
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c6'       !Sel 4726
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c7'       !Sel 4727
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c8'       !Sel 4728
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c9'       !Sel 4729
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d1'       !Sel 4731
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d2'       !Sel 4732
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d3'       !Sel 4733
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d4'       !Sel 4734
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d5'       !Sel 4735
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d6'       !Sel 4736
c     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d7'       !Sel 4737
     &     'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_d8'       !Sel 4738
ccccccccccccc AGN
c     &     'NTHC3Nagnsig1.0E+25cd1.0E+25q1.5nsh30rad0.0col2.5E+17vt10'     !Sel 38
c	 ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dist_pc=3.5e6   !Distance to the source in pc
c      arcsec2_beam=pi*0.022*0.026/4.0/log(2.0)         !arcsec^2/beam
c      arcsec2_beam_345ghz=pi*0.027*0.033/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam=pi*0.022*0.020/4.0/log(2.0)         !arcsec^2/beam
      arcsec2_beam_345ghz=pi*0.028*0.034/4.0/log(2.0)         !arcsec^2/beam
      pixel_size_pc=0.02           !in pc
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
     & areav41v71,areav51v73,areav62a,areav62b,tr1,tr2 ! areas in Jy kms sr^-1
         read(3,*)pnormb,t2,t3,t35,t4,contv6   ! areas in Jy sr^-1
         npts_par=npts_par+1
         if(npts_par.gt.npts_par_max)then
            write(6,*)'WARNING: increase npts_par_max=',npts_par_max
            stop
         endif      
         param(npts_par)=pnorm*source_radius_pc
         area_v71(npts_par)=areav71*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v41(npts_par)=areav41*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v41v71(npts_par)=areav41v71*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v62a(npts_par)=areav62a*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v62b(npts_par)=areav62b*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v51v73(npts_par)=areav51v73*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v61(npts_par)=areav61*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
         area_v72(npts_par)=areav72*(pixel_size_pc/dist_pc)**2 !Jy km/s de 1 pixel
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
               area41_grid(i,j)=0.0
               area47_grid(i,j)=0.0
               area57_grid(i,j)=0.0
               area62a_grid(i,j)=0.0
               area62b_grid(i,j)=0.0
               area61_grid(i,j)=0.0
               area72_grid(i,j)=0.0
               areav0_grid(i,j)=0.0
               contv6_grid(i,j)=0.0
            elseif(r_pc.lt.param(1))then
               area41_grid(i,j)=area_v41(1)
               area47_grid(i,j)=area_v41v71(1)
               area57_grid(i,j)=area_v51v73(1)
               area62a_grid(i,j)=area_v62a(1)
               area62b_grid(i,j)=area_v62b(1)
               area71_grid(i,j)=area_v71(1)
               area61_grid(i,j)=area_v61(1)
               area72_grid(i,j)=area_v72(1)
               areav0_grid(i,j)=area_v0(1)
               contv6_grid(i,j)=cont_v6(1)			 
            else
               do k=1,npts_par-1
                  if(r_pc.gt.param(k).and.r_pc.le.param(k+1))then
                     slope=(area_v41(k+1)-area_v41(k))/
     &                    (param(k+1)-param(k))
                     area41_grid(i,j)=area_v41(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v41v71(k+1)-area_v41v71(k))/
     &                    (param(k+1)-param(k))
                     area47_grid(i,j)=area_v41v71(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v51v73(k+1)-area_v51v73(k))/
     &                    (param(k+1)-param(k))
                     area57_grid(i,j)=area_v51v73(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v62a(k+1)-area_v62a(k))/
     &                    (param(k+1)-param(k))
                     area62a_grid(i,j)=area_v62a(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v62b(k+1)-area_v62b(k))/
     &                    (param(k+1)-param(k))
                     area62b_grid(i,j)=area_v62b(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v71(k+1)-area_v71(k))/
     &                    (param(k+1)-param(k))
                     area71_grid(i,j)=area_v71(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v61(k+1)-area_v61(k))/
     &                    (param(k+1)-param(k))
                     area61_grid(i,j)=area_v61(k)+slope*
     &                    (r_pc-param(k))
                     slope=(area_v72(k+1)-area_v72(k))/
     &                    (param(k+1)-param(k))
                     area72_grid(i,j)=area_v72(k)+slope*
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
         area4171_conv=0.0
         area5173_conv=0.0
         area62a_conv=0.0
         area62b_conv=0.0
         area71_conv=0.0
         area61_conv=0.0
         area72_conv=0.0
         areav0_conv=0.0
         contv6_conv=0.0
         area41_conv_345ghz=0.0
         area47_conv_345ghz=0.0
         area57_conv_345ghz=0.0
         area62a_conv_345gh=0.0
         area62b_conv_345gh=0.0
         area71_conv_345ghz=0.0
         area61_conv_345ghz=0.0
         area72_conv_345ghz=0.0
         areav0_conv_345ghz=0.0
         contv6_conv_345ghz=0.0
         do i=1,npix
            do j=1,npix
               dist2=(offx(i)-offsetx)**2+offy(j)**2 !in pc
               area41_conv=area41_conv+area41_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area4171_conv=area4171_conv+area47_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area5173_conv=area5173_conv+area57_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area62a_conv=area62a_conv+area62a_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area62b_conv=area62b_conv+area62b_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area71_conv=area71_conv+area71_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area61_conv=area61_conv+area61_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area72_conv=area72_conv+area72_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               areav0_conv=areav0_conv+areav0_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               contv6_conv=contv6_conv+contv6_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2)
               area41_conv_345ghz=area41_conv_345ghz+area41_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area47_conv_345ghz=area47_conv_345ghz+area47_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area57_conv_345ghz=area57_conv_345ghz+area57_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area62a_conv_345gh=area62a_conv_345gh+area62a_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area62b_conv_345gh=area62b_conv_345gh+area62b_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area71_conv_345ghz=area71_conv_345ghz+area71_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area61_conv_345ghz=area61_conv_345ghz+area61_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               area72_conv_345ghz=area72_conv_345ghz+area72_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               areav0_conv_345ghz=areav0_conv_345ghz+areav0_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
               contv6_conv_345ghz=contv6_conv_345ghz+contv6_grid(i,j)*
     &              dexp(-dist2/2.0/sigma_beam_pc2_345ghz)
            enddo
         enddo
         area41_conv=area41_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam !Jy km/s /pc^2 x pc^2/beam
         area4171_conv=area4171_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area5173_conv=area5173_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area62a_conv=area62a_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area62b_conv=area62b_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area71_conv=area71_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area61_conv=area61_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam
         area72_conv=area72_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         areav0_conv=areav0_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         contv6_conv=contv6_conv/(2.0*pi*sigma_beam_pc2)*pc2_beam 
         area41_conv_345ghz=area41_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz !Jy km/s /pc^2 x pc^2/beam
         area47_conv_345ghz=area47_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area57_conv_345ghz=area57_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area62a_conv_345gh=area62a_conv_345gh/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area62b_conv_345gh=area62b_conv_345gh/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area71_conv_345ghz=area71_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area61_conv_345ghz=area61_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz
         area72_conv_345ghz=area72_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         areav0_conv_345ghz=areav0_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         contv6_conv_345ghz=contv6_conv_345ghz/
     &        (2.0*pi*sigma_beam_pc2_345ghz)*pc2_beam_345ghz 
         write(30,300)offsetx,areav0_conv,area71_conv,area61_conv,
     &        area72_conv,area41_conv,
     &        area4171_conv,area5173_conv,
     &        area62a_conv,area62b_conv,contv6_conv
         write(305,300)offsetx,areav0_conv_345ghz,area71_conv_345ghz,
     &        area61_conv_345ghz,area72_conv_345ghz,
     &        area41_conv_345ghz,area47_conv_345ghz,
     &        area57_conv_345ghz,area62a_conv_345gh,
     &        area62b_conv_345gh,contv6_conv_345ghz
      enddo
 300  format(f10.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,
     &     3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6,3x,es14.6)
         
      stop
      end
      
      
