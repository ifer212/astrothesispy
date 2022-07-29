# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:57:36 2022

@author: Fer
"""


from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_cubes
from astrothesispy.utiles import utiles_plot as plot_utiles
from astrothesispy.utiles import u_conversion

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, FK5
from matplotlib.colors import LogNorm


plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rc('xtick', color='k', direction='in', labelsize=6)
plt.rc('ytick', color='k', direction='in', labelsize=6)




D_Mpc = 3.5
cont219_plot = True


# =============================================================================
# Global paths
# =============================================================================
NGC253_path = '/mnt/c/Users/Usuario/Documents/CAB/NGC253_HR/'
results_path = NGC253_path+'Results_v2/'
cont_path = 'Continuums/'
location_path = ''
fig_path = NGC253_path+'new_figs/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
# =============================================================================
# Continuum figures
# =============================================================================
if cont219_plot:
    gal_RA_lims = ['00:47:33.32187', '00:47:32.77219']
    gal_Dec_lims = ['-25:17:21.62597', '-25:17:15.15799']
    # Leroy2018 36GHz data
    #Leroy36df = pd.read_excel(results_path+'Leroy_36GHz.xlsx', header=0, na_values='-')
    #Leroy36df = pd.read_excel('/Users/frico/Documents/data/NGC253_HR/Results_v2/Cont219GHz.xlsx', header=0, na_values='-')
    Leroy36df = pd.read_excel(results_path+'Leroy_36GHz_python_219_nospind_v2.xlsx', header=0, na_values='-')
    Leroy36df['Source'] = Leroy36df['Source'].astype(str)
    # 36GHz data is from VLA with 0.096"x0.45" convolved to their 350GHz data 0.11"x0.11" (they made it circular from 0.105"x0.065")
    # Values measured in apertures centered on the peaks. 
    # The apertures have radius equal to the FWHM size of the source before deconvolution 
    # (i.e., to recover this add 1.9 pc in quadrature to the value in the table). See the text for more details.
    Leroy36df['FWHM_pc_conv'] = np.sqrt(Leroy36df['FWHM_pc']**2 + 1.9**2)  
    Leroy36df['FWHM_arcsec'] = np.sqrt(Leroy36df['FWHM_pc']**2 + 1.9**2)  
    levy_posdf = pd.read_excel(results_path+'/Levy2021_posistions.xlsx', header=0, na_values='-')

    cubo_219_path = NGC253_path+cont_path+location_path+'/MAD_CUB_219GHz_continuum.I.image.pbcor.fits'
    cubo_219 = utiles_cubes.Cube_Handler('219', cubo_219_path)
    rms_219 = 1.307E-5#8.629E-6
    cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cubo_219.fitsdata[0,0,:,:], rms_219, 3)
    cubo_219_aboverms = np.copy(cubo_219.fitsdata[0,0,:,:])
    cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan
    figsize = 20
    sscfontsize = 25
    anotfontsize = 25
    labelsize = 34
    fontsize = 34

    fig, axes = plot_utiles.map_figure_starter(wcs=cubo_219.wcs, maxis=1, naxis=1, fsize=figsize, labelsize=labelsize, fontsize=fontsize,
                                                    xlim_ra=gal_RA_lims, ylim_dec=gal_Dec_lims,ticksize = 12)
    

    axes[0].coords[1].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
    axes[0].coords[0].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
    axes[0].tick_params(axis="both", which='minor', length=8)
    axes[0].coords[0].frame.set_linewidth(5)
    axes[0].coords[1].frame.set_linewidth(5)
    my_cmap = plt.cm.get_cmap("jet")#.copy()
    #rgba = my_cmap(0.01)
    #my_cmap.set_bad((255,255,255, 0.3))
    axes[0].imshow(cubo_219_aboverms, origin='lower', cmap =plt.cm.jet, interpolation=None,
                           norm=LogNorm(vmin=3*rms_219, vmax=cubo_219.max/1.2), zorder=1)
    
    
    axes[0].text(0.8, 0.94, r'$\rm{Cont. \, 219 \, GHz}$', transform = axes[0].transAxes, fontsize=anotfontsize)
    axes[0].text(0.8, 0.91, r'$0.020^{\prime\prime} \times 0.019^{\prime\prime} $', transform = axes[0].transAxes, fontsize=anotfontsize)
    
    px, py = utiles_cubes.px_position('00_47_33.2750', '-25_17_21', cubo_219.wcs)
    plot_utiles.plot_pc_scale(D_Mpc=D_Mpc, pcs=5, py=py, px=px, pixsize=cubo_219.pylen*3600, axis=axes[0], color='k',
                             wcs=cubo_219.wcs, vertical=False, text_sep=1.08, fontsize = anotfontsize, lw=1.6, annotate=True)
    
    
    for i, row in Leroy36df.iterrows():
        pos_frv = utiles.HMS2deg(ra=row['RA_219GHz'].replace('_', ' '), dec=row['Dec_219GHz'].replace('_', ' '))
        pxfrv, pyfrv = utiles_cubes.px_position(pos_frv[0], pos_frv[1], cubo_219.wcs)

        if isinstance(row['RA_leroy_deg'], str):
            pos_ler = utiles.HMS2deg(ra=row['RA_219GHz'].replace('_', ' '), dec=row['Dec_219GHz'].replace('_', ' '))
            px, py = utiles_cubes.px_position(pos_ler[0], pos_ler[1], cubo_219.wcs)
        else:
            px, py = utiles_cubes.px_position(row['RA_leroy_deg'], row['Dec_leroy_deg'], cubo_219.wcs)
        pxm=5
        pym=5
        
        if row['Source'] == '14':
            pxm = 20
            pym = 10
        elif row['Source'] == '13':
            pxm = 10
            pym = 15
        elif row['Source'] == '12':
            pxm = -30
            pym = -15
        elif row['Source'] == '11':
            pxm = 5
            pym = -15
        elif row['Source'] == '10':
            pxm = 15
            pym = 15
        elif row['Source'] == '9':
            pxm = 5
            pym = -20
        elif row['Source'] == '8':
            pxm = 10
            pym = 15
        elif row['Source'] == '7':
            pxm = 5
            pym = 5
        elif row['Source'] == '6':
            pxm = -10
            pym = -25
        elif row['Source'] == '5':
            pxm = 6
            pym = 25
        elif row['Source'] == '4':
            pxm = 5
            pym = 20
        elif row['Source'] == '3':
            pxm = -20
            pym = 5
        elif row['Source'] == '2':
            pxm = 0
            pym = -15
        elif row['Source'] == '1':
            pxm = 0
            pym = 20
        elif row['Source'] == 'SN1':
            pxm = -10
            pym = 20
        elif row['Source'] == 'SN3':
            pxm = -30
            pym = -15
        else:
            pxm=5
            pym=5
       #if not pd.isna(row['Source_altern_sub_final']):
        if 'p'not in str(row['Source']) and '_' not in str(row['Source']):
            #axes[0].plot(px, py, markeredgecolor='k', markerfacecolor='g', marker='o')
            axes[0].annotate(row['Source'], xy=(px, py), xytext=(px+pxm, py+pym), fontsize=sscfontsize,
                        #arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                        va='center', color = 'k', zorder=4)     
            
    # Ulvestad1997 manual
    # Ulvestad&Antonucci Table 13
    do_table13 = False #  spectral index but worse positioning
    if do_table13:
        marker_size = 8
        hii_color = plot_utiles.orange 
        hii_marker = 'o'
        snr_color = plot_utiles.purple
        snr_marker = 'd'
        xray_marker = 'X'
        xray_color = plot_utiles.azure#plot_utiles.water
        th2_color = plot_utiles.aquagreen
        th2_marker = 'P'
        kin_color = plot_utiles.aquagreen
        kin_marker = '*'
        ulvestad2 = pd.read_csv(NGC253_path+'Ulvestad_2.txt', header=0, comment='#',delim_whitespace= True)                            
        
        ra_j2000 = []
        dec_j2000 = []
        pxm = 5
        pym = 5
        #utiles.HMS2deg(ra='', dec='')
        for l, line in ulvestad2.iterrows():
            ra_base = '00 45 '
            dec_base= '-25 33 '
            B1950=line['Name'].split('-')
            ra_b1950 = ra_base+B1950[0]
            dec_b1950 = dec_base+B1950[1]
            pos_b1950 = utiles.HMS2deg(ra=ra_b1950, dec=dec_b1950)
            b1950 = SkyCoord(ra=float(pos_b1950[0])*u.degree, dec=float(pos_b1950[1])*u.degree, frame='fk4') #B1950
            j2000 = b1950.transform_to(FK5(equinox='J2000'))
            ra_j2000.append(b1950.fk5.ra.value)
            dec_j2000.append(b1950.fk5.dec.value)
            px_uv, py_uv = cubo_219.wcs.wcs_world2pix(b1950.fk5.ra.value, b1950.fk5.dec.value, 1)
            #px_uv, py_uv = cubo_219.wcs.wcs_world2pix(j2000.ra.value, j2000.dec.value, 1)
            # SNR spectral_index < -0.4
            if line['spectral_index_2_6cm'] <= -0.4 or line['spectral_index_1.3_3.6cm'] <= -0.4:
                marker = snr_marker
                markeredgecolor = snr_color
                markerfacec = 'None'
            # Hii spectral index > -0.4
            elif line['spectral_index_2_6cm'] >= -0.4 or line['spectral_index_1.3_3.6cm'] >= -0.4:
                marker = hii_marker
                markeredgecolor = 'k'
                markerfacec = hii_color
            axes[0].plot(px_uv, py_uv, marker=marker, markerfacecolor=markerfacec, markeredgecolor=markeredgecolor, zorder=1, markersize=marker_size)
            if line['THsource'] != '-':
                axes[0].annotate(line['THsource'], xy=(px_uv, py_uv), xytext=(px_uv+pxm, py_uv+pym), fontsize=20,
                            #arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                            va='center', color = 'r', zorder=4)    
            else:
                labelua = 'UA'+str(l+1)
                axes[0].annotate(labelua, xy=(px_uv, py_uv), xytext=(px_uv+pxm, py_uv+pym), fontsize=20,
                            #arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                            va='center', color = 'r', zorder=4)   
                
        ulvestad2['RA_J2000'] = ra_j2000
        ulvestad2['Dec_J2000'] = dec_j2000
        
    
    # Ulvestad&Antonucci Table 6
    do_table6 = False # Better positions but no spectral index
    if do_table6:
        ulvestad2 = pd.read_excel('/Users/frico/Documents/data/NGC253_HR/ulvestad_table6.xlsx')                            
        
        ra_j2000 = []
        dec_j2000 = []
        pxm = 5
        pym = 5
        marker = 'o'
        markeredgecolor = 'k'
        markerfacec = 'None'
        #utiles.HMS2deg(ra='', dec='')
        for l, line in ulvestad2.iterrows():
            ra_base = '00 45 '
            dec_base= '-25 33 '
            ra_b1950 = ra_base+str(line['RA_1950'])
            dec_b1950 = dec_base+str(line['Dec_1950'])
            pos_b1950 = utiles.HMS2deg(ra=ra_b1950, dec=dec_b1950)
            b1950 = SkyCoord(ra=float(pos_b1950[0])*u.degree, dec=float(pos_b1950[1])*u.degree, frame='fk4') #B1950
            j2000 = b1950.transform_to(FK5(equinox='J2000'))
            ra_j2000.append(b1950.fk5.ra.value)
            dec_j2000.append(b1950.fk5.dec.value)
            px_uv, py_uv = cubo_219.wcs.wcs_world2pix(b1950.fk5.ra.value, b1950.fk5.dec.value, 1)
    
            axes[0].plot(px_uv, py_uv, marker=marker, markerfacecolor=markerfacec, markeredgecolor=markeredgecolor, zorder=1, markersize=marker_size)
            if pd.isna(line['THsource']):
                axes[0].annotate(line['Name'], xy=(px_uv, py_uv), xytext=(px_uv+pxm, py_uv+pym), fontsize=20,
                            #arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                            va='center', color = 'r', zorder=4)   
                 
            else:
                axes[0].annotate(line['THsource'], xy=(px_uv, py_uv), xytext=(px_uv+pxm, py_uv+pym), fontsize=20,
                            #arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                            va='center', color = 'r', zorder=4)   
                
    axes[0].set_ylim([0, cubo_219.shape[2]])
    axes[0].set_xlim([0, cubo_219.shape[3]])
    fig.savefig(fig_path+'219GHz_0.02x0.02_jet_3rms_newnames_ulvestad.pdf', bbox_inches='tight', transparent=True, dpi=800)
    plt.close()
    
    
