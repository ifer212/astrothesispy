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
cont219_plot = False
zoom_cont219_plot = True

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
    
    def plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path):
        """
            Figure 1 for NGC253 HR paper
        """
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
        my_cmap = plt.cm.get_cmap("jet")
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
            if 'p'not in str(row['Source']) and '_' not in str(row['Source']):
                axes[0].annotate(row['Source'], xy=(px, py), xytext=(px+pxm, py+pym), fontsize=sscfontsize,
                            va='center', color = 'k', zorder=4)     
        axes[0].set_ylim([0, cubo_219.shape[2]])
        axes[0].set_xlim([0, cubo_219.shape[3]])
        fig.savefig(fig_path+'219GHz_0.02x0.02_jet_3rms_newnames_ulvestad.pdf', bbox_inches='tight', transparent=True, dpi=800)
        plt.close()
        
    
if zoom_cont219_plot:
    def plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path, ind_fig = False):
        """
            Figure 2 for NGC253 HR paper
            ind_fig = True plots every SHC as an individual figure
            ind_fig = False plots all SHC in subpanels of same fig
        """
        figsize = 20
        sscfontsize = 32
        anotfontsize = 25
        labelsize = 40
        fontsize = 40
        
        gal_RA_lims = ['00:47:33.32187', '00:47:32.77219']
        gal_Dec_lims = ['-25:17:21.62597', '-25:17:15.15799']
        positions_crop = pd.read_excel(NGC253_path+'/Positions/Positions_crop.xlsx', header=0, na_values='-')

        subpositions_df = pd.read_excel(results_path+'Leroy_36GHz_python_219_nospind_v2.xlsx', header=0, na_values='-')
        subpositions_df['Source'] = subpositions_df['Source'].astype(str)
        
        cubo_219_path = NGC253_path+cont_path+location_path+'/MAD_CUB_219GHz_continuum.I.image.pbcor.fits'#MAD_CUB_NGC253_spw_25_briggs_continuum.image.pbcor.fits'
        cubo_219 = utiles_cubes.Cube_Handler('219', cubo_219_path)
        rms_219 = 1.307E-5
        cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cubo_219.fitsdata[0,0,:,:], rms_219, 3)
        cubo_219_aboverms = np.copy(cubo_219.fitsdata[0,0,:,:])
        cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan
        if ind_fig:
            for i, row in positions_crop.iterrows():
                if not pd.isna(row['px_low_new']):
                    
                    fig, axes, lims = plot_utiles.map_figure_starter(wcs=cubo_219.wcs, naxis=1, fsize=figsize, labelsize=labelsize, fontsize=fontsize,
                                                                    xlim_ra=gal_RA_lims, ylim_dec=gal_Dec_lims,ticksize = 12)
                    #if not pd.isna(row['px_low_new']):
                    axes[0].set_xlim(row['px_low_new'], row['px_low_new']+row['px_width'])
                    axes[0].set_ylim(row['py_low_new'], row['py_low_new']+row['py_height'])
        
                    axes[0].coords[1].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
                    axes[0].coords[0].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
                    axes[0].tick_params(axis="both", which='minor', length=8)
                    axes[0].coords[0].frame.set_linewidth(5)
                    axes[0].coords[1].frame.set_linewidth(5)
                    #axes[0].imshow(cubo_219.fitsdata[0,0,:,:], origin='lower', cmap =plt.cm.rainbow, interpolation=None,
                    #                       norm=LogNorm(vmin=rms_219, vmax=cubo_219.max/1.2), zorder=1)
                    axes[0].imshow(cubo_219_aboverms, origin='lower', cmap =plt.cm.jet, interpolation=None,
                                        norm=LogNorm(vmin=5*rms_219, vmax=cubo_219.max/1.2), zorder=1)
                    
                    axes[0].text(0.8, 0.94, r'$\rm{Cont. \, 219 \, GHz}$', transform = axes[0].transAxes, fontsize=anotfontsize)
                    axes[0].text(0.8, 0.91, r'$0.020^{\prime\prime} \times 0.019^{\prime\prime} $', transform = axes[0].transAxes, fontsize=anotfontsize)
                                        
                    axis_to_data = axes[0].transAxes + axes[0].transData.inverted()
                    points_data = axis_to_data.transform([0.1,0.1])
                    py = points_data[1]
                    px = points_data[0]
                    plot_utiles.plot_pc_scale(D_Mpc=D_Mpc, pcs=1, py=py+3, px=px, pixsize=cubo_219.pylen*3600, axis=axes[0], color='k',
                                            wcs=cubo_219.wcs, vertical=False, text_sep=4, fontsize = anotfontsize, lw=1.6, annotate=True, text_sep_percen=False)
                    plot_utiles.Beam_plotter(px=px, py=py, bmin=cubo_219.bmin*3600, bmaj=cubo_219.bmaj*3600,
                                        pixsize=cubo_219.pylen*3600, pa=cubo_219.bpa, axis=axes[0], wcs=cubo_219.wcs,
                                        color='k', linewidth=1.6, rectangle=True)
                    
                    fig.savefig(fig_path+'subcont_'+row['Location']+'_219GHz_0.02x0.02_jet5rms.pdf', bbox_inches='tight', transparent=True, dpi=400)
                    plt.close()
        else:
            maxis=2
            naxis=3
            fig, axes = plot_utiles.map_figure_starter(wcs=cubo_219.wcs, maxis=maxis, naxis=naxis, fsize=figsize, labelsize=labelsize, fontsize=fontsize,
                                                            xlim_ra=gal_RA_lims, ylim_dec=gal_Dec_lims,ticksize = 12)
            p = 0
            for i, row in positions_crop.iterrows():
                if not pd.isna(row['px_low_new']) and row['Location'] not in ['SHC13', 'SHC8', 'SHC9']:
                    p = int(row['subpanel'] )   
                    print(row['Location'])
                    axes[p].set_xlim(row['px_low_new'], row['px_low_new']+row['px_width'])
                    axes[p].set_ylim(row['py_low_new'], row['py_low_new']+row['py_height'])
                    axes[p].coords[1].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
                    axes[p].coords[0].set_ticks(size=14, width=2, color='k', exclude_overlapping=True, number = 5)
                    axes[p].tick_params(axis="both", which='minor', length=8)
                    axes[p].coords[0].frame.set_linewidth(5)
                    axes[p].coords[1].frame.set_linewidth(5)
                    axes[p].imshow(cubo_219_aboverms, origin='lower', cmap =plt.cm.jet, interpolation=None,
                                        norm=LogNorm(vmin=5*rms_219, vmax=cubo_219.max/1.2), zorder=1)
                    
                    axis_to_data = axes[p].transAxes + axes[p].transData.inverted()
                    points_data = axis_to_data.transform([0.1,0.1])
                    py = points_data[1]
                    px = points_data[0]
                    if p<3:
                        axes[p].set_xlabel(' ')
                    if p not in [0, 3]:
                        axes[p].set_ylabel(' ')
                    if row['Location'] == 'SHC4':
                        txtsep = 2
                    else: 
                        txtsep = 4
                    plot_utiles.plot_pc_scale(D_Mpc=D_Mpc, pcs=1, py=py+3, px=px, pixsize=cubo_219.pylen*3600, axis=axes[p], color='k',
                                            wcs=cubo_219.wcs, vertical=False, text_sep=txtsep, fontsize = anotfontsize, lw=1.6, annotate=True, text_sep_percen=False)
                    plot_utiles.Beam_plotter(px=px, py=py, bmin=cubo_219.bmin*3600, bmaj=cubo_219.bmaj*3600,
                                        pixsize=cubo_219.pylen*3600, pa=cubo_219.bpa, axis=axes[p], wcs=cubo_219.wcs,
                                        color='k', linewidth=1.6, rectangle=True)
                    
            for i, row in subpositions_df.iterrows():
                if row['subpanel']>=0:
                    p = int(row['subpanel'])   
                    pos_cont_219 = utiles.HMS2deg(ra=row['RA_219GHz'].replace('_', ' '), dec=row['Dec_219GHz'].replace('_', ' '))
                    px219, py219 = utiles_cubes.px_position(pos_cont_219[0], pos_cont_219[1], cubo_219.wcs)
                    pxm=row['subpxsum']
                    pym=row['subpysum']
                    if row['Source_altern_sub_final'] == '10c':
                        pxm += 4
                        pym += 3
                    axes[p].annotate(row['Source_altern_sub_final'], xy=(px219, py219), xytext=(int(px219+pxm), int(py219+pym)), fontsize=sscfontsize,
                                arrowprops={'headwidth': 0.1, 'headlength': 0.1, 'width':0.5, 'color': 'k'},
                                va='center', color = 'k', zorder=4) 
            fig.savefig(fig_path+'ALL_subcont_219GHz_0.02x0.02_jet5rms_newnames.pdf', pad_inches=0, bbox_inches='tight', transparent=True, dpi=400)
            plt.close()