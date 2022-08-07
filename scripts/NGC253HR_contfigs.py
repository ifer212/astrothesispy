
import os

from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_cubes
from astrothesispy.utiles import utiles_plot as plot_utiles

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm


plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rc('xtick', color='k', direction='in', labelsize=6)
plt.rc('ytick', color='k', direction='in', labelsize=6)


# =============================================================================
# Continuum figures
# =============================================================================
def plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path, D_Mpc = 3.5, fig_name = '', fig_format = '.pdf'):
    """
        Figure 1 for NGC253 HR paper
    """
    gal_RA_lims = ['00:47:33.32187', '00:47:32.77219']
    gal_Dec_lims = ['-25:17:21.62597', '-25:17:15.15799']
    # Leroy2018 36GHz data
    Leroy36df = pd.read_excel(results_path+'Tables/Leroy_36GHz_python_219_nospind.xlsx', header=0, na_values='-')
    Leroy36df['Source'] = Leroy36df['Source'].astype(str)
    # 36GHz data is from VLA with 0.096"x0.45" convolved to their 350GHz data 0.11"x0.11" (they made it circular from 0.105"x0.065")
    # Values measured in apertures centered on the peaks. 
    # The apertures have radius equal to the FWHM size of the source before deconvolution 
    # (i.e., to recover this add 1.9 pc in quadrature to the value in the table). See the text for more details.
    Leroy36df['FWHM_pc_conv'] = np.sqrt(Leroy36df['FWHM_pc']**2 + 1.9**2)  
    Leroy36df['FWHM_arcsec'] = np.sqrt(Leroy36df['FWHM_pc']**2 + 1.9**2)  
    levy_posdf = pd.read_excel(results_path+'Tables/Levy2021_posistions.xlsx', header=0, na_values='-')

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
    plot_utiles.load_map_axes(axes[0], ticksize=14, ticklabelsize=labelsize-4, labelsize=labelsize,
                                  labelpad = -1, axcolor='k', ticknumber = 5,
                                  tickwidth = 2, axiswidth = 5, add_labels = True)
    axes[0].tick_params(axis="both", which='minor', length=8)
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
    fig_spath = f'{fig_path}NGC253/'
    if not os.path.exists(fig_spath):
        os.makedirs(fig_spath)
    fig.savefig(f'{fig_spath}{fig_name}219GHz{fig_format}', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
        
    
def plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path, ind_fig = False, D_Mpc = 3.5, fig_name = '', fig_format = '.pdf'):
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
    positions_crop = pd.read_excel(results_path+'Tables/Positions_crop.xlsx', header=0, na_values='-')
    subpositions_df = pd.read_excel(results_path+'Tables/Leroy_36GHz_python_219_nospind.xlsx', header=0, na_values='-')
    subpositions_df['Source'] = subpositions_df['Source'].astype(str)
    # 219GHz cube
    cubo_219_path = NGC253_path+cont_path+location_path+'/MAD_CUB_219GHz_continuum.I.image.pbcor.fits'
    cubo_219 = utiles_cubes.Cube_Handler('219', cubo_219_path)
    rms_219 = 1.307E-5
    cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cubo_219.fitsdata[0,0,:,:], rms_219, 3)
    cubo_219_aboverms = np.copy(cubo_219.fitsdata[0,0,:,:])
    cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan
    # 350GHz cube
    cubo_350_path = NGC253_path+cont_path+location_path+'/MAD_CUB_350GHz_continuum_mfs.I.manual.image.pbcor.fits'
    cubo_350 = utiles_cubes.Cube_Handler('350', cubo_350_path)
    rms_350 = 3.742e-5
    cubo_350_aboverms_mask = utiles_cubes.sigma_mask(cubo_350.fitsdata[0,0,:,:], rms_350, 5)
    cubo_350_aboverms = np.copy(cubo_350.fitsdata[0,0,:,:])
    cubo_350_aboverms[cubo_350_aboverms_mask.mask] = np.nan
    levels350 = [5*rms_350, 10*rms_350, 25*rms_350, 50*rms_350]
    if ind_fig:
        # Plotting each figure
        for i, row in positions_crop.iterrows():
            if not pd.isna(row['px_low_new']):
                fig, axes = plot_utiles.map_figure_starter(wcs=cubo_219.wcs, maxis=1, naxis=1, fsize=figsize, labelsize=labelsize, fontsize=fontsize,
                                                        xlim_ra=gal_RA_lims, ylim_dec=gal_Dec_lims,ticksize = 12)
                axes[0].set_xlim(row['px_low_new'], row['px_low_new']+row['px_width'])
                axes[0].set_ylim(row['py_low_new'], row['py_low_new']+row['py_height'])
                plot_utiles.load_map_axes(axes[p], ticksize=14, ticklabelsize=labelsize-4, labelsize=labelsize,
                                  labelpad = -1, axcolor='k', ticknumber = 5,
                                  tickwidth = 2, axiswidth = 5, add_labels = True)
                axes[0].tick_params(axis="both", which='minor', length=8)
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
                fig_spath = f'{fig_path}NGC253/Individual/'
                if not os.path.exists(fig_spath):
                    os.makedirs(fig_spath)
                fig.savefig(f'{fig_spath}subcont_{row["Location"]}_219GHz_0.02x0.02_jet5rms{fig_format}', bbox_inches='tight', transparent=True, dpi=400)
                plt.close()
    else:
        # Plotting inside same figure
        maxis=2
        naxis=3
        fig, axes = plot_utiles.map_figure_starter(wcs=cubo_219.wcs, maxis=maxis, naxis=naxis, fsize=figsize, labelsize=labelsize, fontsize=fontsize,
                                                        xlim_ra=gal_RA_lims, ylim_dec=gal_Dec_lims,ticksize = 12, hspace=0.08, wspace=0.33)
        p = 0
        for i, row in positions_crop.iterrows():
            if not pd.isna(row['px_low_new']) and row['Location'] not in ['SHC13', 'SHC8', 'SHC9']:
                p = int(row['subpanel'] )   
                print(f'\r\tPlotting {row["Location"]}')
                axes[p].set_xlim(row['px_low_new'], row['px_low_new']+row['px_width'])
                axes[p].set_ylim(row['py_low_new'], row['py_low_new']+row['py_height'])
                plot_utiles.load_map_axes(axes[p], ticksize=14, ticklabelsize=labelsize-4, labelsize=labelsize,
                                  labelpad = -1, axcolor='k', ticknumber = 5,
                                  tickwidth = 2, axiswidth = 5, add_labels = True)
                axes[p].tick_params(axis="both", which='minor', length=8)
                axes[p].imshow(cubo_219_aboverms, origin='lower', cmap =plt.cm.jet, interpolation=None,
                                    norm=LogNorm(vmin=5*rms_219, vmax=cubo_219.max/1.2), zorder=1)
                axes[p].contour(cubo_350_aboverms, colors='r', linewidths=1.6, zorder=2, levels= levels350,
                                transform=axes[p].get_transform(cubo_350.wcs))
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
                # 219GHz Beam
                plot_utiles.Beam_plotter(px=px, py=py, bmin=cubo_219.bmin*3600, bmaj=cubo_219.bmaj*3600,
                                    pixsize=cubo_219.pylen*3600, pa=cubo_219.bpa, axis=axes[p], wcs=cubo_219.wcs,
                                    color='k', linewidth=1.6, rectangle=True)
                # 350GHz Beam
                plot_utiles.Beam_plotter(px=px+12, py=py, bmin=cubo_350.bmin*3600, bmaj=cubo_350.bmaj*3600,
                                    pixsize=cubo_219.pylen*3600, pa=cubo_350.bpa, axis=axes[p], wcs=cubo_219.wcs,
                                    color='r', linewidth=1.6, rectangle=True)
                
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
                
        fig_spath = f'{fig_path}NGC253/'
        if not os.path.exists(fig_spath):
            os.makedirs(fig_spath)
        fig.savefig(f'{fig_spath}{fig_name}ALL_subcont_219GHz_and_350GHz{fig_format}', pad_inches=0, bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
        
        
def plot_moments(NGC253_path, cont_path, location_path, moments_path, fig_path, D_Mpc = 3.5, source = 'SHC_13', fig_name = '', fig_format = '.pdf'):
    """
        Figure 3 for NGC253 HR paper
    """
    anotfontsize = 40
    labelsize = 40
    fontsize = 40
    # Colorbar fontsizes
    cbarticksize=6
    cbartickwidth=1
    cbartickfont = 35
    cbarlabelfont = 40
    
    # Moments
    m02423v0 = 'NGC253_SHC_13_HC3N_v0_24_23_M0_aboveM0.fits'
    cubo_m02423v0 = utiles_cubes.Cube_Handler('219', moments_path+m02423v0)
    m02625v0 = 'NGC253_SHC_13_HC3N_v0_26_25_M0_aboveM0.fits'
    cubo_m02625v0 = utiles_cubes.Cube_Handler('219', moments_path+m02625v0)
    m02423v6 = 'NGC253_SHC_13_HC3N_v6=1_24_23_-1_+1_M0_aboveM0.fits'
    cubo_m02423v6 = utiles_cubes.Cube_Handler('219', moments_path+m02423v6)
    m02625v6 = 'NGC253_SHC_13_HC3N_v6=1_26_25_-1_+1_M0_aboveM0.fits'
    cubo_m02625v6 = utiles_cubes.Cube_Handler('219', moments_path+m02625v6)
    m02423v7 = 'NGC253_SHC_13_HC3N_v7=1_24_23_+1_-1_M0_aboveM0.fits'
    cubo_m02423v7 = utiles_cubes.Cube_Handler('219', moments_path+m02423v7)
    m02625v7 = 'NGC253_SHC_13_HC3N_v7=1_26_25_+1_-1_M0_aboveM0.fits'
    cubo_m02625v7 = utiles_cubes.Cube_Handler('219', moments_path+m02625v7)
    
    pixsize_arcsec =  cubo_m02423v0.header["CDELT2"]*3600
    pixsize_pc = pixsize_arcsec*D_Mpc*1e6/206265.0
    xcenter = 49
    ycenter = 20
    ix = 51
    jy = 25
    pxcenter_dist = np.sqrt((ix-xcenter)*(ix-xcenter) + (jy-ycenter)*(jy-ycenter))
    pxcenter_dist_pc = pxcenter_dist*pixsize_pc

    # Sigma masks
    cubo_m02423v0_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02423v0.fitsdata[0,0,:,:], cubo_m02423v0.header['INTSIGMA'], 3)
    cubo_m02423v0_aboverms = np.copy(cubo_m02423v7.fitsdata[0,0,:,:])
    cubo_m02423v0_aboverms[cubo_m02423v0_aboverms_mask.mask] = np.nan
    cubo_m02423v6_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02423v6.fitsdata[0,0,:,:], cubo_m02423v6.header['INTSIGMA'], 3)
    cubo_m02423v6_aboverms = np.copy(cubo_m02423v6.fitsdata[0,0,:,:])
    cubo_m02423v6_aboverms[cubo_m02423v6_aboverms_mask.mask] = np.nan
    cubo_m02423v7_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02423v7.fitsdata[0,0,:,:], cubo_m02423v7.header['INTSIGMA'], 3)
    cubo_m02423v7_aboverms = np.copy(cubo_m02423v7.fitsdata[0,0,:,:])
    cubo_m02423v7_aboverms[cubo_m02423v7_aboverms_mask.mask] = np.nan
    
    cubo_m02625v0_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02625v0.fitsdata[0,0,:,:], cubo_m02625v0.header['INTSIGMA'], 3)
    cubo_m02625v0_aboverms = np.copy(cubo_m02625v0.fitsdata[0,0,:,:])
    cubo_m02625v0_aboverms[cubo_m02625v0_aboverms_mask.mask] = np.nan
    cubo_m02625v6_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02625v6.fitsdata[0,0,:,:], cubo_m02625v6.header['INTSIGMA'], 3)
    cubo_m02625v6_aboverms = np.copy(cubo_m02625v6.fitsdata[0,0,:,:])
    cubo_m02625v6_aboverms[cubo_m02625v6_aboverms_mask.mask] = np.nan
    cubo_m02625v7_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02625v7.fitsdata[0,0,:,:], cubo_m02625v7.header['INTSIGMA'], 3)
    cubo_m02625v7_aboverms = np.copy(cubo_m02625v7.fitsdata[0,0,:,:])
    cubo_m02625v7_aboverms[cubo_m02625v7_aboverms_mask.mask] = np.nan
    
    momdict = {'m02434v0': {'wcs': cubo_m02423v0,'cubo': cubo_m02423v0_aboverms, 'panel': 0, 'label': r'$24-23 \quad v=0$'}, 
               'm02434v7': {'wcs': cubo_m02423v7,'cubo': cubo_m02423v7_aboverms, 'panel': 1, 'label': r'$24-23 \quad v_{7}=1$'}, 
               'm02434v6': {'wcs': cubo_m02423v6,'cubo': cubo_m02423v6_aboverms, 'panel': 2, 'label': r'$24-23 \quad v_{6}=1$'}, 
               'm02625v0': {'wcs': cubo_m02625v0,'cubo': cubo_m02625v0_aboverms, 'panel': 3, 'label': r'$26-25 \quad v=0$'}, 
               'm02625v7': {'wcs': cubo_m02625v7,'cubo': cubo_m02625v7_aboverms, 'panel': 4, 'label': r'$26-25 \quad v_{7}=1$'},
               'm02625v6': {'wcs': cubo_m02625v6,'cubo': cubo_m02625v6_aboverms, 'panel': 5, 'label': r'$26-25 \quad v_{6}=1$'} 
               }
    figsize = 20
    maxis=2
    naxis=3
    axcolor = 'k'
    fig, axes = plot_utiles.map_figure_starter(wcs=cubo_m02423v0.wcs, maxis=maxis, naxis=naxis, fsize=figsize,
                                               labelsize=labelsize, fontsize=fontsize, ticksize = 12, wspace = 0, hspace=0, axcolor=axcolor)
    maxlist = []
    rmslist = []
    for cmom in momdict:
        cubeplot = momdict[cmom]['cubo']
        axesind = momdict[cmom]['panel']
        cubeintrms = momdict[cmom]['wcs'].header['INTSIGMA']
        cubemax = np.nanmax(cubeplot)
        rmslist.append(cubeintrms)
        maxlist.append(cubemax)

    momdict['m02434v0']['rms'] = np.nanmin([momdict['m02434v0']['wcs'].header['INTSIGMA'], momdict['m02625v0']['wcs'].header['INTSIGMA']])
    momdict['m02625v0']['rms'] = np.nanmin([momdict['m02434v0']['wcs'].header['INTSIGMA'], momdict['m02625v0']['wcs'].header['INTSIGMA']])
    momdict['m02434v0']['max'] = np.nanmax([np.nanmax(momdict['m02434v0']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v0']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02625v0']['max'] = np.nanmax([np.nanmax(momdict['m02434v0']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v0']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02434v0']['eup'] = 131
    momdict['m02625v0']['eup'] = 153
    
    momdict['m02434v6']['rms'] = np.nanmin([momdict['m02434v6']['wcs'].header['INTSIGMA'], momdict['m02625v6']['wcs'].header['INTSIGMA']])
    momdict['m02625v6']['rms'] = np.nanmin([momdict['m02434v6']['wcs'].header['INTSIGMA'], momdict['m02625v6']['wcs'].header['INTSIGMA']])
    momdict['m02434v6']['max'] = np.nanmax([np.nanmax(momdict['m02434v6']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v6']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02625v6']['max'] = np.nanmax([np.nanmax(momdict['m02434v6']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v6']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02434v6']['eup'] = 849
    momdict['m02625v6']['eup'] = 871

    momdict['m02434v7']['rms'] = np.nanmin([momdict['m02434v7']['wcs'].header['INTSIGMA'], momdict['m02625v7']['wcs'].header['INTSIGMA']])
    momdict['m02625v7']['rms'] = np.nanmin([momdict['m02434v7']['wcs'].header['INTSIGMA'], momdict['m02625v7']['wcs'].header['INTSIGMA']])
    momdict['m02434v7']['max'] = np.nanmax([np.nanmax(momdict['m02434v7']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v7']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02625v7']['max'] = np.nanmax([np.nanmax(momdict['m02434v7']['wcs'].fitsdata[0,0,:,:]), np.nanmax(momdict['m02625v7']['wcs'].fitsdata[0,0,:,:])])
    momdict['m02434v7']['eup'] = 452
    momdict['m02625v7']['eup'] = 474
    
    shc13_RA_lims = ['00:47:33.215', '00:47:33.18']
    shc13_Dec_lims = ['-25:17:16.88', '-25:17:16.5']
    px_low, py_low = utiles_cubes.px_position(shc13_RA_lims[0], shc13_Dec_lims[0], momdict[cmom]['wcs'].wcs)
    px_up, py_up = utiles_cubes.px_position(shc13_RA_lims[1], shc13_Dec_lims[1], momdict[cmom]['wcs'].wcs)
    cubo_219_path = NGC253_path+cont_path+location_path+'/MAD_CUB_219GHz_continuum.I.image.pbcor.fits'
    cubo_219 = utiles_cubes.Cube_Handler('219', cubo_219_path)
    rms_219 = 1.307E-5#8.629E-6
    cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cubo_219.fitsdata[0,0,:,:], rms_219, 2.0)
    cubo_219_aboverms = np.copy(cubo_219.fitsdata[0,0,:,:])
    cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan
    levels219 = [5*rms_219, 10*rms_219, 50*rms_219, 100*rms_219]
    
    cubo_350_path = NGC253_path+cont_path+location_path+'/MAD_CUB_350GHz_continuum_mfs.I.manual.image.pbcor.fits'
    cubo_350 = utiles_cubes.Cube_Handler('350', cubo_350_path)
    rms_350 = 3.742e-5
    cubo_350_aboverms_mask = utiles_cubes.sigma_mask(cubo_350.fitsdata[0,0,:,:], rms_350, 5)
    cubo_350_aboverms = np.copy(cubo_350.fitsdata[0,0,:,:])
    cubo_350_aboverms[cubo_350_aboverms_mask.mask] = np.nan
    levels350 = [5*rms_350, 10*rms_350, 50*rms_350, 100*rms_350]
    
    for cmom in momdict:
        cubeplot = momdict[cmom]['cubo']
        axesind = momdict[cmom]['panel']
        cubeintrms = momdict[cmom]['wcs'].header['INTSIGMA']
        plot_utiles.load_map_axes(axes[axesind], ticksize=14, ticklabelsize=cbartickfont, labelsize=labelsize,
                                  labelpad = -1, axcolor=axcolor, ticknumber = 5,
                                  tickwidth = 2, axiswidth = 5, add_labels = True)
        axes[axesind].tick_params(axis="both", which='minor', length=8)
        mom0 = axes[axesind].imshow(cubeplot*1000, origin='lower',
                            vmin=momdict[cmom]['rms']*1000, vmax=momdict[cmom]['max']*1000,
                            cmap =plt.cm.jet, interpolation=None, zorder=1
                            )
        axes[axesind].contour(cubo_219_aboverms, colors='r', linewidths=1.4, zorder=2, levels= levels219,
            transform=axes[axesind].get_transform(cubo_219.wcs))
        axes[axesind].contour(cubo_350_aboverms, colors='k', linewidths=1.4, zorder=3, levels= levels350,
            transform=axes[axesind].get_transform(cubo_350.wcs))
        axes[axesind].set_xlim(px_low, px_up)
        axes[axesind].set_ylim(py_low, py_up)
        if axesind not in [0, 3]:
            axes[axesind].coords[1].set_ticklabel_visible(False)
        if axesind in [0, 1, 2]:
            axes[axesind].coords[0].set_ticklabel_visible(False)
            ax = axes[axesind]
            pos = ax.get_position()
            # Horizontal colorbar
            axis = 'x'
            orien = 'horizontal'
            sep=0.025
            width=0.02
            labelpad = -120
            framewidth=2
            label = r'mJy km s$^{-1}$'
            cb_axl = [pos.x0 + 0.01, pos.y0 + pos.height + sep,  np.round(pos.width,2)-2*0.01, width] 
            cb_ax = fig.add_axes(cb_axl)
            cbar = fig.colorbar(mom0, orientation=orien, cax=cb_ax)
            cbar.outline.set_linewidth(framewidth)
            cbar.ax.minorticks_off()
            cbar.set_label(label, labelpad=labelpad, fontsize=cbarlabelfont)
            cbar.ax.tick_params(axis=axis, direction='in')
            cbar.ax.tick_params(labelsize=cbartickfont)
            cbar.ax.tick_params(length=cbarticksize, width=cbartickwidth)
        axes[axesind].coords[0].display_minor_ticks(True)
        axes[axesind].coords[1].display_minor_ticks(True)
        axes[axesind].text(0.60, 0.90, momdict[cmom]['label'], transform = axes[axesind].transAxes, fontsize=anotfontsize, color='k')
        axes[axesind].text(0.66, 0.82, r'$\rm{E}_{\rm{up}}='+str(momdict[cmom]['eup'])+r'\,\rm{K}$', transform = axes[axesind].transAxes, fontsize=anotfontsize, color='k')

    pixsize = np.round(cubo_219.header['CDELT2']*3600,4)
    pcs = 1.5
    px = 20
    py = 5
    plot_utiles.Beam_plotter(px=px, py=py, bmin=cubo_m02423v0.bmin*3600, bmaj=cubo_m02423v0.bmaj*3600,
                                pixsize=cubo_m02423v0.pylen*3600, pa=cubo_m02423v0.bpa, axis=axes[0], wcs=cubo_m02423v0.wcs,
                                color='r', linewidth=0.8, rectangle=True)
    plot_utiles.Beam_plotter(px=px+9, py=py, bmin=cubo_350.bmin*3600, bmaj=cubo_350.bmaj*3600,
                                pixsize=cubo_350.pylen*3600, pa=cubo_350.bpa, axis=axes[0], wcs=cubo_m02423v0.wcs,
                                color='k', linewidth=0.8, rectangle=True)
    fig_spath = f'{fig_path}{source}'
    if not os.path.exists(fig_spath):
        os.makedirs(fig_spath)
    fig.savefig(f'{fig_spath}/{fig_name}{source}_HC3N_0moments.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()