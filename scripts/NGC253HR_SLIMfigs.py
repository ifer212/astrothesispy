from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_cubes
from astrothesispy.utiles import utiles_plot as plot_utiles
from astrothesispy.utiles import u_conversion
from astrothesispy.utiles import utiles_nLTEmodel

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import pandas as pd
import numpy as np
from copy import deepcopy
import astropy.units as u
from scipy import ndimage

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rc('xtick', color='k', direction='in', labelsize=6)
plt.rc('ytick', color='k', direction='in', labelsize=6)

# =============================================================================
# SLIM figures
# =============================================================================

def plot_SLIM2D(NGC253_path,  cont_path, location_path, fig_path, molecule = 'HC3Nvib_J24J26', source = 'SHC_13', D_Mpc = 3.5, fig_name = ''):
    """
        Figure 5 for NGC253 HR paper
    """
    rms_219 = 1.307E-5
    path = f'{NGC253_path}SHC/'
    slim_cube_path = f'{path}{source}/SLIM/'
    use_madcuba_moments = False
    if molecule == 'HC3Nvib_J24J26':
        crop_pre = 'Crop_VF_'
        corr_str = 'corrected_'
        crop_pre = ''
        corr_str = ''
    else:
        crop_pre = ''
        corr_str = ''
    if molecule == 'HC3Nvib_J24J26':
        crop = False
    else:
        crop = True
    if crop:
        crop_str = 'MAD_CUB_CROP_'
    else:
        crop_str = ''
        
    if use_madcuba_moments:
        moments_suf = ''
        moments_pre = 'MAD_CUB_Moments_0_'
        sigmastr = 'SIGMA' # For madcuba moments cube the sigma is not the integrated sigma of the cube
        moment_cube_path = f'{path}{source}/moments/Madcuba/'
    else:
        moments_suf = '_M0_aboveM0'
        moments_pre = ''
        sigmastr = 'INTSIGMA'
        moment_cube_path = f'{path}{source}/moments/'
        
    # SLIM Cubes
    columndens = crop_pre+'ColumnDensity_'+corr_str+molecule+'_'+source+'.fits'
    columndens_cube = utiles_cubes.Cube_Handler('logn', slim_cube_path+columndens)
    columndens_err = crop_pre+'ColumnDensity_err_'+molecule+'_'+source+'.fits'
    columndens_err_cube = utiles_cubes.Cube_Handler('logn_err', slim_cube_path+columndens_err)
    tex = crop_pre+'Tex_'+corr_str+molecule+'_'+source+'.fits'
    tex_cube = utiles_cubes.Cube_Handler('tex', slim_cube_path+tex)
    tex_err = crop_pre+'Tex_err_'+molecule+'_'+source+'.fits'
    tex_err_cube = utiles_cubes.Cube_Handler('tex_err', slim_cube_path+tex_err)
    FWHM = crop_pre+'FWHM_'+molecule+'_'+source+'.fits'
    FWHM_cube = utiles_cubes.Cube_Handler('FWHM', slim_cube_path+FWHM)
    FWHM_err = crop_pre+'FWHM_err_'+molecule+'_'+source+'.fits'
    FWHM_err_cube = utiles_cubes.Cube_Handler('FWHM_err', slim_cube_path+FWHM_err)
    vel = crop_pre+'vel_'+molecule+'_'+source+'.fits'
    vel_cube = utiles_cubes.Cube_Handler('vel', slim_cube_path+vel)
    vel_err = crop_pre+'vel_err_'+molecule+'_'+source+'.fits'
    vel_err_cube = utiles_cubes.Cube_Handler('vel_err', slim_cube_path+vel_err)
    
    # Cont cube
    cubo_219_path = NGC253_path+cont_path+location_path+'/MAD_CUB_219GHz_continuum.I.image.pbcor.fits'
    cont219_cube = utiles_cubes.Cube_Handler('219', cubo_219_path)
    cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cont219_cube.fitsdata[0,0,:,:], rms_219, 2.0)
    cubo_219_aboverms = np.copy(cont219_cube.fitsdata[0,0,:,:])
    cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan    
    RA, Dec = utiles_cubes.RADec_position(49, 20, tex_cube.wcs, origin=1)
    # J = 24-23
    m0_v0_2423 = moments_pre+'NGC253_'+source+'_HC3N_v0_24_23'+moments_suf+'.fits'
    m0_v0_2423_cube = utiles_cubes.Cube_Handler('m02423', moment_cube_path+m0_v0_2423)
    v0_2423_intrms = m0_v0_2423_cube.header[sigmastr]
    m0_v7_2423 = moments_pre+'NGC253_'+source+'_HC3N_v7=1_24_23_+1_-1'+moments_suf+'.fits'
    m0_v7_2423_cube = utiles_cubes.Cube_Handler('m72423', moment_cube_path+m0_v7_2423)
    v7_2423_intrms = m0_v7_2423_cube.header[sigmastr]
    m0_v6_2423 = moments_pre+'NGC253_'+source+'_HC3N_v6=1_24_23_-1_+1'+moments_suf+'.fits'
    m0_v6_2423_cube = utiles_cubes.Cube_Handler('m62423', moment_cube_path+m0_v6_2423)
    v6_2423_intrms = m0_v6_2423_cube.header[sigmastr]
    
    cube_dict = {'momv0': m0_v0_2423_cube, 'momv7': m0_v7_2423_cube, 'coldens': columndens_cube,
                 'tex': tex_cube, 'vel': vel_cube, 'fwhm': FWHM_cube}
    cube_dict_err = {'coldens_err': columndens_err_cube,
                'tex_err': tex_err_cube, 'vel_err': vel_err_cube, 'fwhm': FWHM_err_cube}
    
    columndens = crop_pre+'ColumnDensity_'+corr_str+molecule+'_'+source+'.fits'
    columndens_cube = utiles_cubes.Cube_Handler('logn', slim_cube_path+columndens)
    columndens_err = crop_pre+'ColumnDensity_err_'+molecule+'_'+source+'.fits'
    tex = crop_pre+'Tex_'+corr_str+molecule+'_'+source+'.fits'
    tex_cube = utiles_cubes.Cube_Handler('tex', slim_cube_path+tex)
    tex_err = crop_pre+'Tex_err_'+molecule+'_'+source+'.fits'
    tex_err_cube = utiles_cubes.Cube_Handler('tex_err', slim_cube_path+tex_err)
    FWHM = crop_pre+'FWHM_'+molecule+'_'+source+'.fits'
    FWHM_cube = utiles_cubes.Cube_Handler('FWHM', slim_cube_path+FWHM)
    FWHM_err = crop_pre+'FWHM_err_'+molecule+'_'+source+'.fits'
    vel = crop_pre+'vel_'+molecule+'_'+source+'.fits'
    vel_cube = utiles_cubes.Cube_Handler('vel', slim_cube_path+vel)
    vel_err = crop_pre+'vel_err_'+molecule+'_'+source+'.fits'

    if molecule == 'HC3Nvib_J24J26':
        if source == 'SHC_13':
            px_mid = 50-1 # Fits starts on 1, python on 0
            py_mid = 20-1 # Fits starts on 1, python on 0
        elif source == 'SHC_14':
            px_mid = 67-1 # Fits starts on 1, python on 0
            py_mid = 96-1 # Fits starts on 1, python on 0
    else:
        px_mid = 70-1 # Fits starts on 1, python on 0
        py_mid = 31-1 # Fits starts on 1, python on 0
    
    cbar_tickfont = 14
    cbar_labelfont = 30
    labelsize = 14
    fontsize = 14
    
    # Starting figure
    figsize = 10
    naxis = 2
    maxis = 2
    ticksize = 6
    cbar_pad = -65
    axcolor = 'k'
    wcs_plot = columndens_cube.wcs
    fig = plt.figure(figsize=(naxis*figsize*0.7, figsize))
    gs1 = gridspec.GridSpec(maxis, naxis)  
    gs1.update(wspace = 0.0, hspace=0.28, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    
    RA_start, Dec_start = utiles_cubes.RADec_position(47, 70, wcs_plot, origin = 1)
    RA_end, Dec_end = utiles_cubes.RADec_position(91, 118, wcs_plot, origin = 1)
    # Adding wcs frame
    wcs_plot.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    axes = []
    for i in range(naxis*maxis):
        axes.append(fig.add_subplot(gs1[i], aspect='equal', projection=wcs_plot))
        axes[i].tick_params(labelsize=labelsize)
        axes[i].tick_params(direction='in')
        axes[i].coords.frame.set_color(axcolor)
        axes[i].coords[0].set_major_formatter('hh:mm:ss.sss')
        axes[i].coords[1].set_ticks(size=14, width=2, color=axcolor, exclude_overlapping=True, number = 5)
        axes[i].coords[0].set_ticks(size=14, width=2, color=axcolor, exclude_overlapping=True, number = 5)
        axes[i].coords[0].frame.set_linewidth(2)
        axes[i].coords[1].frame.set_linewidth(2)
        axes[i].coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        axes[i].coords[0].display_minor_ticks(True)
        axes[i].coords[1].display_minor_ticks(True)
        if i in [1,3]:
            axes[i].set_yticklabels([])
            axes[i].tick_params(axis="both", which='minor', length=8)
            axes[i].xaxis.set_tick_params(top =True, labeltop=False)
            axes[i].yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
            axes[i].set_xlabel('RA (J2000)', fontsize = labelsize)
            axes[i].coords[1].set_ticklabel_visible(False)
            axes[i].coords[1].set_axislabel('')
        else:
            axes[i].tick_params(direction='in')
            axes[i].tick_params(axis="both", which='minor', length=8)
            axes[i].set_ylabel('Dec (J2000)', fontsize = labelsize, labelpad=-1)
        if i in [0,1]:
            axes[i].tick_params(axis="both", which='minor', length=8)
            axes[i].coords[0].set_ticklabel_visible(False)
        else:
            axes[i].set_xlabel('RA (J2000)', fontsize = labelsize)
        if source == 'SHC_14':
            axes[i].set_xlim([47-1, 91-1])
            axes[i].set_ylim([70-1, 118-1])
        elif source == 'SHC_13':
            axes[i].set_xlim([23-1, 68-1])
            axes[i].set_ylim([4-1, 41-1])
                
    # ColDens
    if molecule == 'HC3Nvib_J24J26': 
        logn_min = utiles.round_to_multiple(np.nanmin(columndens_cube.fitsdata[0,0,:,:]), 0.5)
        logn_min = 14.8
    else: 
        logn_min = 14.85 
    logn_max = utiles.round_to_multiple(np.nanmax(columndens_cube.fitsdata[0,0,:,:]), 0.5)
    axes[0].imshow(columndens_cube.fitsdata[0,0,:,:], norm=LogNorm(vmin=logn_min, vmax=logn_max), cmap =plt.cm.rainbow)
    plot_utiles.add_cbar(fig, axes[0], columndens_cube.fitsdata[0,0,:,:], r'logN(HC$_3$N) (cm$^{-2}$)', color_palette='rainbow', colors_len = 0,
                 orientation='h_short', sep=0.03, width=0.02, height=False,
                 ticks=[15, 15.5, 16, 16.5], Mappable=False, cbar_limits=[logn_min, logn_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                 discrete_colorbar=False, formatter = '%1.1f', norm='log', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                 )

        
    # Tex
    tex_min = utiles.round_to_multiple(np.nanmin(tex_cube.fitsdata[0,0,:,:]), 10)
    tex_min = 150 # 50
    tex_max = 900# utiles.round_to_multiple(np.nanmax(tex_cube.fitsdata[0,0,:,:]), 10)
    tex_ticks = list(np.linspace(tex_min, tex_max, 5)) #np.arange(tex_min, tex_max, 10, 5)
    axes[1].imshow(tex_cube.fitsdata[0,0,:,:], transform=axes[1].get_transform(tex_cube.wcs), cmap =plt.cm.rainbow, vmin=tex_min, vmax=tex_max)
    plot_utiles.add_cbar(fig, axes[1], tex_cube.fitsdata[0,0,:,:], r'T$_\text{vib}$ (K)', color_palette='rainbow', colors_len = 0,
                 orientation='h_short', sep=0.03, width=0.02, height=False, ticks = tex_ticks,
                 Mappable=False, cbar_limits=[tex_min, tex_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                 discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                 )
    
    # Vel
    if source == 'SHC_13':
        vel_min = 235
        vel_max = 265
    elif source == 'SHC_14':
        vel_min = 160
        vel_max = 240
    vel_ticks = list(np.linspace(vel_min, vel_max, 5))
    axes[2].imshow(vel_cube.fitsdata[0,0,:,:], transform=axes[2].get_transform(vel_cube.wcs), vmin=vel_min, vmax=vel_max, cmap =plt.cm.RdBu_r)
    plot_utiles.add_cbar(fig, axes[2], vel_cube.fitsdata[0,0,:,:], r'V$_{\text{LSR}}$ (km s$^{-1}$)', color_palette='RdBu_r', colors_len = 0,
                 orientation='h_short', sep=0.03, width=0.02, height=False,ticks=vel_ticks,
                 Mappable=False, cbar_limits=[vel_min, vel_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                 discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                 )
   
    # FWHM
    fwhm_min = 8
    fwhm_max = 60
    fwhm_ticks = list(np.linspace(fwhm_min, fwhm_max, 5))
    axes[3].imshow(FWHM_cube.fitsdata[0,0,:,:], transform=axes[3].get_transform(FWHM_cube.wcs), vmin=fwhm_min, vmax=fwhm_max, cmap =plt.cm.plasma)
    plot_utiles.add_cbar(fig, axes[3], FWHM_cube.fitsdata[0,0,:,:], r'FWHM (km s$^{-1}$)', color_palette='plasma', colors_len = 0,
                 orientation='h_short', sep=0.03, width=0.02, height=False, ticks=fwhm_ticks,
                 Mappable=False, cbar_limits=[fwhm_min, fwhm_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                 discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                 )
    
    levels_tex = [500]
    axes[0].contour(tex_cube.fitsdata[0,0,:,:], transform=axes[0].get_transform(tex_cube.wcs), colors='k', linewidths=2.0, zorder=1,
                     levels= levels_tex)
    axes[1].contour(tex_cube.fitsdata[0,0,:,:], transform=axes[1].get_transform(tex_cube.wcs), colors='k', linewidths=2.0, zorder=1,
                     levels= levels_tex)
    axes[2].contour(tex_cube.fitsdata[0,0,:,:], transform=axes[2].get_transform(tex_cube.wcs), colors='k', linewidths=2.0, zorder=1,
                    levels= levels_tex)
    axes[3].contour(tex_cube.fitsdata[0,0,:,:], transform=axes[3].get_transform(tex_cube.wcs), colors='k', linewidths=2.0, zorder=1,
                     levels= levels_tex)
    
    pixsize = np.round(columndens_cube.header['CDELT2']*3600,4)
    pcs = 1
    px = px_mid 
    py = 8
    plot_utiles.plot_pc_scale(D_Mpc, pcs, py, px, pixsize, axes[1], color='k', wcs = columndens_cube.wcs, vertical=False, text_sep=1.2, fontsize = fontsize, lw=2, annotate=True)
    plot_utiles.Beam_plotter(px=28, py=py, bmin=columndens_cube.bmin*3600, bmaj=columndens_cube.bmaj*3600,
                                pixsize=columndens_cube.pylen*3600, pa=columndens_cube.bpa, axis=axes[0], wcs=columndens_cube.wcs,
                                color='k', linewidth=0.8, rectangle=True)
    fig.savefig(f'{fig_path}{fig_name}{source}_SLIM_cubes_{molecule}_no_profile_noM0_v2.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
def plot_SLIMprofiles(NGC253_path, fig_path, molecule = 'HC3Nvib_J24J26', source = 'SHC_13', D_Mpc = 3.5, style = 'onecol', fig_name = ''):
    """
        Figure 6 for NGC253 HR paper
        style = "onecol" plots one column with two rows
        style = "twocol" plots two colunms with one row
    """

    only_SM = True
    labelsize = 32
    ticksize = 28
    # Cubes
    path = f'{NGC253_path}SHC/'
    final_results_path = f'{NGC253_path}Results_v2/'
    slim_cube_path = f'{path}{source}/SLIM/'
    madcub_path = f'{slim_cube_path}Figures_v8/'
    # Beams
    beam_orig = np.pi*0.022*0.020/(4*np.log(2))
    beam_345  = np.pi*0.028*0.034/(4*np.log(2))
    pc2_beam_orig = beam_orig*(D_Mpc*1e6*np.pi/(180.0*3600.0))**2
    pc2_beam_345  = beam_345*(D_Mpc*1e6*np.pi/(180.0*3600.0))**2
    pixsize_arcsec = 0.007
    
    # MADCUBA Temp profile
    ringave_file = 'SLIM_rings_average_v2.xlsx' # Spectra Average
    rings_df = pd.read_excel(final_results_path+ringave_file, header=0)
    
    tex_file = f'tex_{molecule}.csv' # Pixel average
    col_file = f'coldens_{molecule}.csv' # Pixel average
    madcol_df = pd.read_csv(madcub_path+col_file, header=0, sep=';')
    madcol_df_nouplim = madcol_df[madcol_df['value_err'] > 0]
    madcol_df_nouplim.reset_index(inplace=True)
    madtex_df = pd.read_csv(madcub_path+tex_file, header=0, sep=';')
    madtex_df_nouplim = madtex_df[madtex_df['value_err'] > 0]
    madtex_df_nouplim.reset_index(inplace=True)
    start_pc = 0
    end_pc = 1.5
    step = 0.1
    distances_pc = np.arange(start_pc, end_pc+step, step)
    dict_ring = {}
    for d, dist in enumerate(distances_pc):
        if d != 0:
            dist_mean_pc = np.round((distances_pc[d-1]+distances_pc[d])/2,2)
            ring_subdf = rings_df[rings_df['dist_pc']==dist_mean_pc]
            Tex_avering = ring_subdf['Tex'].tolist()[0]
            Tex_avering_err = ring_subdf['Tex_err'].tolist()[0]
            logN_avering = ring_subdf['logN'].tolist()[0]
            logN_avering_err = ring_subdf['logN_err'].tolist()[0]
            Tex_avering_SM = ring_subdf['Tex_SM'].tolist()[0]
            Tex_avering_SM_err = ring_subdf['Tex_SM_err'].tolist()[0]
            logN_avering_SM = ring_subdf['logN_SM'].tolist()[0]
            logN_avering_SM_err = ring_subdf['logN_SM_err'].tolist()[0]
            dist_subdf = madtex_df[(distances_pc[d-1] <= madtex_df['distance_pc']) & (madtex_df['distance_pc'] < distances_pc[d])]
            coldist_subdf = madcol_df[(distances_pc[d-1] <= madcol_df['distance_pc']) & (madcol_df['distance_pc'] < distances_pc[d])]
            Col_all = np.nanmean(coldist_subdf['value'])
            pxcount = len(coldist_subdf['value'])
            Tex_all = np.nanmean(dist_subdf['value'])
            dist_mean = np.nanmean(dist_subdf['distance_pc'])
            Tex_no_uplim_subdf = dist_subdf[dist_subdf['value_err'] > 0]
            Col_no_uplim_subdf = coldist_subdf[coldist_subdf['value_err'] > 0]
            if len(Tex_no_uplim_subdf) > 1:
                Tex_det, Tex_det_err = utiles.weighted_avg_and_stdv2(Tex_no_uplim_subdf['value'], Tex_no_uplim_subdf['value_err'])
            else:
                Tex_det = np.nanmean(Tex_no_uplim_subdf['value'])
                Tex_det_err = np.nanmean(Tex_no_uplim_subdf['value_err'])
            if len(Col_no_uplim_subdf) > 1:
                Col_det, Col_det_err = utiles.weighted_avg_and_stdv2(Col_no_uplim_subdf['value'], Col_no_uplim_subdf['value_err'])
            else:
                Col_det = np.nanmean(Col_no_uplim_subdf['value'])
                Col_det_err = np.nanmean(Col_no_uplim_subdf['value_err'])
            dict_ring[dist_mean_pc] =  {'Tex_all': Tex_all, 'Tex_det': Tex_det, 'Tex_det_err': Tex_det_err, 'Dist_mean_pc':dist_mean,
                                         'Col_all': Col_all, 'Col_det': Col_det, 'Col_det_err': Col_det_err,
                                         'Tex_ave_ring': Tex_avering, 'Tex_ave_ring_err': Tex_avering_err, 'Col_ave_ring': logN_avering, 'Col_ave_ring_err': logN_avering_err,
                                         'Tex_SM_ave_ring': Tex_avering_SM, 'Tex_SM_ave_ring_err': Tex_avering_SM_err, 'Col_SM_ave_ring': logN_avering_SM, 'Col_SM_ave_ring_err': logN_avering_SM_err,
                                         'px_count' : pxcount}
            
    mean_madcuba_df = pd.DataFrame(dict_ring).transpose()
    mean_madcuba_df['dist_ring_pc'] = mean_madcuba_df.index.tolist()
    mean_madcuba_df.reset_index(inplace=True)
    mean_madcuba_df.to_csv(final_results_path+'SHC_13_SLIM_Tex_and_logN_profiles.csv', header=True, sep=';', index=False, na_rep='-')   
    
    mean_madcuba_df['Col_ave_ring_rel_err'] = (10**mean_madcuba_df['Col_ave_ring_err'])*(1/np.log(10))/(10**mean_madcuba_df['Col_ave_ring'])
    if style == 'twocol':
        m=1
        n=2
        size_x = 10.
        size_y = 10
        fig = plt.figure(figsize=(size_x*n*1.2, size_y))
    elif style == 'onecol':
        m=2
        n=1
        size_x = 10.
        size_y = 10
        fig = plt.figure(figsize=(size_x, size_y*1.55))
    size_rat = float(n)/float(m)
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.125, hspace=0.0, top=0.95, bottom = 0.05) 
    axis = []
    axis_ind = []
    # Generating specified number of axis
    for i in range(m*n):
        row = (i // n)
        axis.append(fig.add_subplot(gs1[i]))
        
    indspec_col = '0.25'
    SMringslim_col = plot_utiles.redpink
    ringslim_col   = plot_utiles.redpink
    pxslim_col = plot_utiles.azure
    # Plotting all points
    for i, var in enumerate(['Tex', 'Col']):
        axis[i].tick_params(direction='in')
        axis[i].tick_params(axis="both", which='major', length=8)
        axis[i].tick_params(axis="both", which='minor', length=4)
        axis[i].xaxis.set_tick_params(which='both', top ='on')
        axis[i].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axis[i].tick_params(axis='both', which='major', labelsize=ticksize)
        axis[i].tick_params(labelleft=True,
                       labelright=False)
        axis[i].minorticks_on()
        axis[i].yaxis.set_minor_locator(AutoMinorLocator(4))
        axis[i].xaxis.set_minor_locator(AutoMinorLocator(2))
        if var == 'Tex':
            mad_df = madtex_df_nouplim
            mad_df_lims = madtex_df
        else:
            mad_df = madcol_df_nouplim
            mad_df_lims = madcol_df
        for r, row in mad_df.iterrows():
            if row['distance_pc']<= end_pc:
                if var == 'Col':
                    points_err_plot = (10**row['value_err'])*(1/np.log(10))/(10**row['value'])
                else:
                    points_err_plot = row['value_err']
                if row['distance_pc'] < 0.07 and var =='Tex':
                    continue
                else:
                     axis[i].errorbar(row['distance_pc'], row['value'], 
                                             yerr=points_err_plot,
                                             marker='o', markersize=5,
                                             markerfacecolor=indspec_col,
                                             markeredgecolor=indspec_col, markeredgewidth=0.8,
                                             ecolor=indspec_col,
                                             color =indspec_col,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=2)
        
        for r, row in mad_df_lims.iterrows():
            if row['distance_pc']<= end_pc:
                if row['value_err'] <0:
                    if var == 'Col':
                        errplot = row['value']*0.005
                    else:
                        errplot = row['value']*0.15
                    axis[i].errorbar(row['distance_pc'], row['value'], 
                                         uplims=True,
                                         yerr=errplot,
                                         marker='o', markersize=5,
                                         markerfacecolor='None',
                                         markeredgecolor='0.75', markeredgewidth=0.8,
                                         ecolor='0.75',
                                         color = '0.75',
                                         elinewidth= 0.7,
                                         barsabove= True,
                                         zorder=1)
        ring_val = []
        ring_dist = []
        ave_ring_val = []
        SMave_ring_val = []
        for ring in dict_ring:
            # Pixels Average
            if var == 'Col':
                points_err_plot = 1.5*(10**dict_ring[ring][var+'_det_err'])*(1/np.log(10))/(10**dict_ring[ring][var+'_det'])
            else:
                points_err_plot = dict_ring[ring][var+'_det_err']
            if ring < 0.1 and var =='Tex':
                uplim = 65
                axis[i].errorbar(dict_ring[ring]['Dist_mean_pc'], dict_ring[ring][var+'_det'], 
                             uplims=True,
                             yerr=65,
                             marker='o', markersize=10,
                             markerfacecolor=pxslim_col,
                             markeredgecolor=pxslim_col, markeredgewidth=0.8,
                             ecolor=pxslim_col,
                             color =pxslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=3)
            else:
                axis[i].errorbar(dict_ring[ring]['Dist_mean_pc'], dict_ring[ring][var+'_det'], 
                             yerr=points_err_plot,
                             marker='o', markersize=10,
                             markerfacecolor=pxslim_col,
                             markeredgecolor=pxslim_col, markeredgewidth=0.8,
                             ecolor=pxslim_col,
                             color =pxslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=3)
            # Spectra Average
            if dict_ring[ring][var+'_SM_ave_ring_err'] >0:
                if var == 'Col':
                    points_err_plot    = 1.5*(10**dict_ring[ring][var+'_ave_ring_err'])*(1/np.log(10))/(10**dict_ring[ring][var+'_ave_ring'])
                    points_err_plot_sm = 1.5*(10**dict_ring[ring][var+'_SM_ave_ring_err'])*(1/np.log(10))/(10**dict_ring[ring][var+'_SM_ave_ring'])

                else:
                    points_err_plot = dict_ring[ring][var+'_ave_ring_err']
                    points_err_plot_sm = dict_ring[ring][var+'_SM_ave_ring_err']
                
                if only_SM:
                    if ring < 0.3 and var =='Tex':
                        uplim = 65
                        axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_SM_ave_ring'],
                             uplims=True,
                             yerr=uplim,
                             marker='o', markersize=10,
                             markerfacecolor=SMringslim_col,
                             markeredgecolor=SMringslim_col, markeredgewidth=0.8,
                             ecolor=SMringslim_col,
                             color = SMringslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=4)
                    elif ring > 1.0 and var =='Col':
                        uplim = 65
                        axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_SM_ave_ring'],
                             uplims=True,
                             yerr=0.1,
                             marker='o', markersize=10,
                             markerfacecolor=SMringslim_col,
                             markeredgecolor=SMringslim_col, markeredgewidth=0.8,
                             ecolor=SMringslim_col,
                             color = SMringslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=4)
                    else:
                        axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_SM_ave_ring'], 
                                 yerr=points_err_plot,
                                 marker='o', markersize=10,
                                 markerfacecolor=SMringslim_col,
                                 markeredgecolor=SMringslim_col, markeredgewidth=0.8,
                                 ecolor=SMringslim_col,
                                 color = SMringslim_col,
                                 elinewidth= 1.6,
                                 barsabove= True,
                                 zorder=4)
                else:
                    axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_ave_ring'], 
                                 yerr=points_err_plot,
                                 marker='o', markersize=10,
                                 markerfacecolor=ringslim_col,
                                 markeredgecolor=ringslim_col, markeredgewidth=0.8,
                                 ecolor=ringslim_col,
                                 color = ringslim_col,
                                 elinewidth= 1.6,
                                 barsabove= True,
                                 zorder=4)
            else:
                # Upper limit
                if only_SM:
                    if var == 'Tex':
                        uplim = 65
                    else:
                        uplim = dict_ring[ring][var+'_SM_ave_ring']*0.2
                    axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_SM_ave_ring'],
                             uplims=True,
                             yerr=uplim,
                             marker='o', markersize=10,
                             markerfacecolor=SMringslim_col,
                             markeredgecolor=SMringslim_col, markeredgewidth=0.8,
                             ecolor=SMringslim_col,
                             color = SMringslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=4)
                else:
                    if var == 'Tex':
                        uplim = 65
                    else:
                        uplim = dict_ring[ring][var+'_SM_ave_ring']*0.2
                    axis[i].errorbar(dict_ring[ring]['Dist_mean_pc']+0.01, dict_ring[ring][var+'_ave_ring'],
                             uplims=True,
                             yerr=uplim,
                             marker='o', markersize=10,
                             markerfacecolor=ringslim_col,
                             markeredgecolor=ringslim_col, markeredgewidth=0.8,
                             ecolor=ringslim_col,
                             color = ringslim_col,
                             elinewidth= 1.6,
                             barsabove= True,
                             zorder=4)
            ring_val.append(dict_ring[ring][var+'_det'])
            ave_ring_val.append(dict_ring[ring][var+'_ave_ring'])
            SMave_ring_val.append(dict_ring[ring][var+'_SM_ave_ring'])
            ring_dist.append(dict_ring[ring]['Dist_mean_pc'])
        axis[i].plot(ring_dist, ring_val, marker='None',linestyle='-', linewidth=1.4, color = pxslim_col)
        if only_SM:
            axis[i].plot(np.array(ring_dist)+0.01, SMave_ring_val, marker='None',linestyle='-', linewidth=1.4, color = SMringslim_col)
        else:
            axis[i].plot(ring_dist, ave_ring_val, marker='None',linestyle='-', linewidth=1.4, color = ringslim_col)


    axis[0].set_ylim([0, 950])
    axis[1].set_ylim([14.8, 16.5])
    axis[0].set_ylabel(r'$T_\text{vib}$ (K)', fontsize = labelsize)
    axis[1].set_ylabel(r'log N(HC$_3$N) (cm$^{-2}$)', fontsize = labelsize)
    
    axis[1].set_xlabel('r (pc)', fontsize = labelsize)
    if style == 'onecol':
        axis[0].tick_params(labelbottom=False)   
    else:
        axis[0].set_xlabel('r (pc)', fontsize = labelsize)
    fig.savefig(f'{fig_path}{fig_name}{source}_SLIM_Tex_and_logN_profiles_dfcolors_newcols_1x2_v2.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
def plot_velprofiles(NGC253_path, source, fig_path, rad_transf_path, results_path, molecule = 'HC3Nvib_J24J26', modelname = 'model2', Rcrit = 0.85, D_Mpc = 3.5, style = 'onepanel', fig_name = ''):
    """
        Plots SLIM velocity profiles and Calculations for the cloud-cloud collision origin of SHC13 and its poss. outflow. mass
        style = 'onepanel' plots only one panel (one direction)
        style = 'twocol' plots both directions
    """
    labelsize = 34
    ticksize = 32
    fontsize = 30
    # Paths
    slim_cube_path = f'{NGC253_path}SHC/{source}/SLIM/'
    vel_path = slim_cube_path+'Vel_'+molecule+'_'+source+'.fits'            
    profiles_path = slim_cube_path+source+'_velprofiles/'
    my_model_path = f'{rad_transf_path}/models/mymods/'
    
    modelos = {
                    'model1': ['m13_LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5vt5_b3','dustsblum1.2E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.azure, [1, 1, 2.0]],
                    'model2': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.redpink , [1, 2.3, 1.9]],
                    'model3': ['m23_LTHC3Nsbsig5.5E+07cd1.0E+25q1.0nsh30rad1.5vt5_b7','dustsblum5.0E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.green, [1, 1, 1]],
                    'model4': ['m24_LTHC3Nsbsig5.5E+07cd1.0E+25q1.5nsh30rad1.5vt5_a7','dustsblum5.0E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.violet, [1, 1, 1]],
                    'model5': ['agn4_LTHC3Nagnsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_a6','dustagnlum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.dviolet, [1, 1, 1.0]],
                    'model6': ['agn5_LTHC3Nagnsig1.3E+07cd1.0E+25q1.0nsh30rad1.5vt5_a2','dustagnlum1.2E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.dazure, [1, 1, 1.0]],
                    'model7': ['agn9_LTHC3Nagnsig1.3E+07cd5.6E+24q1.0nsh30rad1.5vt5_x4','dustagnlum1.2E+10cd5.6E+24exp1.0nsh564rad17',
                                    1.5, plot_utiles.dgreen, [1, 1, 1.0]],
                    }

    modelo = modelos[modelname]
    obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = utiles_nLTEmodel.read_model_input(modelo, my_model_path, results_path, Rcrit)
    total_mass = np.nansum(Mgas_fromnH2_Msun_profile_corr)
    
    
    #Rings definition
    start_pc = 0
    end_pc = 1.5
    step = 0.1
    distances_pc = np.arange(start_pc, end_pc+step, step)

    xcenter = 48
    ycenter = 20
    vel_cube =  utiles_cubes.Cube_Handler('vels', vel_path)
    vel_cube_pixlen_pc = u_conversion.lin_size(D_Mpc, vel_cube.pylen*3600).to(u.pc).value
    plt.imshow(vel_cube.fitsdata[0,0,:,:], origin='lower')
    plt.plot(xcenter,ycenter,  marker='x')
    vel_cube_vel_mask =  np.ma.masked_where(vel_cube.fitsdata[0,0,:,:] >= 250, vel_cube.fitsdata[0,0,:,:], copy=True)
    vel_cube_below250 = np.copy(vel_cube.fitsdata[0,0,:,:])
    vel_cube_below250[vel_cube_vel_mask.mask] = np.nan
    vel_cube_below250_crop = vel_cube_below250[17:29,45:57]
    plt.imshow(vel_cube_below250_crop, origin='lower')
    # Number of pixels with v<250
    npix = np.count_nonzero(~np.isnan(vel_cube_below250_crop))
    sup_below250 = npix*vel_cube_pixlen_pc**2
    dist_dicts = []
    for ix in range(vel_cube.shape[3]):
        for jy in range(vel_cube.shape[2]):
            pxcenter_dist = np.sqrt((ix-xcenter)*(ix-xcenter) + (jy-ycenter)*(jy-ycenter))
            pxcenter_dist_pc = pxcenter_dist*vel_cube_pixlen_pc
            dist_dicts.append({'px': ix, 'py':jy, 'dist_pc':pxcenter_dist_pc})
    dist_df = pd.DataFrame(dist_dicts)

    px_count_dict = []
    for i, rad in enumerate(distances_pc):
        if i < len(distances_pc)-1:
            ring_dist = (distances_pc[i+1]+distances_pc[i])/2
            ring_vol = 4/3*np.pi*(distances_pc[i+1]**3-distances_pc[i]**3)
            ring_sup = np.pi*(distances_pc[i+1]**2-distances_pc[i]**2)
            ring_dif = distances_pc[i+1]-distances_pc[i]
            subdf = dist_df[(dist_df['dist_pc']>=distances_pc[i]) & (dist_df['dist_pc']<distances_pc[i+1])]
            if len(subdf)>0:
                print(f'{ring_dist:1.2f} \t {len(subdf)}')
                px_count_dict.append({'dist_pc': ring_dist, 'px_count': len(subdf), 'ring_vol':ring_vol, 'ring_sup':ring_sup, 'ring_dif':ring_dif})
    px_count_df = pd.DataFrame(px_count_dict)
    
    # Generating mass cube from models profile
    masses_cube = deepcopy(vel_cube.fitsdata[0,0,:,:])*np.nan
    dens_cube = deepcopy(vel_cube.fitsdata[0,0,:,:])*np.nan
    dens_sup_cube = deepcopy(vel_cube.fitsdata[0,0,:,:])*np.nan
    dist_dicts2 = []
    coldenh2 = 1e25
    nshells = len(rpc_profile)
    qexp = 1.5 
    colden_false = 300
    pc_to_cm = (1*u.pc).to(u.cm).value
    atomic_hydrogen_mass_kg = 1.6737236E-27*u.kg 
    for ix in range(masses_cube.shape[1]):
        for jy in range(masses_cube.shape[0]):
            pxcenter_dist = np.sqrt((ix-xcenter)*(ix-xcenter) + (jy-ycenter)*(jy-ycenter))
            pxcenter_dist_pc = pxcenter_dist*vel_cube_pixlen_pc
            if np.abs(pxcenter_dist_pc) <= 1.5:
                idx, close_rad = utiles.find_nearest(rpc_profile, pxcenter_dist_pc)
                if idx//2 ==0:
                    rint = rpc_profile[idx]
                    rout = rpc_profile[idx+1]
                else:
                    rint = rpc_profile[idx-1]
                    rout = rpc_profile[idx]
                rmed_shell = (rint+rout)/2
                dens_smooth = (rint/rmed_shell)**qexp
                sup_profile = (rout*pc_to_cm-rint*pc_to_cm)
                colden_false = colden_false+dens_smooth*(rout-rint)/pc_to_cm
                dens = nh2_profile_corr[idx]*2.*atomic_hydrogen_mass_kg.to(u.Msun).value
                dens_sup = nh2_profile_corr[idx]*ring_dif
                jdx, dis = utiles.find_nearest(px_count_df['dist_pc'], pxcenter_dist_pc)
                px_count = px_count_df.loc[jdx,'px_count']
                ring_vol = px_count_df.loc[jdx,'ring_vol']
                ring_sup = px_count_df.loc[jdx,'ring_sup']
                dens_sup = nh2_profile_corr[idx]*ring_sup
                mass = Mgas_fromnH2_Msun_profile_corr[idx]
                #dens = nh2_profile_corr[idx]
            else:
                mass = np.nan
                dens = np.nan
                px_count = np.nan
                dens_sup = np.nan
            masses_cube[jy,ix] = mass/px_count
            dens_cube[jy,ix] = dens
            dens_sup_cube[jy,ix] = dens_sup/px_count
            dist_dicts2.append({'px': ix, 'py':jy, 'dist_pc':pxcenter_dist_pc, 'px_count': px_count, 'mass':mass, 'dens':dens})
    
    
    
    dist_df2 = pd.DataFrame(dist_dicts2)
    dist_df2 = dist_df2.dropna()
    masses_cube_crop = masses_cube[3:40,22:68]
    # Applying vel <250 mask
    masses_cube_masked = deepcopy(masses_cube)
    masses_cube_masked[vel_cube_vel_mask.mask] = np.nan
    masses_cube_below250_crop = masses_cube_masked[17:29,45:57]
    npix_total = np.count_nonzero(~np.isnan(masses_cube))
    npix_outfl = np.count_nonzero(~np.isnan(masses_cube_below250_crop))
    fig = plt.figure()
    plt.imshow(masses_cube, origin='lower')
    fig.savefig(f'{fig_path}{source}_masses_cube.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()

    # Total mass from models inside region with vels <250km/s
    # with dens q=1.5
    # lolim
    Mass_below250_MSun = np.nansum(masses_cube_below250_crop)
    
    # Outflow mass assuming Mass eq. distr along cube
    # uplim
    outflow_pc_percen = npix_outfl/npix_total
    Mass_below250_MSun_uplim = outflow_pc_percen *total_mass
    
    # Assuming average surface density 
    ave_dens_sup = total_mass/(np.pi*1.5**2) # Msun/pc2
    average_vol_dens = total_mass/(4/3*np.pi*1.5**3) # Msun/pc3
    mass_2 = ave_dens_sup*npix_outfl*(vel_cube_pixlen_pc**2)
    
    routflow = np.sqrt((npix_outfl*vel_cube_pixlen_pc**2)/np.pi)
    vel_diff = 21 # km/s
    age1 = 1e4
    age2 = 5e4
    factor_conv = 0.5*(1*u.Msun).to(u.g)*(1*u.km**2/u.s**2).to(u.cm**2/u.s**2)
    Mass_below250_MSun_uplim
    Outflow_kin_energy_erg_j = 1e41*Mass_below250_MSun_uplim*vel_diff**2
    Outflow_kin_energy_erg_f = (factor_conv*Mass_below250_MSun_uplim*vel_diff**2).to(u.erg)
    Outflow_kin_energy_erg_f_mass1e5 = (factor_conv*1e5*vel_diff**2).to(u.erg)
    L_j_age1 = (Outflow_kin_energy_erg_j*u.erg/(age1*(1*u.yr).to(u.s))).to(u.Lsun).value
    L_j_age2 = (Outflow_kin_energy_erg_j*u.erg/(age2*(1*u.yr).to(u.s))).to(u.Lsun).value
    L_f_age1 = (Outflow_kin_energy_erg_f/(age1*(1*u.yr).to(u.s))).to(u.Lsun).value
    L_f_age2 = (Outflow_kin_energy_erg_f/(age2*(1*u.yr).to(u.s))).to(u.Lsun).value
    L_f_age1_mass1e5 = (Outflow_kin_energy_erg_f_mass1e5/(age1*(1*u.yr).to(u.s))).to(u.Lsun).value
    L_f_age2_mass1e5 = (Outflow_kin_energy_erg_f_mass1e5/(age2*(1*u.yr).to(u.s))).to(u.Lsun).value

    # Haowrth2015 clou cloud 
    # small cloud compression
    time_compression = (routflow*(1*u.pc).to(u.km)/(vel_diff*(u.km/u.s))).to(u.yr).value
    
    # Perpendicular profile to gal rotation
    profile_NW_SE = pd.read_csv(profiles_path+'vel_NW_SE.csv')
    profile_NW_SE.columns = ['px', 'V']
    rmid = len(profile_NW_SE)/2
    profile_NW_SE['px_res'] = profile_NW_SE['px']-rmid
    profile_NW_SE['dist_pc'] = profile_NW_SE['px_res']*vel_cube_pixlen_pc

    # Parallel profile to gal rotation
    profile_NE_SW = pd.read_csv(profiles_path+'vel_NE_SW.csv')
    profile_NE_SW.columns = ['px', 'V']
    rmid = len(profile_NE_SW)/2
    profile_NE_SW['px_res'] = profile_NE_SW['px']-rmid
    profile_NE_SW['dist_pc'] = profile_NE_SW['px_res']*vel_cube_pixlen_pc


    figsize = 14
    if style=='twocol':
        naxis = 2
        maxis = 1
        fig = plt.figure(figsize=(figsize*1.95, figsize*1.13))
    elif style == 'onepanel':
        naxis = 1
        maxis = 1
        fig = plt.figure(figsize=(figsize*1.15, figsize*1.))
    else: 
        naxis = 1
        maxis = 2
        fig = plt.figure(figsize=(figsize, figsize*1.55))
    gs = gridspec.GridSpec(maxis, naxis)
    if style == 'onepanel':
        gs.update(wspace = 0.0, hspace=0.0, top=0.975, bottom = 0.07)
    else:
        gs.update(wspace = 0.125, hspace=0.0, top=0.95, bottom = 0.05)
    
    axis = []
    axis.append(fig.add_subplot(gs[0]))
    if style != 'onepanel':
        axis.append(fig.add_subplot(gs[1]))
    
    axis[0].plot(profile_NW_SE['dist_pc'],profile_NW_SE['V'],color= 'k', linestyle='-',label='data', marker='',lw=1.4)
    if style != 'onepanel':
        axis[1].plot(profile_NE_SW['dist_pc']*-1,profile_NE_SW['V'],color= 'k', linestyle='-',label='data', marker='', lw=1.4)
    axis[0].set_ylim([242, 263])
    axis[0].set_xlim([-1.58, 1.58])
    if style != 'onepanel':
        axis[1].set_ylim([242, 263])
        axis[1].set_xlim([-1.58, 1.58])
    
    axis[0].text(0.15, 0.95, 'SE-NW',
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axis[0].transAxes)
    if style != 'onepanel':
        axis[1].text(0.15, 0.95, 'NE-SW',
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axis[1].transAxes)
    
    for v,ax in enumerate(axis):
        axis[v].tick_params(direction='in')
        axis[v].tick_params(axis="both", which='major', length=8)
        axis[v].tick_params(axis="both", which='minor', length=4)
        axis[v].xaxis.set_tick_params(which='both', top ='on')
        axis[v].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axis[v].tick_params(axis='both', which='major', labelsize=ticksize, width=1.75)
        for axs in ['top', 'bottom', 'left', 'right']:
            axis[v].spines[axs].set_linewidth(1.5)  # change width
        axis[v].tick_params(labelright=False)
    if style=='twocol':
        axis[0].set_xlabel(r'$r$ (pc)', fontsize=labelsize)
        axis[1].tick_params(labelleft=False,
                       labelright=False)
        axis[0].set_ylabel(r'$V$ (km s$^{-1}$)', fontsize=labelsize)
    elif style == 'onepanel':
        axis[0].set_xlabel(r'$r$ (pc)', fontsize=labelsize)
        axis[0].set_ylabel(r'$V$ (km s$^{-1}$)', fontsize=labelsize)
    else:
        axis[0].tick_params(labelbottom=False)   
        axis[0].tick_params(labelleft=True,
                       labelright=False)
        axis[0].set_ylabel(r'$V$ (km s$^{-1}$)', fontsize=labelsize)
        axis[1].set_ylabel(r'$V$ (km s$^{-1}$)', fontsize=labelsize)
    if style != 'onepanel':
        axis[1].set_xlabel(r'$r$ (pc)', fontsize=labelsize)
    fig.savefig(f'{fig_path}{fig_name}Vel_radprofile_1x2_'+style+'.pdf', dpi=300)
    plt.close()
    
def plot_pvdiagram(NGC253_path, source, fig_path, moments_path, molecule = 'HC3Nvib_J24J26', D_Mpc = 3.5, style = 'onecol', fig_name = ''):
    """
        Plots the Position-Velocity diagram for the cloud-cloud collision
        style = 'onecol' one column two rows
        style = 'twocol' two cols one row
    """
    labelsize = 20
    cbar_labelfont = 20
    ticklabelsize = 16
    fontsize = 20
    cbar_tickfont = 14

    slimpath = f'{NGC253_path}SHC/{source}/SLIM/'
    cube = f'{slimpath}Vel_{molecule}_{source}.fits'
    cont_cube = f'{NGC253_path}Continuums/{source}/MAD_CUB_219GHz_continuum.I.image.pbcor_{source}_pycut_coord.fits'

    cubo = utiles_cubes.Cube_Handler('cubo', cube)
    cubocont = utiles_cubes.Cube_Handler('cubocont', cont_cube)
    rms_219 = 1.307E-5#8.629E-6
    cubo_219_aboverms_mask = utiles_cubes.sigma_mask(cubocont.fitsdata[0,0,:,:], rms_219, 2.0)
    cubo_219_aboverms = np.copy(cubocont.fitsdata[0,0,:,:])
    cubo_219_aboverms[cubo_219_aboverms_mask.mask] = np.nan
    
    # Velocity min and max
    if source == 'SHC_13':
        vel_min = 235
        vel_max = 265
    else:
        utiles.round_to_multiple(np.nanmin(cube.fitsdata[0,0,:,:]), 5)
        utiles.round_to_multiple(np.nanmax(cube.fitsdata[0,0,:,:]), 5)
    vel_ticks = list(np.linspace(vel_min, vel_max, 5))
    
    # Starting figure
    figsize = 10
    naxis = 1
    maxis = 1
    cbar_pad = -65
    axcolor = 'k'
    wcs_plot = cubo.wcs
    fig = plt.figure(figsize=(naxis*figsize*0.7, figsize))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.0, hspace=0.28, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    
    RA_start, Dec_start = utiles_cubes.RADec_position(47, 70, wcs_plot, origin = 1)
    RA_end, Dec_end = utiles_cubes.RADec_position(91, 118, wcs_plot, origin = 1)    
    # Adding wcs frame
    wcs_plot.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    axes = []
    for i in range(naxis*maxis):
        axes.append(fig.add_subplot(gs1[i], aspect='equal', projection=wcs_plot))
        axes[i].tick_params(labelsize=ticklabelsize)
        axes[i].tick_params(direction='in')
        axes[i].coords.frame.set_color(axcolor)
        axes[i].coords[0].set_major_formatter('hh:mm:ss.sss')
        axes[i].coords[1].set_ticks(size=14, width=2, color=axcolor, exclude_overlapping=True, number = 5)
        axes[i].coords[0].set_ticks(size=14, width=2, color=axcolor, exclude_overlapping=True, number = 5)
        axes[i].coords[0].frame.set_linewidth(2)
        axes[i].coords[1].frame.set_linewidth(2)
        axes[i].coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        axes[i].coords[0].display_minor_ticks(True)
        axes[i].coords[1].display_minor_ticks(True)
    
    axes[0].imshow(cubo.fitsdata[0,0,:,:], transform=axes[0].get_transform(cubo.wcs), vmin=vel_min, vmax=vel_max, cmap =plt.cm.RdBu, aspect='equal')
    plot_utiles.add_cbar(fig, axes[0], cubo.fitsdata[0,0,:,:], r'VLSR (km s$^{-1}$)', color_palette='RdBu', colors_len = 0,
                 orientation='h_short', sep=0.075, width=0.02, height=False,ticks=vel_ticks,
                 Mappable=False, cbar_limits=[vel_min, vel_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                 discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                 )
    
    levels219 = [5*rms_219, 10*rms_219, 50*rms_219, 100*rms_219]
    axes[0].contour(cubo_219_aboverms, colors='r', linewidths=1.4, zorder=2, levels= levels219,
        transform=axes[0].get_transform(cubocont.wcs))
    
    xcenter = 48
    ycenter = 20
    axes[0].plot(xcenter,ycenter, marker='x', color='k')
    axes[0].tick_params(axis="both", which='minor', length=8)
    axes[0].xaxis.set_tick_params(top =True, labeltop=False)
    axes[0].yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
    axes[0].set_xlabel('RA (J2000)', fontsize = labelsize)
    axes[0].tick_params(direction='in')
    axes[0].tick_params(axis="both", which='minor', length=8)
    axes[0].set_ylabel('Dec (J2000)', fontsize = labelsize, labelpad=-1)
    
    
    ymin = 10
    ymax = 30
    xmin = 41
    xmax = 56
    if source == 'SHC_13':
        axes[0].set_xlim([xmin, xmax])
        axes[0].set_ylim([ymin, ymax])
        
    pixsize_arcsec =  cubo.pylen*3600
    pixsize_pc = pixsize_arcsec*D_Mpc*1e6/206265.0
    ymin = 8
    ymax = 32
    xmin = 39
    xmax = 58
    m02625v0 = f'NGC253_{source}_HC3N_v0_26_25_M0_aboveM0.fits'
    cubo_m02625v0 = utiles_cubes.Cube_Handler('219', moments_path+m02625v0)
    cubo_m02625v0_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02625v0.fitsdata[0,0,:,:], cubo_m02625v0.header['INTSIGMA'], 3)
    cubo_m02625v0_aboverms = np.copy(cubo_m02625v0.fitsdata[0,0,:,:])
    cubo_m02625v0_aboverms[cubo_m02625v0_aboverms_mask.mask] = np.nan
    cubo_m02625v0_aboverms = cubo_m02625v0_aboverms[ymin:ymax,xmin:xmax]
    m02625v7 = f'NGC253_{source}_HC3N_v7=1_26_25_+1_-1_M0_aboveM0.fits'
    cubo_m02625v7 = utiles_cubes.Cube_Handler('219', moments_path+m02625v7)
    cubo_m02625v7_aboverms_mask = utiles_cubes.sigma_mask(cubo_m02625v7.fitsdata[0,0,:,:], cubo_m02625v7.header['INTSIGMA'], 3)
    cubo_m02625v7_aboverms = np.copy(cubo_m02625v7.fitsdata[0,0,:,:])
    cubo_m02625v7_aboverms[cubo_m02625v7_aboverms_mask.mask] = np.nan
    cubo_m02625v7_aboverms = cubo_m02625v7_aboverms[ymin:ymax,xmin:xmax]
    path_v6 = f'{slimpath}/{source}_velprofiles/MAD_CUB_II_NGC253_v6_1_2625_250_270kms.fits'
    cubo_v6 = utiles_cubes.Cube_Handler('v6', path_v6)
    int_vel_range = (265-235)
    rms = 2e-4
    sigma = utiles.integratedsigma(1, int_vel_range, rms, 1.23)
    cubo_v6_aboverms_mask = utiles_cubes.sigma_mask(cubo_v6.fitsdata[0,0,:,:], sigma, 3)
    cubo_v6_aboverms = np.copy(cubo_v6.fitsdata[0,0,:,:])
    cubo_v6_aboverms[cubo_v6_aboverms_mask.mask] = np.nan
    path_v6v7 = f'{slimpath}{source}_velprofiles/MAD_CUB_II_v6v7_1_2625_230_265_kms.fits'
    cubo_v6v7 = utiles_cubes.Cube_Handler('v6', path_v6v7)
    int_vel_range = (265-230)
    rms = 200e-3
    sigma = utiles.integratedsigma(1, int_vel_range, rms, 1.23)
    cubo_v6v7_aboverms_mask = utiles_cubes.sigma_mask(cubo_v6v7.fitsdata[0,0,:,:], sigma, 3)
    cubo_v6v7_aboverms = np.copy(cubo_v6v7.fitsdata[0,0,:,:])
    cubo_v6v7_aboverms[cubo_v6v7_aboverms_mask.mask] = np.nan
    
    
    col_path = f'{slimpath}ColumnDensity_HC3Nvib_J24J26_{source}.fits'
    col_cube = utiles_cubes.Cube_Handler('col', col_path)
    tex_path = f'{slimpath}Tex_HC3Nvib_J24J26_{source}.fits'
    tex_cube = utiles_cubes.Cube_Handler('tex', tex_path)
    fwhm_path = f'{slimpath}FWHM_HC3Nvib_J24J26_{source}.fits'
    fwhm_cube = utiles_cubes.Cube_Handler('fwhm', fwhm_path)

    cubo_rest = cubo.fitsdata[0,0,:,:]
    cubo_rest = cubo_rest[ymin:ymax,xmin:xmax]
    col_cubo = col_cube.fitsdata[0,0,:,:]
    col_cubo = col_cubo[ymin:ymax,xmin:xmax]
    tex_cubo = tex_cube.fitsdata[0,0,:,:]
    tex_cubo = tex_cubo[ymin:ymax,xmin:xmax]
    fwhm_cubo = fwhm_cube.fitsdata[0,0,:,:]
    fwhm_cubo = fwhm_cubo[ymin:ymax,xmin:xmax]
    v6_cubo = cubo_v6_aboverms
    v6_cubo = v6_cubo[ymin:ymax,xmin:xmax]
    v6v7_cubo = cubo_v6v7_aboverms
    v6v7_cubo = v6v7_cubo[ymin:ymax,xmin:xmax]
    rotate = False
    if rotate:
        col_cubo = ndimage.rotate(col_cubo, 130+90, reshape=False, cval=np.nan)
        cubo_rest = ndimage.rotate(cubo_rest, 130+90, reshape=False, cval=np.nan)
        tex_cubo = ndimage.rotate(tex_cubo, 130+90, reshape=False, cval=np.nan)
        v6_cubo = ndimage.rotate(v6_cubo, 130+90, reshape=False, cval=np.nan)
        v6v7_cubo = ndimage.rotate(v6_cubo, 130+90, reshape=False, cval=np.nan)
    
    pv_diag = np.zeros(shape=(cubo_rest.shape[0],cubo_rest.shape[1]))
    distances = []
    velocities = []
    hc3n_intes = []
    v6_intens = []
    v6v7_intens = []
    columns = []
    texs = []
    fwhms = []
    new_ycenter = 10+2 #ycenter - ymin
    new_xcenter = 7+2  #xcenter - xmin
    
    plt.imshow(cubo_rest, origin='lower')
    plt.plot(new_xcenter, new_ycenter, marker='x',color='k')
    fig.savefig(f'{fig_path}pvtest_unknwn.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()

    for j in range(0, cubo_rest.shape[0]):
        col_dist = j-new_ycenter
        col_dist_pc = col_dist*pixsize_pc
        sorted_array = cubo_rest[j,:]#[j,:]
        distances = distances+[col_dist_pc]*cubo_rest.shape[1]
        velo = cubo_rest[j,:].tolist()
        velo_sorted = np.sort(velo)
        velocities = velocities + velo_sorted.tolist()
        hintes = cubo_m02625v7_aboverms[j,:].tolist()
        hintes_sorted = [x for _,x in sorted(zip(velo,hintes))]
        hc3n_intes = hc3n_intes + hintes_sorted
        v6_int = v6_cubo[j,:].tolist()
        v6_sort = [x for _,x in sorted(zip(velo,v6_int))]
        v6_intens = v6_intens + v6_sort
        v6v7_int = v6v7_cubo[j,:].tolist()
        v6v7_sort = [x for _,x in sorted(zip(velo,v6v7_int))]
        v6v7_intens = v6v7_intens + v6v7_sort
        cols = col_cubo[j,:].tolist()
        cols_sort = [x for _,x in sorted(zip(velo,cols))]
        columns = columns + cols_sort
        txs = tex_cubo[j,:].tolist()
        txs_sort = [x for _,x in sorted(zip(velo,txs))]
        texs = texs + txs_sort
        fwhm = fwhm_cubo[j,:].tolist()
        fwhms_sort = [x for _,x in sorted(zip(velo,fwhm))]
        fwhms = fwhms + fwhms_sort
        
    x_data = np.array(distances)
    y_data = np.array(velocities)
    z_data = np.array(hc3n_intes)
    z_coldata = np.array(columns)
    z_texdata = np.array(texs)
    z_fwhmdata = np.array(fwhms)
    z_v6data = np.array(v6_intens)
    z_v6v7data = np.array(v6v7_intens)
    Ztex = z_texdata.reshape(tex_cubo.shape)
    Zintens = z_data.reshape(tex_cubo.shape)
    Zcols = z_coldata.reshape(tex_cubo.shape)
    Zv6 = z_v6data.reshape(tex_cubo.shape)
    Zv6v7 = z_v6v7data.reshape(tex_cubo.shape)
    Zfwhm = z_fwhmdata.reshape(tex_cubo.shape)

    figsize = 8
    if style=='twocol':
        naxis = 2
        maxis = 1
        fig = plt.figure(figsize=(naxis*figsize*1.25, figsize))
    elif style=='onecol':
        naxis = 1
        maxis = 2
        fig = plt.figure(figsize=(figsize, figsize*1.55))
    
    cbar_pad = -65
    axcolor = 'k'
    wcs_plot = cubo.wcs
    gs1 = gridspec.GridSpec(maxis, naxis)    
    if style=='twocol':
        gs1.update(wspace = 0.08, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    else:
        gs1.update(wspace = 0.125, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    
    # Adding wcs frame
    wcs_plot.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    axes = []
    for i in range(naxis*maxis):
        axes.append(fig.add_subplot(gs1[i]))
    fig_cbar = []
    logn_min = 14.8#utiles.round_to_multiple(np.nanmin(Zcols), 0.5) #np.round(utiles.truncate(np.nanmin(columndens_cube.fitsdata[0,0,:,:]), 1),1)
    logn_max = 16.5#utiles.round_to_multiple(np.nanmax(Zcols), 0.5)
    fcbar1 = axes[0].imshow(Zcols, extent=(241.7,263.4, np.amin(x_data), np.amax(x_data)), aspect = 'auto', norm=LogNorm(vmin=logn_min, vmax=logn_max), cmap =plt.cm.rainbow)
    tex_min = 150 # np.nanmin(Zv6v7)# 150
    tex_max = 900 # np.nanmax(Zv6v7)# 900
    tex_ticks = list(np.linspace(tex_min, tex_max, 5)) #np.arange(tex_min, tex_max, 10, 5)
    fcbar2 = axes[1].imshow(Ztex, extent=(241.7,263.4, np.amin(x_data), np.amax(x_data)), aspect = 'auto', cmap =plt.cm.rainbow, vmin=tex_min, vmax=tex_max)
    fig_cbar.append(fcbar1)
    fig_cbar.append(fcbar2)
    label_cbar = [r'log N(HC$_3$N) (cm$^{-2}$)', r'T$_{\text{vib}}$ (K)']
    cbar_pad = -55
    for l, ax in enumerate(axes):
        minor_locator = AutoMinorLocator(5)
        minor_locatory = AutoMinorLocator(5)
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=10)
        axes[l].tick_params(axis="both", which='minor', length=5)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticklabelsize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].yaxis.set_minor_locator(minor_locatory)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        for axis in ['top', 'bottom', 'left', 'right']:
            axes[l].spines[axis].set_linewidth(1.5)  # change width
        if l ==1:
            if style == 'twocol':
                axes[l].tick_params(
                       labelleft=False)
            axes[l].text(0.05, 0.95, r'T$_{\text{vib}}$',
                                color = 'white',
                                horizontalalignment='left',
                                verticalalignment='top',
                                weight='bold',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
            if style == 'twocol':
                plot_utiles.add_cbar(fig, axes[l], Ztex, r'T$_\text{vib}$ (K)', color_palette='rainbow', colors_len = 0,
                                     orientation='h_short', sep=0.03, width=0.02, height=False, ticks = tex_ticks,
                                     Mappable=False, cbar_limits=[tex_min, tex_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                                     discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                                     )
            else:
                plot_utiles.add_cbar(fig, axes[l], Ztex, r'T$_\text{vib}$ (K)', color_palette='rainbow', colors_len = 0,
                                     orientation='v', sep=0.03, width=0.02, height=False, ticks = tex_ticks,
                                     Mappable=False, cbar_limits=[tex_min, tex_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                                     discrete_colorbar=False, formatter = '%1.0f', norm='lin', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                                     )
        else:
            axes[l].text(0.05, 0.95, r'log N(HC$_3$N)',
                                color = 'white',
                                weight='bold',
                                horizontalalignment='left',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
            if style == 'twocol':
                plot_utiles.add_cbar(fig, axes[l], Zcols, r'logN(HC$_3$N) (cm$^{-2}$)', color_palette='rainbow', colors_len = 0,
                                     orientation='h_short', sep=0.03, width=0.02, height=False,
                                     ticks=[15, 15.5, 16, 16.5], Mappable=False, cbar_limits=[logn_min, logn_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                                     discrete_colorbar=False, formatter = '%1.1f', norm='log', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                                     )
            else:
                plot_utiles.add_cbar(fig, axes[l], Zcols, r'logN(HC$_3$N) (cm$^{-2}$)', color_palette='rainbow', colors_len = 0,
                                     orientation='v', sep=0.03, width=0.02, height=False,
                                     ticks=[15, 15.5, 16, 16.5], Mappable=False, cbar_limits=[logn_min, logn_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
                                     discrete_colorbar=False, formatter = '%1.1f', norm='log', labelpad = cbar_pad, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
                                     )
    if style == 'onecol':
        axes[0].tick_params(labelbottom=False)
        axes[0].set_ylabel(r'Dec offset (pc)', fontsize = labelsize)
        axes[1].set_ylabel(r'Dec offset (pc)', fontsize = labelsize)
        axes[1].set_xlabel(r'V (km s$^{-1}$)', fontsize = labelsize)
    else:
        axes[0].set_ylabel(r'Dec offset (pc)', fontsize = labelsize)
        axes[0].set_xlabel(r'V (km s$^{-1}$)', fontsize = labelsize)
        axes[1].set_xlabel(r'V (km s$^{-1}$)', fontsize = labelsize)
    fig.savefig(f'{fig_path}{fig_name}pvtest_texandcol_1x2_v2.pdf', bbox_inches='tight', transparent=True, dpi=400)

