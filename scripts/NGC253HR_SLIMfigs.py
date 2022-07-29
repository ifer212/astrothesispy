from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_cubes
from astrothesispy.utiles import utiles_plot as plot_utiles

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import pandas as pd
import numpy as np

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rc('xtick', color='k', direction='in', labelsize=6)
plt.rc('ytick', color='k', direction='in', labelsize=6)

# =============================================================================
# SLIM figures
# =============================================================================

def plot_SLIM2D(NGC253_path,  cont_path, location_path, fig_path, molecule = 'HC3Nvib_J24J26', source = 'SHC_13', D_Mpc = 3.5):
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
                 ticks=[14.5, 15, 15.5, 16, 16.5], Mappable=False, cbar_limits=[logn_min, logn_max], tick_font = cbar_tickfont, label_font = cbar_labelfont,
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
    fig.savefig(fig_path+source+'_SLIM_cubes_'+molecule+'_no_profile_noM0_v2.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()


def plot_SLIMprofiles(NGC253_path, fig_path, molecule = 'HC3Nvib_J24J26', source = 'SHC_13', D_Mpc = 3.5, style = 'onecol'):
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
    fig.savefig(f'{fig_path}{source}_SLIM_Tex_and_logN_profiles_dfcolors_newcols_1x2_v2.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
