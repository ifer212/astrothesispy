from astrothesispy.utiles import utiles
from astrothesispy.utiles import u_conversion

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import rotate

# Nice seaborn colors
redpink     = sns.xkcd_palette(['red pink'])[0]
oblue       = sns.xkcd_palette(['ocean blue'])[0]
elime       = sns.xkcd_palette(['electric lime'])
melon       = sns.xkcd_palette(['melon'])[0]
aquamarine  = sns.xkcd_palette(['aquamarine'])[0]
aquagreen   = sns.xkcd_palette(['aqua green'])[0]
turqoise    = sns.xkcd_palette(['turquoise'])[0]
water       = sns.xkcd_palette(['water blue'])[0]
brown       = sns.xkcd_palette(['terracota'])[0]
purple      = sns.xkcd_palette(['purple'])[0]
orange      = sns.xkcd_palette(['blood orange'])[0]
green       = sns.xkcd_palette(['kelly green'])[0]
azure       = sns.xkcd_palette(['azure'])[0]
violet      = sns.xkcd_palette(['violet'])[0]
bred        = sns.xkcd_palette(['blood red'])[0]
yellow      = sns.xkcd_palette(['orange yellow'])[0]
dviolet     = sns.xkcd_palette(['light magenta'])[0]
dazure      = sns.xkcd_palette(['bright blue'])[0]
dgreen      = sns.xkcd_palette(['darkish green'])[0]

# Loading amsmath LaTeX
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# Matplotlib default parameters
mpl.rc('xtick', color='k', direction='in', labelsize=6)
mpl.rc('ytick', color='k', direction='in', labelsize=6)

class InputError(LookupError):
    '''Raise this when there's an error in the input parameter'''

def rotate_cube(data_cube, pa):
    rot = rotate(data_cube, pa-90., reshape=False)
    return rot

def rotate_cubev2(data_cube, pa):
    rot = rotate(data_cube, pa, reshape=False)
    return rot
    
def add_cbar(fig, ax, data, label, color_palette='Spectral', colors_len = 0,
             orientation='v', sep=0.02, width=0.02, height=False,
             ticks=False, Mappable=False, cbar_limits=[], tick_font = 8, label_font = 10,
             discrete_colorbar=True, formatter = '', norm='log', labelpad = -30, custom_cmap=False, ticksize=6, framewidth=2, tickwidth=1
             ):
    """
    Adds a  colorbar next to specified ax
    if ax is a list:
        vertical    [axis bottom  to top axis]
        horizontyal [axis left to right  axis]
    """
    if isinstance(ax, (list,)):
        # If more than one axis we have to reorder from min to max positions 
        if orientation == 'v':
            ypos  = [ax[0].get_position().y0, ax[-1].get_position().y0]
            ysort = [ax[ypos.index(min(ypos))], ax[ypos.index(max(ypos))]]
            pos = utiles.data(x0 = ysort[0].get_position().x0, y0 = ysort[0].get_position().y0,
                              width = ysort[0].get_position().width,
                              height= ysort[1].get_position().y0+ysort[1].get_position().height-ysort[0].get_position().y0)
        elif orientation == 'h':
            xpos  = [ax[0].get_position().x0, ax[-1].get_position().x0]
            xsort = [ax[xpos.index(min(xpos))], ax[xpos.index(max(xpos))]]
            pos = utiles.data(x0 = xsort[0].get_position().x0, y0 = xsort[0].get_position().y0,
                              width = xsort[1].get_position().x0 + xsort[1].get_position().width-xsort[0].get_position().x0,
                              height= xsort[0].get_position().height)
        elif orientation == 'h_short':
            xpos  = [ax[0].get_position().x0, ax[-1].get_position().x0]
            xsort = [ax[xpos.index(min(xpos))], ax[xpos.index(max(xpos))]]
            pos = utiles.data(x0 = xsort[0].get_position().x0, y0 = xsort[0].get_position().y0,
                              width = (xsort[1].get_position().x0 + xsort[1].get_position().width-(xsort[0].get_position().x0)),
                              height= xsort[0].get_position().height)
    else:
        pos = ax.get_position()
    if orientation == 'v':
        # Vertical colorbar
        orien = 'vertical'
        axis = 'y'
        if height is False:
            cb_axl = [pos.x0 + pos.width + sep, pos.y0,  width, pos.height] 
        else:
            cb_axl = [pos.x0 + pos.width + sep, pos.y0,  width, height]
        labelpad = None
    elif orientation == 'h':
        # Horizontal colorbar
        axis = 'x'
        orien = 'horizontal'
        if height is False:
            cb_axl = [pos.x0, pos.y0 + pos.height + sep,  pos.width, width] 
        else:
            cb_axl = [pos.x0, pos.y0 + pos.height + sep,   10, width]
        labelpad = labelpad
    elif orientation == 'h_short':
        # Horizontal colorbar
        axis = 'x'
        orien = 'horizontal'
        if height is False:
            cb_axl = [pos.x0 + 0.01, pos.y0 + pos.height + sep,  np.round(pos.width,2)-2*0.01, width] 
        else:
            cb_axl = [pos.x0 + 0.01, pos.y0 + pos.height + sep,   10-2*0.01, width]
        labelpad = labelpad
    cb_ax = fig.add_axes(cb_axl)
    
    if Mappable == True:
        if ticks is False:
            cbar = fig.colorbar(data, orientation=orien, cax=cb_ax)
        else:
            cbar = fig.colorbar(data, orientation=orien, cax=cb_ax, ticks=ticks)
    else:
        # Building colors from colorbar
        if len(cbar_limits)>=1:
            cmin = cbar_limits[0]
            cmax = cbar_limits[1]
        else:
            cmin = np.min(data)
            cmax = np.max(data)
            
        if colors_len == 0:
            lencolors = 0
        else:
            lencolors = colors_len
        if discrete_colorbar == True:
            cmap =  matplotlib.colors.ListedColormap(sns.color_palette(color_palette, lencolors))
        elif custom_cmap:
            cmap = color_palette
        else:
            cmap = plt.cm.get_cmap(color_palette)
        if norm == 'log':
            normalize = matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax) 
        else:
            normalize = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
        cbar_colors = [cmap(normalize(value)) for value in data]
        from matplotlib.ticker import FormatStrFormatter
        if formatter == '':
            sformatter = FormatStrFormatter('%1.3f')
        else:
            sformatter = FormatStrFormatter(formatter)
        
        cbar = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=normalize, orientation=orien, format=sformatter)
        cbar.ax.minorticks_off()
        if isinstance(ticks, (list,)):
            cbar.set_ticks(ticks)
    cbar.outline.set_linewidth(framewidth)
    cbar.ax.minorticks_off()
    cbar.set_label(label, labelpad=labelpad, fontsize=label_font)
    cbar.ax.tick_params(axis=axis, direction='in')
    cbar.ax.tick_params(labelsize=tick_font)
    cbar.ax.tick_params(length=ticksize, width=tickwidth)
    return cbar, cbar_colors

def map_figure_starter(wcs, maxis, naxis, fsize, labelsize, fontsize=12, axcolor='k', xlim_ra=False, ylim_dec=False, ticksize=6, hspace=0.08, wspace = -0.08):
    """
    Figure with RA and DEc starter
    """
    fig = plt.figure(figsize=(maxis*fsize*0.8, fsize*0.9))
    gs1 = gridspec.GridSpec(nrows=maxis, ncols=naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = wspace, hspace=hspace, top=0.95, bottom = 0.05, left=0.05, right=0.95)
    # Adding wcs frame
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    axes = []
    for i in range(naxis*maxis):
        axes.append(fig.add_subplot(gs1[i], aspect='equal', projection=wcs))
        axes[i].tick_params(labelsize=labelsize)
        axes[i].tick_params(direction='in')
        axes[i].coords.frame.set_color(axcolor)
        axes[i].xaxis.set_tick_params(top =True, labeltop=False)
        axes[i].yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
        axes[i].coords[0].set_major_formatter('hh:mm:ss.ss')
        axes[i].coords[0].set_ticks(size=ticksize, width=1, color=axcolor, number = 5)
        axes[i].coords[1].set_ticks(size=ticksize, width=1, color=axcolor, number = 5)
        axes[i].coords[0].set_ticklabel(exclude_overlapping=True)
        axes[i].coords[1].set_ticklabel(exclude_overlapping=True)
        axes[i].coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        axes[i].coords[1].set_separator((r'$^{\circ}$', r'$^{\prime}$', r'$^{\prime \prime}$'))
        axes[i].coords[0].display_minor_ticks(True)
        axes[i].coords[1].display_minor_ticks(True)
        axes[i].set_xlabel('RA (J2000)', fontsize = fontsize)
        axes[i].set_ylabel('Dec (J2000)', fontsize = fontsize, labelpad=-1)
        axes[i].tick_params(direction='in')
    return fig, axes

def load_map_axes(axis, ticksize, ticklabelsize, labelsize, labelpad = -1, axcolor='k', ticknumber = 5,
                  tickwidth = 1, axiswidth = 1, add_labels = True):
    """
        Load WCS coordinates proper axis format
    """
    axis.tick_params(labelsize=ticklabelsize)
    axis.tick_params(direction='in')
    axis.coords.frame.set_color(axcolor)
    axis.xaxis.set_tick_params(top =True, labeltop=False)
    axis.yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
    axis.coords[0].set_major_formatter('hh:mm:ss.ss')
    axis.coords[0].set_ticks(size=ticksize, width=tickwidth, color=axcolor, number = ticknumber)
    axis.coords[1].set_ticks(size=ticksize, width=tickwidth, color=axcolor, number = ticknumber)
    axis.coords[0].set_ticklabel(exclude_overlapping=True)
    axis.coords[1].set_ticklabel(exclude_overlapping=True)
    axis.coords[0].frame.set_linewidth(axiswidth)
    axis.coords[1].frame.set_linewidth(axiswidth)
    axis.coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
    axis.coords[1].set_separator((r'$^{\circ}$', r'$^{\prime}$', r'$^{\prime \prime}$'))
    axis.coords[0].display_minor_ticks(True)
    axis.coords[1].display_minor_ticks(True)
    axis.tick_params(direction='in')
    if add_labels:
        axis.set_xlabel('RA (J2000)', fontsize = labelsize)
        axis.set_ylabel('Dec (J2000)', fontsize = labelsize, labelpad=labelpad)
    
def map_figure_starter_velcomp(wcs, maxis, naxis, fsize, labelsize, fontsize=12, axcolor='k', xlim_ra=False, ylim_dec=False, ticksize=6, hspace=0.08, wspace = -0.08,
                               xrat=1, yrat=1):
    """
    Figure with RA and DEc starter
    """
    fig = plt.figure(figsize=(xrat*fsize, yrat*fsize))
    gs1 = gridspec.GridSpec(nrows=maxis, ncols=naxis)   
    gs1.update(wspace = wspace, hspace=hspace, top=0.95, bottom = 0.05, left=0.05, right=0.95)
    # Adding wcs frame
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    axes = []
    for i in range(naxis*maxis):
        axes.append(fig.add_subplot(gs1[i], aspect='equal', projection=wcs))
        axes[i].tick_params(labelsize=labelsize)
        axes[i].tick_params(direction='in')
        axes[i].coords.frame.set_color(axcolor)
        #axes[i].set_yticklabels([])
        axes[i].xaxis.set_tick_params(top =True, labeltop=False)
        axes[i].yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
        axes[i].coords[0].set_major_formatter('hh:mm:ss.ss')
        axes[i].coords[0].set_ticks(size=ticksize, width=1, color=axcolor, number = 5)
        axes[i].coords[1].set_ticks(size=ticksize, width=1, color=axcolor, number = 5)
        axes[i].coords[0].set_ticklabel(exclude_overlapping=True)
        axes[i].coords[1].set_ticklabel(exclude_overlapping=True)
        axes[i].coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        axes[i].coords[1].set_separator((r'$^{\circ}$', r'$^{\prime}$', r'$^{\prime \prime}$'))
        axes[i].coords[0].display_minor_ticks(True)
        axes[i].coords[1].display_minor_ticks(True)
        axes[i].set_xlabel('RA (J2000)', fontsize = fontsize)
        axes[i].set_ylabel('Dec (J2000)', fontsize = fontsize, labelpad=-1)
        axes[i].tick_params(direction='in')
        
    return fig, axes
        
def truncate_cmap (cmap,n_min=0,n_max=256):
    """ Generate a truncated colormap 
    Colormaps have a certain number of colors (usually 255) which is 
    available via cmap.N
    This is a simple function which returns a new color map from a 
    subset of the input colormap. For example, if you only want the jet 
    colormap up to green colors you may use 
    tcmap = truncate_cmap(plt.cm.jet,n_max=150)
    This function is especially useful when you want to remove the white
    bounds of a colormap 
    Parameters 
    cmap : plt.matplotlib.colors.Colormap 
    n_min : int 
    n_max : int 
    Return 
    truncated_cmap : plt.matplotlib.colors.Colormap
        
    """
    # From https://gist.github.com/astrodsg/09bfac1b68748967ed8b#file-mpl_colormap_tools
    color_index = np.arange(n_min,n_max).astype(int)
    colors = cmap(color_index)
    name = "truncated_{}".format(cmap.name)
    return plt.matplotlib.colors.ListedColormap(colors,name=name)
    
def plot_cube_moments(cube, save_path, outname, cbar0=False, cbar1=False, sigma_mask=False, plot_beam=False):
    sigma_mask = np.array(sigma_mask)
    if sigma_mask.any():
        fig_len = 5
    else:
        fig_len = 4
    fig = plt.figure(figsize=(fig_len*4, 4))
    gs1 = gridspec.GridSpec(1, fig_len)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    # Adding wcs frame
    wcs_1 = WCS(cube.header, naxis=2)
    wcs_1.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    ax1 = fig.add_subplot(gs1[0], aspect='equal', projection=wcs_1)
    ax2 = fig.add_subplot(gs1[1], aspect='equal', projection=wcs_1)
    ax3 = fig.add_subplot(gs1[2], aspect='equal', projection=wcs_1)
    ax4 = fig.add_subplot(gs1[3], aspect='equal', projection=wcs_1)
    cube.M1[~cube.mask] = np.nan
    cube.M2[~cube.mask] = np.nan
    cube.above_rmsM0[~cube.mask] = np.nan
    
    mom0 = ax1.imshow(cube.M0, origin='lower', vmax=cube.M0.max(), vmin=cube.M0.min(), cmap=cm.gnuplot2, interpolation="none",zorder=1)#, norm=LogNorm(vmin=0.2*cont_stdev, vmax=cont_max/3.))# vmax=cm_max, vmin=cm_std*2, interpolation="none")
    ax1.contour(cube.M0, colors=aquagreen, levels = [3*cube.integstd, 5*cube.integstd, 10*cube.integstd], linewidths=0.7, transform=ax1.get_transform(wcs_1), zorder=2)
    
    if sigma_mask.any():
        ax5 = fig.add_subplot(gs1[1], aspect='equal', projection=wcs_1)
        # Getting pixels inside sigma_level*self.integstd
        cs2 = ax1.contour(cube.M0, levels = [3*cube.integstd], linewidths=0.0, transform=ax1.get_transform(wcs_1), colors=None, zorder=0)
        path = cs2.collections[0].get_paths()[0]
        x, y = np.meshgrid(np.arange(0, cube.M0.shape[0], 1), np.arange(0, cube.M0.shape[1], 1))
        points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
        mask_inside = path.contains_points(points)
        mask_inside.shape = x.shape
        above_sigma_values = np.ma.masked_where(~mask_inside, cube.M0)
        if sigma_mask.all == True:
            ax5.imshow(above_sigma_values, zorder=3, origin='lower', transform=ax5.get_transform(wcs_1), cmap=cm.Spectral)
        else:
            ax5.imshow(sigma_mask, zorder=3, origin='lower', transform=ax5.get_transform(wcs_1), cmap=cm.Spectral)
    plot_m1 = cube.M1+cube.velref-np.nanmedian(cube.M1+cube.velref)
    mom1 = ax2.imshow(plot_m1, origin='lower', vmax=np.nanmax(plot_m1), vmin=np.nanmin(plot_m1), cmap=cm.Spectral, interpolation="none")#, norm=LogNorm(vmin=0.2*cont_stdev, vmax=cont_max/3.))# vmax=cm_max, vmin=cm_std*2, interpolation="none")
    mom2 = ax3.imshow(cube.M2, origin='lower', vmax=np.nanmax(cube.M2), vmin=np.nanmin(cube.M2), cmap=cm.gnuplot2, interpolation="none")#, norm=LogNorm(vmin=0.2*cont_stdev, vmax=cont_max/3.))# vmax=cm_max, vmin=cm_std*2, interpolation="none")
    above_rms = ax4.imshow(cube.above_rmsM0, origin='lower', vmax=np.nanmax(cube.above_rmsM0), vmin=np.nanmin(cube.above_rmsM0), cmap=cm.gnuplot2, interpolation="none")#, norm=LogNorm(vmin=0.2*cont_stdev, vmax=cont_max/3.))# vmax=cm_max, vmin=cm_std*2, interpolation="none")
    
    # Adding colorbar for moment0
    if cbar0:
        if not np.isnan(cube.M0).all():
            add_cbar(fig, ax1, mom0, 'Jy km/s', orientation='h', sep=0.02, width=0.02, height=False, ticks=False)

    # Adding colorbar for moment1
    if cbar1:
        if not np.isnan(plot_m1).all():
            print(np.nanmin(plot_m1))
            print(np.nanmax(plot_m1))
            cbar_limits = np.arange(np.round(np.nanmin(plot_m1),0),np.round(np.nanmax(plot_m1),0)+cube.chanwidth,cube.chanwidth)
            add_cbar(fig, ax2, mom1, 'km/s', orientation='h', sep=0.02, width=0.02, height=False, ticks=False)
    
    # Axis parameters
    axis = [ax1, ax2, ax3,ax4]
    for ax in axis:
        ax.tick_params(labelsize=6)
        ax.tick_params(direction='in')
        ax.coords.frame.set_color('k')
        if ax == ax1:
            axcolor='w'
        else:
            axcolor='k'
            ax.set_yticklabels([])
        # RA
        ax.coords[0].set_major_formatter('hh:mm:ss.ss')
        ax.coords[0].set_ticks(size=6, width=1, color=axcolor, exclude_overlapping=True, number = 5)
        ax.coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        ax.coords[0].display_minor_ticks(True)
        ax.set_xlabel('RA (J2000)')
        # Dec
        if ax != ax1:
            ax.xaxis.set_tick_params(top =True, labeltop=False)
            ax.yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
            ax.set_ylabel('Dec (J2000)')
            ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_ticks(size=6, width=1, color=axcolor)
        ax.coords[1].display_minor_ticks(True)
    
    if plot_beam:
        # Plottng Cont Beam 
        pixscale = np.abs(cube.header['CDELT1'])#*3600 # arcsec
        Beam_plotter(px=10, py=10, bmin=cube.bmin, bmaj=cube.bmaj,
                        pixsize=pixscale, pa=cube.bpa, axis=ax4,
                        color='blue', wcs=wcs_1, rectangle=True)
        x, y = np.meshgrid(np.arange(0, cube.M0.shape[0], 1), np.arange(0, cube.M0.shape[1], 1))
    
    plt.savefig(save_path+'/'+outname+'.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)
    
def plot_continuum(cube, save_path, outname, rms, RA_lim=[0], Dec_lim=[0], cbar0=False, cbar1=False, sigma_mask=False, plot_beam=False):
    
    cont = cube.fitsdata[0,0,:,:]
    sigma_mask = np.array(sigma_mask)
    if sigma_mask.any():
        fig_len = 2
    else:
        fig_len = 1
    fig = plt.figure(figsize=(fig_len*4, 4))
    gs1 = gridspec.GridSpec(1, fig_len)
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    # Adding wcs frame
    wcs_1 = WCS(cube.header, naxis=2)
    wcs_1.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    ax1 = fig.add_subplot(gs1[0], aspect='equal', projection=wcs_1)
    cont_im = ax1.imshow(cont, origin='lower', vmax=cube.max, vmin=cube.min, cmap=cm.gnuplot2, interpolation="none", zorder=1)#, norm=LogNorm(vmin=0.2*cont_stdev, vmax=cont_max/3.))# vmax=cm_max, vmin=cm_std*2, interpolation="none")
    ax1.contour(cont, colors=aquagreen, levels = [3*rms, 5*rms, 10*rms], linewidths=0.7, transform=ax1.get_transform(wcs_1), zorder=2)
    
    if sigma_mask.any():
        ax5 = fig.add_subplot(gs1[1], aspect='equal', projection=wcs_1)
        masked_image = cont.copy()
        masked_image[~sigma_mask] = np.nan
        ax5.imshow(masked_image, zorder=3, origin='lower', transform=ax5.get_transform(wcs_1), cmap=cm.Spectral)
        axis = [ax1,ax5]
    else:
        axis = [ax1]
    # Adding colorbar
    if cbar0:
        if not np.isnan(cont).all():
            add_cbar(fig, ax1, cont, 'Jy', orientation='h_short', sep=0.02, width=0.02, height=False, ticks=False)
    # Axis parameters
    for ax in axis:
        ax.tick_params(labelsize=6)
        ax.tick_params(direction='in')
        ax.coords.frame.set_color('k')
        if ax == ax1:
            axcolor='w'
        else:
            axcolor='k'
            ax.set_yticklabels([])
        # RA
        ax.coords[0].set_major_formatter('hh:mm:ss.ss')
        ax.coords[0].set_ticks(size=6, width=1, color=axcolor, number = 5)
        ax.coords[0].set_ticklabel(exclude_overlapping=True)
        ax.coords[0].set_separator((r'$^{\rm{h}}$', r'$^{\rm{m}}$', r'$^{\rm{s}}$'))
        ax.coords[0].display_minor_ticks(True)
        ax.set_xlabel('RA (J2000)')
        # Dec
        if ax != ax1:
            ax.xaxis.set_tick_params(top =True, labeltop=False)
            ax.yaxis.set_tick_params(right=True, labelright=False, labelleft=False)
            ax.set_ylabel('Dec (J2000)')
            ax.coords[1].set_ticklabel_visible(False)
        ax.coords[1].set_ticks(size=6, width=1, color=axcolor)
        ax.coords[1].display_minor_ticks(True)
    # Setting figure limits
    if (len(RA_lim) > 1 and len(Dec_lim) <= 1) or (len(RA_lim) <= 1 and len(Dec_lim) > 1):
        raise InputError('Only limits for one axis given, not impemented yet')
    elif len(RA_lim) > 1 and len(Dec_lim) > 1:
        px_lim = []
        py_lim = []
        for i, lin in enumerate(RA_lim):
            px_lim.append(utiles.px_position(RA_lim[i], Dec_lim[i], cube.wcs)[0])
            py_lim.append(utiles.px_position(RA_lim[i], Dec_lim[i], cube.wcs)[1])
        for ax in axis:
            ax.set_xlim(px_lim)
            ax.set_ylim(py_lim[::-1])
        if plot_beam:
            # Plottng Cont Beam 
            pixscale = np.abs(cube.header['CDELT1'])#*3600 # arcsec
            if cube.header['CUNIT1']=='deg':
                pixscale = np.abs(cube.header['CDELT1'])*3600.
            ellipse = Beam_plotter(px=px_lim[0]+10, py=py_lim[::-1][0]+10, bmin=cube.bmin, bmaj=cube.bmaj,
                            pixsize=pixscale, pa=cube.bpa, axis=ax1,
                            color='blue', wcs=wcs_1, rectangle=True)
            x, y = np.meshgrid(np.arange(0, cont.shape[0], 1), np.arange(0, cont.shape[1], 1))
            points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
            mask_inside = ellipse.contains_points(points)
            mask_inside.shape = x.shape
    else:
        if plot_beam:
            # Plottng Cont Beam 
            pixscale = np.abs(cube.header['CDELT1'])#*3600 # arcsec
            if cube.header['CUNIT1']=='deg':
                pixscale = np.abs(cube.header['CDELT1'])*3600.
            ellipse = Beam_plotter(px=10, py=10, bmin=cube.bmin, bmaj=cube.bmaj,
                            pixsize=pixscale, pa=cube.bpa, axis=ax1,
                            color='blue', wcs=wcs_1, rectangle=True)
            x, y = np.meshgrid(np.arange(0, cont.shape[0], 1), np.arange(0, cont.shape[1], 1))
            points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
            mask_inside = ellipse.contains_points(points)
            mask_inside.shape = x.shape
    plt.savefig(save_path+'/'+outname+'.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)
    return mask_inside
    
def plot_pc_scale(D_Mpc, pcs, py, px, pixsize, axis, color, wcs, vertical=False, text_sep=1.05, fontsize = 6, lw=1, annotate=True, text_sep_percen=True):
    L = (pcs *u.pc).to(u.m)
    pc = u_conversion.ang_size(D_Mpc, L).value
    pc_in_px = pc/pixsize
    if vertical:
        axis.vlines(px, ymin=py-(pc_in_px/2.), ymax=py+(pc_in_px/2.), color=color, linestyle='-', lw=lw, transform=axis.get_transform(wcs))
        if annotate:
            axis.annotate(str(pcs)+' pc', xy=(px,py), xytext=(px*text_sep,py),
                      fontsize=fontsize, color=color,
                      horizontalalignment='center',
                      verticalalignment='center',)
    else:
        axis.hlines(py, xmin=px-(pc_in_px/2.), xmax=px+(pc_in_px/2.), color=color, linestyle='-', lw=lw, transform=axis.get_transform(wcs))
        if annotate:
            if text_sep_percen:
                axis.annotate(str(pcs)+' pc', xy=(px,py), xytext=(px,py*text_sep), weight='bold',
                          fontsize=fontsize, color=color,
                          horizontalalignment='center',
                          verticalalignment='center',)
            else:
                axis.annotate(str(pcs)+' pc', xy=(px,py), xytext=(px,py+text_sep), weight='bold',
                          fontsize=fontsize, color=color,
                          horizontalalignment='center',
                          verticalalignment='center',)
   
def draw_ellipse(px, py, bmin, bmaj, pixsize, pa, axis, wcs=False, facecolor='none', color='red', linewidth=0.4, alpha=1, plot=True, zorder=1, linestyle='-', gaussfit=False):
    """
    Plotting beam
    pixel size in arcsec / px
    position angle (deg)
    bmaj ellipse_major (arcsec)
    bmin ellipse_minor (arcsec)
    """
    from matplotlib.patches import Ellipse
    # Beam
    if pa < 0:
        angle = 90 - pa
    else:
        angle = 90 +pa
    if gaussfit:
        angle = pa-90
    if wcs!= False:
        ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                        angle=angle, edgecolor=color, facecolor=facecolor, linewidth=linewidth,
                        transform=axis.get_transform(wcs),zorder=zorder, linestyle=linestyle)
    else: 
        ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                        angle=angle, edgecolor=color, facecolor=facecolor, linewidth=linewidth,
                        zorder=zorder, linestyle=linestyle)
    Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                    angle=angle, edgecolor='none', facecolor='none', linewidth=None,)
    if plot:
        axis.add_patch(ellipse)
    return ellipse
    
def Beam_plotter(px, py, bmin, bmaj, pixsize, pa, axis, wcs=False, color='red',linewidth=0.4, rectangle=True, label=False):
    """
    Plotting beam
    pixel size in arcsec / px
    position angle (deg)
    bmaj ellipse_major (arcsec)
    bmin ellipse_minor (arcsec)
    """
    from matplotlib.patches import Ellipse
    from matplotlib.patches import Rectangle
    # Beam
    if pa < 0:
        angle = 90 - pa
    else:
        angle = 90 +pa
    if wcs!= False:
        ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                        angle=angle, edgecolor=color, facecolor=color, linewidth=0.1,
                        transform=axis.get_transform(wcs))
    else:
        ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                        angle=angle, edgecolor=color, facecolor=color, linewidth=0.1)
    ellipse2 = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                        angle=angle, edgecolor=None, facecolor=None, linewidth=None,)
    
    axis.add_patch(ellipse)
    if rectangle:        
        # Rectangle Border
        r_ax1 = bmaj*1.4
        r_ax2 = bmaj*1.4
        if wcs!= False:
            r = Rectangle(xy=(px-r_ax2/pixsize/2, py-r_ax2/pixsize/2), width=r_ax1/pixsize,
                          height=r_ax2/pixsize, edgecolor=color, facecolor='none', linewidth=linewidth,
                      transform=axis.get_transform(wcs))
        else:
            r = Rectangle(xy=(px-r_ax2/pixsize/2, py-r_ax2/pixsize/2), width=r_ax1/pixsize,
                          height=r_ax2/pixsize, edgecolor=color, facecolor='none', linewidth=linewidth)
        axis.add_patch(r)
    #if label:
        # Writting label
    return ellipse2

def subfig_with_1cbar_initializer(layout, cbar_data, cbar_label, color_palette='Spectral', colors_len = 0,
                                  cbar_limits = [], sharex= False, sharey= False, plot_cbar = True,
                                  cbar_separation = 0.02, ticks=False,
                                  hspace=0.0, wspace=0.0, cbar_orientation='v', fisize=[4,4], discrete_colorbar=True,
                                  height_ratios = [1], tick_font = 8, label_font = 10, formatter = '', norm='log', labelpad=-30):
    
    if height_ratios == [1]:
        fig_len = layout[0]/layout[1]
        if fig_len==1:
            fig = plt.figure(figsize=(1+fig_len*fisize[0], fisize[1]))
        elif fig_len>1:
            fig = plt.figure(figsize=(fig_len*fisize[0], fisize[1]))
        else:
            fig = plt.figure(figsize=(fisize[0], fisize[1]*fig_len))
    else:
        fig = plt.figure(figsize=(10,12))
    
    # Generating subplot laout
    if height_ratios != [1]:
        gs1 = gridspec.GridSpec(layout[0], layout[1], height_ratios=height_ratios)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    else:
        gs1 = gridspec.GridSpec(layout[0], layout[1])
    gs1.update(wspace = wspace, hspace=hspace, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    
    # Building colors from colorbar
    if len(cbar_limits)>=1:
        cmin = cbar_limits[0]
        cmax = cbar_limits[1]
    else:
        cmin = np.min(cbar_data)
        cmax = np.max(cbar_data)
        
    if colors_len == 0:
        lencolors = len(cbar_data)
    else:
        lencolors = colors_len
    cmap =  matplotlib.colors.ListedColormap(sns.color_palette(color_palette, lencolors))
    normalize = matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax)
    cbar_colors = [cmap(normalize(value)) for value in cbar_data]

    
    # Generating axes 
    axes = []
    ax_num = 0
    for row in range(layout[0]):
        for col in range(layout[1]):
            
            #if sharex and row != 0 and col != 0:
            #    ax = fig.add_subplot(gs1[row, col], sharex=axes[0])
            #else:
            ax = fig.add_subplot(gs1[row, col])
                    
            axes.append(ax)
            # Accesing axis data
            #x = axis_data[ax_num][0]
            #y = axis_data[ax_num][1]
            #if colors[ax_num] == 'cbar':
            #    ax.scatter(x, y, c=cbar_colors, marker='o', edgecolors='k', zorder=3)
            #else:
            #    ax.scatter(x, y, c=colors[ax_num], marker='.', zorder=3)
            #ax.set_xlabel(xy_labels[ax_num][0], fontsize=fontsize, labelpad=1)
            #ax.set_ylabel(xy_labels[ax_num][1], fontsize=fontsize, labelpad=1)
            ax.tick_params(labelsize=10)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', width=1.0)
            ax.tick_params(axis='both', which='major', direction='in', length=5.0)
            ax.xaxis.set_tick_params(which='both', top =True, labeltop=False)
            ax.yaxis.set_tick_params(which='both', right=True, labelright=False)
            ax.tick_params(axis='both', which='both', direction='in')
            ax_num += 1
            
    if plot_cbar:
        # Adding colorbar
        ax_bar = [axes[0], axes[-1]]
        #cbar_mappable = cm.ScalarMappable(norm=normalize, cmap=cmap)
        cbar, cbar_colors = add_cbar(fig, ax_bar, cbar_data, cbar_label, color_palette=color_palette,
                               colors_len=colors_len, orientation=cbar_orientation,
                               sep=cbar_separation, width=0.02, height=False, ticks=ticks,
                               Mappable=False, cbar_limits=cbar_limits, tick_font = tick_font, label_font = label_font, 
                               discrete_colorbar=discrete_colorbar, formatter = formatter, norm=norm, labelpad=labelpad)

    return fig, gs1, axes, cbar_colors

def plot_contours_rangevel(cube_list, range_vel, rms, chan_width,
                           axis, color='k', lw=0.5, sigma_levels = [1,3,5], overall_sigma = 0):
    """
    Plot contours of a list of cubes
    The cubes are integ. intensities
    The range_vel is a dictio with the limits of the integrated velocities
    """
    cubes_dict = {}
    for j,cube in enumerate(cube_list):
        cube_fits = fits.open(cube)
        cube_data = cube_fits[0]
        cube_shape = cube_data.data.shape
        cube_header = cube_data.header
        cube_header['CDELT4'] = 1
        cube_wcs = WCS(cube_header)
        cube_wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
        cube_wcs = cube_wcs.dropaxis(3)
        cube_wcs = cube_wcs.dropaxis(2)
        for key, value in range_vel.items():
            if key in cube:
                dV = np.abs(value)
                if overall_sigma==0:
                    std = utiles.integratedsigma(1, dV, rms, chan_width)
                else:
                    std = overall_sigma
                cubes_dict[key] = {}
                cubes_dict[key]['std'] = std
                cubes_dict[key]['max'] = np.nanmax(cube_data.data[0,:,:,:])
                cubes_dict[key]['min'] = np.nanmin(cube_data.data[0,:,:,:])
                cube_levels = list(s*std for s in sigma_levels)
                for i in range(cube_shape[1]): 
                    axis.contour(cube_data.data[0,i,:,:], colors=color, levels=cube_levels,
                                linewidths=lw, transform=axis.get_transform(cube_wcs))
    return cubes_dict

def plot_contours_rangevel2(cube_list, FWHMs, rms, chan_width,
                           axis, color='k', lw=0.5, sigma_levels = [1,3,5]):
    """
    Plot contours of a list of cubes
    The cubes are integ. intensities
    The FWHMs is a dictio FWHM
    """
    cubes_dict = {}
    for j,cube in enumerate(cube_list):
        cube_fits = fits.open(cube)
        cube_data = cube_fits[0]
        cube_shape = cube_data.data.shape
        cube_header = cube_data.header
        cube_header['CDELT4'] = 1
        cube_wcs = WCS(cube_header)
        cube_wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
        cube_wcs = cube_wcs.dropaxis(3)
        cube_wcs = cube_wcs.dropaxis(2)
        for key, value in FWHMs.items():
            if key in cube:
                dV = np.abs(value)
                std = utiles.integratedsigma(1, dV, rms, chan_width)
                cubes_dict[key] = {}
                cubes_dict[key]['std'] = std
                cubes_dict[key]['max'] = np.nanmax(cube_data.data[0,:,:,:])
                cubes_dict[key]['min'] = np.nanmin(cube_data.data[0,:,:,:])
                cube_levels = list(s*std for s in sigma_levels)
                cubes_dict[key]['levels'] = cube_levels
                for i in range(cube_shape[1]): 
                    axis.contour(cube_data.data[0,i,:,:], colors=color, levels=cube_levels,
                                linewidths=lw, transform=axis.get_transform(cube_wcs))
    return cubes_dict
  
def subfig_initializer(layout,
                       hspace=0.0, wspace=0.0, figsize=[4,4]):
    # Generating figure size
    fig_len = layout[0]/layout[1]
    if fig_len==1:
        fig = plt.figure(figsize=(1+fig_len*figsize[0], figsize[1]))
    elif fig_len>1:
        fig = plt.figure(figsize=(fig_len*figsize[0], figsize[1]))
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1]*fig_len))
    # Generating subplot laout
    gs1 = gridspec.GridSpec(layout[0], layout[1])#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = wspace, hspace=hspace, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    # Generating axes 
    axes = []
    ax_num = 0
    for row in range(layout[0]):
        for col in range(layout[1]):
            ax = fig.add_subplot(gs1[row, col])
            axes.append(ax)
            ax.tick_params(labelsize=10)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', width=1.0)
            ax.tick_params(axis='both', which='major', direction='in', length=5.0)
            ax.xaxis.set_tick_params(which='both', top =True, labeltop=False)
            ax.yaxis.set_tick_params(which='both', right=True, labelright=False)
            ax.tick_params(axis='both', which='both', direction='in')
            ax_num += 1
    return fig, gs1, axes

def plotter_PV(dir_init, dir_fin, nombre, angle, centro, smooth=None, bins_pv=[10,10]):
    """
    Rotate the vel map and computes its Position-Velocity diagram
    
    INPUTS
    -------------
    nombre:        fits with vel map
    angle:         rotation angle, defined from N to E
    centro:        center of rotation
    smooth:        (opt). Smoothed map before rotation
    bin_pv:        list with bins in vel and dist for contours in PV
    
    OUTPUTS
    ------------- 
    m_rotada:      rotated vel map 
    Vel_rot:       velocity profile within 3pix along major axis
    CEN:           center of the rotated image
    fig:           figure of P-V
    """
    hdu=fits.open(dir_init+nombre)[0]
    m=hdu.data[0]
    m-=m[40,40]
    #if smooth!=None:
    #    g=Gaussian2DKernel(smooth)
    #    m=convolve2d(m,g,mode='same')
    m_rotada=rotateImage(m, angle, centro)
    CEN=[np.mean(np.where(m_rotada==0)[0],dtype=int),np.mean(np.where(m_rotada==0)[1],dtype=int)]
    x,y=np.meshgrid(np.arange(m_rotada.shape[1]),np.arange(m_rotada.shape[0]))
    #sacamos el perfil en las 3 pix centrales
    Vel_rot=[]
    #pixscale=0.125/abs(np.cos(np.radians(angle)))
    for i in np.arange(m_rotada.shape[0]):
        Vel_rot.append((m_rotada[i,CEN[0]]+m_rotada[i,CEN[0]-1]+m_rotada[i,CEN[0]+1])/3.0)#mean de las 3 lineas centrales
    Vel_rot=np.asarray(Vel_rot)
    #contornos con la densidad del P-V
    counts,xbins,ybins=np.histogram2d((y[~np.isnan(m_rotada)]-CEN[0]).ravel(),m_rotada[~np.isnan(m_rotada)].ravel(),bins=bins_pv)
    fig=plt.figure()
    """
    Ver que los (0,0) en pv y en vectores estan bien
    """
    plt.scatter(y-CEN[0],m_rotada,s=1)
    plt.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],linewidths=1,colors='k')
    plt.plot(np.arange(Vel_rot.shape[0])-CEN[0],Vel_rot,'r')
    plt.xlabel('Distance alog the major axis (arcsec)')
    plt.ylabel('Velocity (km/s)')
    plt.title('PA= '+str(round(angle,2)))
    return m_rotada,Vel_rot,CEN,fig

def rotateImage(img, angle, pivot):
    from scipy import ndimage
    padX = [img.shape[1] - pivot[1], pivot[1]]
    padY = [img.shape[0] - pivot[0], pivot[0]]
    imgP = np.pad(img-img[pivot[0],pivot[1]], [padY, padX], 'constant',constant_values=np.nan)
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def plot_profile(yvar, pix_range, px_mid, py_mid, cube_dict, cube_dict_err,
                 xlabel, ylabel, D_Mpc, plot_errors=False, plot_nouplim=True, dist_pc = True, molecname = '', xlog = False, tex_uplim = 100,
                 line_width_err = 0.7, labelsize = 12, logscale = True, ylims = False, xlims = False, yformatter='%.0f', xformatter ='%.1f'
                 , figures_path='./', source='', pc_distance = False):
    fig = plt.figure()  #
    ax = fig.add_subplot(1, 1, 1)
    
    xtotal_no_uplim = []
    ytotal_no_uplim = []
    ytotal_np_uplim_err = []
    x_uplim = []
    y_uplim = []
    y_uplim_err = []
    
    pixsize = np.round(cube_dict[yvar].header['CDELT2']*3600,4)
    pixel_pc = u_conversion.lin_size(D_Mpc, pixsize).to(u.pc).value
    
    data = cube_dict[yvar].fitsdata[0,0,:,:]
    data_err = cube_dict_err[yvar+'_err'].fitsdata[0,0,:,:]
    
    df_list = []
    for rad in range(0,pix_range+1):
        
        mask, distance = utiles.createAnnularMask_1px(data.shape[0], data.shape[1], [px_mid, py_mid], rad)
        result = data*mask # keeps the ndarray shape
        points = data[mask] # selects only the px in the ring
        mask_err, distance = utiles.createAnnularMask_1px(data_err.shape[0], data_err.shape[1], [px_mid, py_mid], rad)
        result_err = data_err*mask_err # keeps the ndarray shape
        points_err = data_err[mask_err] # selects only the px in the ring
        if yvar == 'tex':
            for i,po in enumerate(points):
                if (po == tex_uplim) and (np.isnan(points_err[i])):
                    points_err[i] = -10
        dist_px = distance[mask]
        df_dict = {'value': points, 'value_err': points_err,
                   'distance_px': dist_px, 'distance_pc': dist_px*pixel_pc
                   }
        df = pd.DataFrame(df_dict)
        df_list.append(df)
        #xrange = [distance]*len(points)
        if dist_pc:
            xrange = dist_px*pixel_pc
            if xlog:
                ax.set_xscale('log')
        else:
            if xlog:
                xrange = dist_px
            else:
                xrange = np.log10(distance[mask])
        if plot_errors:
            if yvar == 'coldens':
                points_err_plot = (10**np.array(points_err))*(1/np.log(10))/(10**np.array(points))
            else:
                points_err_plot = points_err
            for p, point in enumerate(points):
                if points_err[p] == -1:
                    # Upper limit
                    ax.errorbar(xrange[p], points[p], 
                             uplims=True,
                             yerr=points[p]*0.1,
                             marker='o', markersize=5,
                             markerfacecolor='k',
                             markeredgecolor='k', markeredgewidth=0.8,
                             ecolor='k',
                             color = 'k',
                             elinewidth= line_width_err,
                             barsabove= True,
                             zorder=2)
                elif points_err[p] == -10:
                    # Tex fixed for logN upper limit
                    ax.errorbar(xrange[p], points[p], 
                             uplims=True,
                             yerr=points[p]*0.1,
                             marker='o', markersize=5,
                             markerfacecolor='k',
                             markeredgecolor='k', markeredgewidth=0.8,
                             ecolor='r',
                             color = 'k',
                             elinewidth= line_width_err,
                             barsabove= True,
                             zorder=2)
                    
                else:
                    ax.errorbar(xrange[p], points[p], 
                             yerr=points_err_plot[p],
                             marker='o', markersize=5,
                             markerfacecolor='k',
                             markeredgecolor='k', markeredgewidth=0.8,
                             ecolor='k',
                             color = 'k',
                             elinewidth= line_width_err,
                             barsabove= True,
                             zorder=2)
        elif plot_nouplim:
            x_no_uplim = []
            y_no_uplim = []
            y_np_uplim_err = []
            plotstr = 'nouplim'
            for p, point in enumerate(points):
                
                if (points_err[p] != -1) and (points_err[p] != -10):
                    ax.errorbar(xrange[p], points[p], 
                             yerr=points_err[p],
                             marker='o', markersize=5,
                             markerfacecolor='None',
                             markeredgecolor='k', markeredgewidth=0.8,
                             ecolor='k',
                             color = 'k',
                             elinewidth= line_width_err,
                             barsabove= True,
                             zorder=2)
                    x_no_uplim.append(xrange[p])
                    y_no_uplim.append(points[p])
                    y_np_uplim_err.append(points_err[p])
        else:
            plotstr = 'all'
            x_no_uplim = []
            y_no_uplim = []
            y_np_uplim_err = []
            x_uplim = []
            y_uplim = []
            y_uplim_err = []
            
            for p, point in enumerate(points):
                if ((points_err[p] != -1) and (points_err[p] != -10)) and (~np.isnan(points_err[p])):

                    xtotal_no_uplim.append(xrange[p])
                    ytotal_no_uplim.append(points[p])
                    ytotal_np_uplim_err.append(points_err[p])
                    x_no_uplim.append(xrange[p])
                    y_no_uplim.append(points[p])
                    y_np_uplim_err.append(points_err[p])
                elif yvar == 'tex' and xrange[p] == 0.0:
                    # Central pixel is not upper limit despite we have calculated it that way
                    xtotal_no_uplim.append(xrange[p])
                    ytotal_no_uplim.append(points[p])
                    ytotal_np_uplim_err.append(points_err[p])
                    x_no_uplim.append(xrange[p])
                    y_no_uplim.append(points[p])
                    y_np_uplim_err.append(points_err[p])
                else:

                    x_uplim.append(xrange[p])
                    y_uplim.append(points[p])
                    y_uplim_err.append(points_err[p])
            #if yvar == 'coldens':
            if yvar == 'coldens':
                
                y_uplim_err = (10**np.array(y_uplim_err))*(1/np.log(10))/(10**np.array(y_uplim))    
            ax.plot(x_no_uplim, y_no_uplim, color='k', marker='o', markersize = 5, linestyle='None', markerfacecolor = 'None')
            ax.plot(x_uplim, y_uplim, color='b', marker='o', markersize = 5, linestyle='None', markerfacecolor = 'None')        
    if logscale:
        ax.set_yscale('log')
    def func(x, a, b, c, d):
        return  a * x *x + b*x + c

    ax.yaxis.set_major_formatter(FormatStrFormatter(yformatter))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xformatter))
    if ylims:
        ax.set_ylim(ylims)
    if xlims:
        ax.set_xlim(xlims)   
    ax.set_xlabel(xlabel, fontsize = labelsize)
    ax.set_ylabel(ylabel, fontsize = labelsize)
    
    df_final = pd.concat(df_list)
    figsavename = figures_path+source+'_'+molecname+'_SLIM_'+yvar+'_profile_'+plotstr+'_v1.pdf'
    return fig, ax, figsavename, df_final
    
    
    
    

def plot_mean_profile(yvar, pix_range, px_mid, py_mid, cube_dict, cube_dict_err,
                 xlabel, ylabel, D_Mpc, plot_errors=False, plot_nouplim=True, dist_pc = False, molecname='',
                 line_width_err = 0.7, labelsize = 12, logscale = True, ylims = False, xlims=False, xlog=False, tex_uplim=100,
                 yformatter='%.0f', xformatter ='%.1f', figures_path='./', source=''):
    """
    Plots the profile of a cube taking "rings" and averaging by distance
    """
    
    fig = plt.figure()  
    ax = fig.add_subplot(1, 1, 1)
    
    pixsize = np.round(cube_dict[yvar].header['CDELT2']*3600,4)
    pixel_pc = u_conversion.lin_size(D_Mpc, pixsize).to(u.pc).value
    
    x_no_uplim = []
    y_no_uplim = []
    y_no_uplim_err = []
    y_np_uplim_err = []
    x_uplim = []
    y_uplim = []
    y_uplim_err = []
    
    
    df_list = []
    ave_y_vals = []
    ave_y_vals_err = []
    ave_x_vals_err = []
    for rad in range(0,pix_range+1):
        data = cube_dict[yvar].fitsdata[0,0,:,:]
        data_err = cube_dict_err[yvar+'_err'].fitsdata[0,0,:,:]
        mask, distance = utiles.createAnnularMask_1px(data.shape[0], data.shape[1], [px_mid, py_mid], rad)
        result = data*mask # keeps the ndarray shape
        points = data[mask] # selects only the px in the ring
        points_ave = np.nanmean(points)
        mask_err, distance = utiles.createAnnularMask_1px(data_err.shape[0], data_err.shape[1], [px_mid, py_mid], rad)
        result_err = data_err*mask_err # keeps the ndarray shape
        points_err = data_err[mask_err] # selects only the px in the ring
        if yvar == 'tex':
            for i,po in enumerate(points):
                if (po == tex_uplim) and (np.isnan(points_err[i])):
                    points_err[i] = -10
        xrange = distance[mask]
        dist_px = distance[mask]
        df_dict = {'value': points, 'value_err': points_err,
                   'distance_px': dist_px, 'distance_pc': dist_px*pixel_pc
                   }
        df = pd.DataFrame(df_dict)
        df_list.append(df)
        if dist_pc:
            xrange = dist_px*pixel_pc
            if xlog:
                ax.set_xscale('log')
        else:
            if xlog:
                xrange = dist_px
            else:
                xrange = np.log10(distance[mask])
        
        unique_rad =np.unique(xrange)

        for u_rad in unique_rad:
            rad_sep = utiles.list_duplicates_of(list(xrange), u_rad)
            xrange_rad = xrange[rad_sep]
            points_rad = points[rad_sep]
            points_err_rad = points_err[rad_sep]
            x_sub_no_uplim = []
            y_sub_no_uplim = []
            y_sub_np_uplim_err = []
            y_sub_no_uplim_err = []
            x_sub_uplim = []
            y_sub_uplim = []
            y_sub_uplim_err = []
            
            if plot_nouplim:
                plotstr = 'nouplim'
                for p, point in enumerate(points_rad):
                    if ((points_err_rad[p] != -1) and (points_err_rad[p] != -10)) and (~np.isnan(points_err_rad[p])):
                        x_sub_no_uplim.append(xrange_rad[p])
                        y_sub_no_uplim.append(points_rad[p])
                        y_sub_no_uplim_err.append(points_err_rad[p])
                    elif yvar == 'tex' and xrange_rad[p] == 0.0:
                        x_sub_no_uplim.append(xrange_rad[p])
                        y_sub_no_uplim.append(points_rad[p])
                        y_sub_no_uplim_err.append(120)
                            
                    else:
                        x_sub_uplim.append(xrange_rad[p])
                        y_sub_uplim.append(points_rad[p])
                        y_sub_uplim_err.append(points_err_rad[p])
                if x_sub_no_uplim: # Not empty
                    x_no_uplim.append(np.nanmean(x_sub_no_uplim))
                    y_no_uplim.append(np.nanmean(y_sub_no_uplim))
                    y_no_uplim_err.append(np.nanmean(y_sub_no_uplim_err))
                if x_sub_uplim:
                    x_uplim.append(np.nanmean(x_sub_uplim))
                    y_uplim.append(np.nanmean(y_sub_uplim))
                    y_uplim_err.append(np.nanmean(y_sub_uplim_err))
                
                
                if yvar == 'coldens':
                    y_sub_no_uplim_err = (10**np.array(y_sub_no_uplim_err))*(1/np.log(10))/(10**np.array(y_sub_no_uplim))
                    uplim = 0.2
                else:
                    uplim = np.nanmean(y_sub_uplim)*0.1
                print(f'val: {y_sub_no_uplim} err: {y_sub_no_uplim_err}')
                
                weights = 1/(np.array(y_sub_no_uplim_err)**2)
                if len(x_sub_no_uplim)>1 and len(y_sub_no_uplim)>1:
                    ave_error1 = utiles.weight_mean_err(y_sub_no_uplim, weights)
                    ave, ave_error = utiles.weighted_avg_and_std(y_sub_no_uplim, weights)
                    print(f'ave {ave:1.3f}  ave_error {ave_error:1.3f} ave_error1 {ave_error1:1.3f}')
                    #ax.errorbar(np.nanmean(x_sub_no_uplim), np.nanmean(y_sub_no_uplim), 
                    ax.errorbar(np.nanmean(x_sub_no_uplim), ave, 
                                         yerr=ave_error,
                                         marker='o', markersize=5,
                                         markerfacecolor='None',
                                         markeredgecolor='k', markeredgewidth=0.8,
                                         ecolor='k',
                                         color = 'k',
                                         elinewidth= line_width_err,
                                         barsabove= True,
                                         zorder=2)
                    ave_y_vals.append(ave)
                    ave_y_vals_err.append(ave_error)
                    ave_x_vals_err.append(np.nanmean(x_sub_no_uplim))
                elif len(x_sub_no_uplim)==1 and len(y_sub_no_uplim)==1:
                    print(f'ave {np.nanmean(y_sub_no_uplim):1.3f}  ave_error {np.nanmean(y_sub_no_uplim_err):1.3f} ave_error1 0000')
                    ax.errorbar(np.nanmean(x_sub_no_uplim), np.nanmean(y_sub_no_uplim), 
                                         yerr=np.nanmean(y_sub_no_uplim_err),
                                         marker='o', markersize=5,
                                         markerfacecolor='None',
                                         markeredgecolor='k', markeredgewidth=0.8,
                                         ecolor='k',
                                         color = 'k',
                                         elinewidth= line_width_err,
                                         barsabove= True,
                                         zorder=2)
                    ave_y_vals.append(np.nanmean(y_sub_no_uplim))
                    ave_y_vals_err.append(np.nanmean(y_sub_no_uplim_err))
                    ave_x_vals_err.append(np.nanmean(x_sub_no_uplim))
                    
                ax.errorbar(np.nanmean(x_sub_uplim), np.nanmean(y_sub_uplim), 
                                     uplims=True,
                                     yerr=uplim,
                                     marker='o', markersize=5,
                                     markerfacecolor='None',
                                     markeredgecolor='b', markeredgewidth=0.8,
                                     ecolor='b',
                                     color = 'b',
                                     elinewidth= line_width_err,
                                     barsabove= True,
                                     zorder=2)
                plotstr = 'all'
            else:
                plotstr = 'all'
    
                for p, point in enumerate(points):
                    if (points_err[p] != -1) and (points_err[p] != -10):
                        x_sub_no_uplim.append(xrange_rad[p])
                        y_sub_no_uplim.append(points_rad[p])
                        y_sub_np_uplim_err.append(points_err_rad[p])
                        
                    if x_sub_no_uplim: # Not empty
                        x_no_uplim.append(np.nanmean(x_sub_no_uplim))
                        y_no_uplim.append(np.nanmean(y_sub_no_uplim))
                        y_np_uplim_err.append(np.nanmean(y_sub_no_uplim_err))
                        
                ax.plot(x_no_uplim, y_no_uplim, color='k', marker='o', markersize = 5, linestyle='None', markerfaceolor='None')
    if logscale:
        ax.set_yscale('log')
                
    def func(x, a, b, c, d):
        return  a * x *x + b*x + c
    
    if yvar != 'coldens':
        weights = 1/np.array(y_no_uplim_err)
        def logtex_prof(rn, A, alpha, beta, b, gamma):
            return (A * (rn**alpha)*np.exp(-beta*rn)/(1.+b*(rn**gamma)))

    ax.yaxis.set_major_formatter(FormatStrFormatter(yformatter))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xformatter))
    if ylims:
        ax.set_ylim(ylims)
    if xlims:
        ax.set_xlim(xlims)  
    ax.set_xlabel(xlabel, fontsize = labelsize)
    ax.set_ylabel(ylabel, fontsize = labelsize)
    plotstr = 'all'
    df_final = pd.concat(df_list)
    figsavename = figures_path+source+'_'+molecname+'_SLIM_'+yvar+'_mean_profile_'+plotstr+'_v1.pdf'
    df_dict = {'distance_pc': ave_x_vals_err, 'vale': ave_y_vals, 'value_err':ave_y_vals_err}
    df = pd.DataFrame(df_dict)
    return fig, ax, figsavename, df_final, df

    #fig.savefig(figures_path+source+'_SLIM_'+yvar+'_mean_profile_'+plotstr+'_v0.pdf', bbox_inches='tight', transparent=True, dpi=400)
    #plt.close()
    
def plot_ratio_profile(pix_range, px_mid, py_mid, cube,
                       xlabel, ylabel, D_Mpc, dist_pc = True, molecname = '', xlog = False,
                       line_width_err = 0.7, labelsize = 12, logscale = True, ylims = False, xlims = False, yformatter='%.0f', xformatter ='%.1f',
                       figures_path='./', source='', pc_distance = False):
    fig = plt.figure()  #
    ax = fig.add_subplot(1, 1, 1)
    
    pixsize = np.round(cube.header['CDELT2']*3600,4)
    pixel_pc = u_conversion.lin_size(D_Mpc, pixsize).to(u.pc).value
    
    data = cube.fitsdata[0,0,:,:]
    
    df_list = []
    for rad in range(1,pix_range+1):
        
        mask, distance = utiles.createAnnularMask_1px(data.shape[0], data.shape[1], [px_mid, py_mid], rad)
        points = data[mask] # selects only the px in the ring
        
        dist_px = distance[mask]
        df_dict = {'value': points, 
                   'distance_px': dist_px, 'distance_pc': dist_px*pixel_pc
                   }
        df = pd.DataFrame(df_dict)
        df_list.append(df)
        #xrange = [distance]*len(points)
        if dist_pc:
            xrange = dist_px*pixel_pc
        else:
            if xlog:
                xrange = dist_px
            else:
                xrange = np.log10(distance[mask]) 
        plotstr = 'all'
        ax.plot(xrange, points, color='k', marker='o', markersize = 5, linestyle='None', markerfacecolor = 'None')
    if logscale:
        ax.set_yscale('log')
    
    ax.yaxis.set_major_formatter(FormatStrFormatter(yformatter))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xformatter))
    if ylims:
        ax.set_ylim(ylims)
    if xlims:
        ax.set_xlim(xlims)   
    ax.set_xlabel(xlabel, fontsize = labelsize)
    ax.set_ylabel(ylabel, fontsize = labelsize)
    
    df_final = pd.concat(df_list)
    figsavename = figures_path+source+'_ratio_'+molecname+'_SLIM_profile_'+plotstr+'_v1.pdf'
    return fig, ax, figsavename, df_final

def curly_arrow(start, end, ax, arr_size = 1, n = 5, col='gray', linew=1., width = 0.1):
    # https://stackoverflow.com/questions/45365158/matplotlib-wavy-arrow
    xmin, ymin = start
    xmax, ymax = end
    dist = np.sqrt((xmin - xmax)**2 + (ymin - ymax)**2)
    n0 = dist / (2 * np.pi)

    x = np.linspace(0, dist, 151) + xmin
    y = width * np.sin(n * x / n0) + ymin
    line = plt.Line2D(x,y, color=col, lw=linew)

    del_x = xmax - xmin
    del_y = ymax - ymin
    ang = np.arctan2(del_y, del_x)

    line.set_transform(mpl.transforms.Affine2D().rotate_around(xmin, ymin, ang) + ax.transData)
    ax.add_line(line)

    verts = np.array([[0,1],[0,-1],[2,0],[0,1]]).astype(float) * arr_size
    verts[:,1] += ymax
    verts[:,0] += xmax
    path = mpath.Path(verts)
    patch = mpatches.PathPatch(path, fc=col, ec=col)

    patch.set_transform(mpl.transforms.Affine2D().rotate_around(xmax, ymax, ang) + ax.transData)
    return patch

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)