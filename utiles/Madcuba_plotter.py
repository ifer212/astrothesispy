# =============================================================================
#   Spectra plotter for data extracter with MADCUBA
#       https://cab.inta-csic.es/madcuba/
# =============================================================================

import re
import os
import glob
import warnings
import string
import itertools
import pandas as pd
import numpy as np
from numpy.random import *
from pathlib import Path
from copy import deepcopy
import astropy.units as u
import astropy.constants.si as _si

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

def MADCUBA_plot(m, n, data_path, molec_info, save_path, fig_name='Figure', fig_format='png', 
                 x_limits={}, y_limits={},x_label='x', y_label='y',
                 drawstyle='histogram', spec_color='k', linewidth=0.8, linestyle='-',
                 fit_color=['red'], fit_linewidth=0.8, fit_linestyle='-', only_sumfit = True,
                 unit = 'mJy', spectra_rms = [], xsep = 4.0, vel_shift=0,
                 highlight_lines={'NA':'color'}, label_color='k', fill_color='0.65',
                 labelsize=12, labelfont='Arial', molfontize=6, moleculefont='courier new', 
                 anotcolor='0.6', panelfont='Arial', panfontsize=8, tick_font = 12, txtypercen=0.04,
                 txtxpercen=0.04, x_axis_tickpos = [], erase_molecule = [], erase_line = {},
                 panel_naming='letters', panesanot_right = ['NA'], panesanot_left = ['NA'], new_labelling = {},
                 all_species = True, move_lines = {}, plot_as_one = [True,4], frequencies=True
                 ):
    """ Ring plot from Madcuba extracted spectra and LTE model

    Args:
        m (int): Number of rows.
        n (int): Number of columns.
        data_path (str): Path to spectral data.
        save_path (str): Path to save figure.
        fig_name (str, optional): Figure name. Defaults to 'Figure'.
        fig_format (str, optional): Figure format. Defaults to 'png'.
        x_limits (dict, optional): X axis limits. Defaults to {}, leaving matplotlib to choose them.
        y_limits (dict, optional): Y axis limits. Defaults to {}, leaving matplotlib to choose them.
        x_label (str, optional): X axis label. Defaults to 'x'.
        y_label (str, optional): Y axis label. Defaults to 'y'.
        drawstyle (str, optional): Drawing style for the spectra. Defaults to 'histogram'.
        spec_color (str, optional): Spectra line color. Defaults to 'k'.
        linewidth (float, optional): Spectra line width. Defaults to 0.8.
        linestyle (str, optional): Spectra line style. Defaults to '-'.
        fit_color (list, optional): Fitted model line color. Defaults to ['red'].
        fit_linewidth (float, optional): Fitted model line width. Defaults to 0.8.
        fit_linestyle (str, optional): Fitted model line style. Defaults to '-'.
        only_sumfit (bool, optional): Plots only the sum of all species (all species = True in Madcuba). Defaults to True.
        unit (str, optional): Y axis units. Defaults to 'mJy'.
        spectra_rms (list, optional): Spectra rms, fills with fill_color above 3*rms. Defaults to [].
        xsep (float, optional): Separation percentage between labels. Defaults to 4.0.
        vel_shift (int, optional): Manual velocity shift for the spectra. Defaults to 0.
        highlight_lines (dict, optional): Molecular lines to highlight and color. Defaults to {'NA':'color'}.
        label_color (str, optional): Molcule labels color. Defaults to 'k'.
        fill_color (str, optional): Fill color for spectra above 3*rms. Defaults to '0.65'.
        labelsize (int, optional): Label font size. Defaults to 12.
        labelfont (str, optional): Label font. Defaults to 'Arial'.
        molfontize (int, optional): Molecule label font size. Defaults to 6.
        moleculefont (str, optional): Molecule font. Defaults to 'courier new'.
        anotcolor (str, optional): Color panel for anotations. Defaults to '0.6'.
        panelfont (str, optional): Font for panel anotations. Defaults to 'Arial'.
        panfontsize (int, optional): Panel anotation font size. Defaults to 8.
        tick_font (int, optional): Tick label font size. Defaults to 12.
        txtypercen (float, optional): Percentage of panel height to consider labels are close. Defaults to 0.04.
        txtxpercen (float, optional): Percentage of panel width to consider labels are close. Defaults to 0.04.
        x_axis_tickpos (list, optional): Place only xaxis tick labels at these positions. Defaults to [].
        erase_molecule (list, optional): Not plotting line nor label for specified molecules. Defaults to [].
        erase_line (dict, optional): Not plotting specific line, given label, panel number and frequency, e.g. erase_line = {'label':  ['H2CO', 'C2H3CN,v=0'], 'panel': [2, 2], 'freq': [2.18475632E11, 2.185850717E11]}. Defaults to {}.
        panel_naming (str, optional): Anotate panel name. Defaults to 'letters'.
        panesanot_right (list, optional): Location to write panel name. Defaults to ['NA'].
        panesanot_left (list, optional): Location to write panel name. Defaults to ['NA'].
        new_labelling (dict, optional): Renaming some molecules. Defaults to {}.
        all_species (bool, optional): Plotting all species model. Defaults to True.
        move_lines (dict, optional): Manually positioning some labels. Defaults to {}.
        plot_as_one (list, optional): Plotting only 1 line and label for transitions from the same molecule with the same freq up to X decimals in GHz. Defaults to [True,4].
        frequencies (bool, optional): Plot in frequencies instead of velocity. Defaults to True.

    Raises:
        IOError: _description_
        IOError: _description_
    """

    matplotlib.rcParams['pdf.fonttype'] = 42 # Using TrueType fonts
    matplotlib.rcParams['ps.fonttype']  = 42 # Using TrueType fonts
    mpl.rc('xtick', color='k', direction='in', labelsize=tick_font)
    mpl.rc('ytick', color='k', direction='in', labelsize=tick_font)
    # Width of axes
    plt.rc('axes', linewidth=1.5)
    
    # Paths
    data_files = sorted(glob.glob(data_path+'data/*'))
    data_hc3n_files = sorted(glob.glob(data_path+'data_HC3N/*'))
    spec_all_files = sorted(glob.glob(data_path+'spectroscopy/*'))
    spec_sel_files = sorted(list(set(spec_all_files)-set(sorted(glob.glob(data_path+'spectroscopy/*TRANSITIONS_ALL*')))))
    spec_all_files = sorted(glob.glob(data_path+'spectroscopy/*TRANSITIONS_ALL*'))
    # Number of existing specs
    spec_number = len(data_files)
    # Checking if grid is smaller than number of panels from SLIM
    if m*n < spec_number:
        raise IOError('Entered grid (%sx%x) smaller than number of panels: %s' %(m,n,len(glob.glob(data_path+'/data/*'))))
    # Setting draw style
    if drawstyle=='histogram':
        ds='steps-mid'
    else:
        ds='default'
    # Figure Size
    size_rat = float(n)/float(m)
    size_x = 23.#*size_rat
    size_y = 24.-1.
    
    fig = plt.figure(figsize=(size_x, size_y*1.15))
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.065, top=0.95, bottom = 0.05)   
    
    axis = []
    y_max_list = []
    y_min_list = []
    y_lims_s = []
    panel_names = []
    y_max_dict = {}
    y_local_limits = []
    panel_spec_df_list = []
    axis_ind = []

    # Generating specified number of axis
    for i in range(m*n):
        row = (i // n)
        col = i % n
    ind = 0
    axis_ind = []
    for i in range(m):
        axis_ind.append([])
        for j in range(n):
            axis_ind[i].append(ind)
            ind += 1
    # Reading Data for plotting
    for j in range(m*n):
        x_limits_spec = x_limits[j][0]
        file_ind = x_limits[j][1]
        if Path(data_files[file_ind]).is_file():        
            panel_names.append(data_files[file_ind].split('/')[-1])
            # Reading asciis from SLIM
            data = pd.read_csv(data_files[file_ind], delim_whitespace= True, header=None)
            data_ncolumns = data.shape[1]
            data_cols = ['vel', 'int']
            fit_ncols = data_ncolumns - 2
            data_hc3n = pd.read_csv(data_hc3n_files[file_ind], delim_whitespace= True, header=None)
            axis.append(fig.add_subplot(gs1[j]))
            # Raise warning when fit_color list is smaller than the number of fits
            if only_sumfit != True: 
                if len(fit_color) < fit_ncols:
                    warnings.warn('Fit color list lenght is smaller than number of fits plotted,'
                                  ' setting default colors.')
                    cmap = matplotlib.cm.get_cmap('tab10')
                    lin = np.linspace(0, 1, fit_ncols)
                    for r, rgb in enumerate(lin):
                        fit_color.append(cmap(rgb)) 
            # Appending if there is more than one fit available
            if data_ncolumns > 2:
                for f in range(data_ncolumns-2):
                    data_cols.append('fit_'+str(f+1))
            data.columns = data_cols
            data_hc3n.columns = data_cols
            # y-axis Max and min for every panel
            subdata = data[(data['vel'] >= x_limits_spec[0]) & (data['vel'] <= x_limits_spec[1])]
            y_max_list.append(np.nanmax([subdata['int'].max(skipna=True), subdata['fit_'+str(1)].max(skipna=True)]))
            y_min_list.append(np.nanmin([subdata['int'].min(skipna=True), subdata['fit_'+str(1)].max(skipna=True)]))
            y_lims_s.append([np.nanmin([subdata['int'].min(skipna=True), subdata['fit_'+str(1)].max(skipna=True)]),
                            np.nanmax([subdata['int'].max(skipna=True), subdata['fit_'+str(1)].max(skipna=True)])])
            y_max_dict[j] = np.nanmax([subdata['int'].max(skipna=True), subdata['fit_'+str(1)].max(skipna=True)])
            # Ploting rms line
            axis[j].fill_between(data['vel'], spectra_rms[j], data['int'], 
                                where=spectra_rms[j] <= data['int'], color=fill_color)#'0.7')
            # Plotting spectrum
            axis[j].plot(data['vel'], data['int'], linewidth=linewidth, 
                        linestyle=linestyle, drawstyle=ds, color=spec_color)
            # Plotting LTE fit
            if only_sumfit == True: # only sum of all fits
                axis[j].plot(data['vel'], data['fit_'+str(1)], color=fit_color[0],
                            linewidth=fit_linewidth, linestyle=fit_linestyle)
            else: # all fits
                if np.nanmin(data['vel']) < 230:
                    fit_color[1] = 'g' # J=24-23
                else:  
                    fit_color[1] = 'b' # J=26-25
                axis[j].plot(data_hc3n['vel'], data_hc3n['fit_'+str(1)], color=fit_color[1],
                            linewidth=fit_linewidth, linestyle=fit_linestyle)
                axis[j].plot(data['vel'], data['fit_'+str(1)], color=fit_color[0],
                            linewidth=fit_linewidth, linestyle='--')
            panel_spec_df_list.append(data)
            # Overall max and min to set same axis limits in all panels
            axis[j].set_xlim(x_limits_spec)
            y_local_limits.append([y_min_list[j],  y_max_list[j]])
            # Y axis limits
            ## Checking if there are different y limits per row
            if y_limits != {}:
                y_total_max = np.nanmax(y_limits[j])
                y_total_min = np.nanmin(y_limits[j])
                axis[j].set_ylim(y_limits[j])    
            else:
                y_total_max = np.nanmax(y_max_list)
                y_total_min = np.nanmin(y_min_list)
                y_limits_pl = [y_lims_s[j][0], y_lims_s[j][1]+y_lims_s[j][1]*0.45]
                ymax_t =  y_lims_s[j][1]+ y_lims_s[j][1]*0.45
                yaxis_len = np.abs(y_limits_pl[0]-y_limits_pl[1])
                txt_height = 0.04*(yaxis_len)
    
    # Text annotation
    xtext, ytext, ltext = line_annotation(all_species, data_files, m, n, spec_all_files, spec_sel_files, panel_names,
                                           y_total_min, y_total_max, y_local_limits, y_limits, x_limits, axis_ind, y_max_list,
                                           axis, molfontize, moleculefont, 
                                           highlight_lines, erase_molecule, erase_line, vel_shift, label_color,
                                           panel_spec_df_list, xsep, new_labelling,
                                           move_lines, plot_as_one, txtypercen, txtxpercen, frequencies, molec_info)
    text_maxy = {}
    for r,row in enumerate(axis_ind):
        ysublims = [ytext[i] for i in row ]
        ysublimstext = [ltext[i] for i in row]
        text_panel = {}
        rowypos = []
        textlen = []
        rowmaxy = []
        for i,panel in enumerate(row):
            if ysublims[i]:
                last =  {'ypos': np.nanmax(ysublims[i]), 'textlen': len(ysublimstext[i][np.where(ysublims[i]==np.nanmax(ysublims[i]))[0][0]])}
                text_panel[panel] = {'ypos': np.nanmax(ysublims[i]), 'textlen': len(ysublimstext[i][np.where(ysublims[i]==np.nanmax(ysublims[i]))[0][0]])}
            else:
                text_panel[panel] = last
            rowypos.append(text_panel[panel]['ypos'])
            textlen.append(text_panel[panel]['textlen'])
            rowmaxy.append(y_max_dict[panel])
        for i,panel in enumerate(row):
            text_maxy[panel] = {'ypos': np.nanmax(rowypos), 'textlen': np.nanmax(textlen), 'ylim': np.nanmax(rowmaxy)}
    #### Panel parameters
    # Left Figures
    left_ind = []
    for i in range(m):
        left_ind.append(axis_ind[i][0])
    # Pane names
    if panel_naming == 'letters':
        panenames_1 = list(string.ascii_lowercase)[0:len(axis)+1]
        panenames_2 = []
        panenames_3 = []
        if m*n > 26:
            panenames_2 = [''.join(i).strip() for i in zip(['a']*len(panenames_1), panenames_1)]
        elif m*n > 26*2:
            panenames_3 = [''.join(i).strip() for i in zip(['b']*len(panenames_1), panenames_1)]
        elif m*n > 26*3:
            raise IOError('Too many plots (>78): %s' %(m*n))
        panenames = panenames_1+panenames_2+panenames_3
    elif panel_naming == 'numbers':
        panenames =  [str(x)+')' for x in range(1, len(axis) + 1)]
    else:
        panenames = []
        
    for i, ax in enumerate(axis):
        ax.tick_params(axis='both', which='both', direction='in', width=1.0)
        if bool(x_axis_tickpos):
            xticks_loc = x_axis_tickpos
        else:
            if i == 0:
                xticks_loc = ax.get_xticks().tolist()
                xticks_loc = xticks_loc[2:-2]
        if frequencies:
            ax.minorticks_on()
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            minor_locator = AutoMinorLocator(5)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_minor_locator(minor_locator)
            ax.xaxis.set_tick_params(which='both', top ='on')
            ax.yaxis.set_tick_params(which='both', right='on', labelright='off')
        else:
            label_format = '{:,.0f}'
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks_loc))
            ax.set_xticklabels([label_format.format(x) for x in xticks_loc])
            ax.minorticks_on()
            ax.xaxis.set_tick_params(which='both', top ='on')
            ax.yaxis.set_tick_params(which='both', right='on', labelright='off')
        if len(panenames)>0:
            axis[i].text(0.04, 0.95, panenames[i],
                                horizontalalignment='left',
                                verticalalignment='top',
                                fontsize=panfontsize, fontname=panelfont,
                                transform=axis[i].transAxes)
        if panesanot_right != ['NA']:
            axis[i].text(0.65, 0.95, panesanot_right[i],
                                    horizontalalignment='left',
                                    verticalalignment='top',
                                    color = anotcolor,
                                    fontsize=panfontsize, fontname=panelfont,
                                    fontweight='bold',
                                    transform=axis[i].transAxes)
        if panesanot_left != ['NA']:
            axis[i].text(0.25, 0.95, panesanot_left[i],
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    color = anotcolor,
                                    fontsize=panfontsize, fontname=panelfont,
                                    fontweight='bold',
                                    transform=axis[i].transAxes)
        # Y axis limits
        ## Checking if there are different y limits per row
        for j in range(m*n):
            if y_limits != {}:
                ax.set_ylim(y_limits[j])    
            else:
                y_limits_pl = [y_lims_s[j][0], y_lims_s[j][1]+y_lims_s[j][1]*0.45]
                ymax_t =  y_lims_s[j][1]+ y_lims_s[j][1]*0.45
                yaxis_len = np.abs(y_limits_pl[0]-y_limits_pl[1])
                txt_height = 0.04*(yaxis_len)
                if text_maxy[i]['ypos']+txt_height*text_maxy[i]['textlen'] >= y_limits_pl[1]*0.5:
                    ax.set_ylim([y_limits_pl[0], text_maxy[i]['ypos']+0.75*txt_height*text_maxy[i]['textlen']]) 
                else:
                    ax.set_ylim([y_lims_s[j][0], ymax_t]) 
        ax.tick_params(axis="x", which='major', length=14, width=1.3)
        ax.tick_params(axis="x", which='minor', length=8, width = 1.3)
        ax.tick_params(axis="y", which='major', length=11)
        ax.tick_params(axis="y", which='minor', length=5)
        # Only adding y axis labels to left axes
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        ax.tick_params(labelleft=True,
                       labelright=False)
        if unit == 'mJy':
            axis_mJy = [i for i in np.round(ax.get_yticks()*1000, decimals=2)]
            ticks_loc = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            label_format = '{:,.1f}'
            ax.set_yticklabels([label_format.format(x) for x in axis_mJy])
        if y_label != 'NA':
            ax.set_ylabel(y_label, fontsize=labelsize, fontname=labelfont, fontweight='bold')
        else:
            ax.set_yticklabels([])
        # Only adding x axis labels to bottom axes
        if i >= len(axis)-1:
            ax.set_xlabel(x_label, fontsize=labelsize, fontname=labelfont, fontweight='bold')
    fig.savefig(f'{save_path}/{fig_name}{fig_format}', bbox_inches='tight', transparent=True,
                dpi=400)
    plt.close()
    
    
    
    

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions


def findMiddle_index(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return int(middle - .5)
    else:
        return (int(middle), int(middle-1))


def get_text_positions_vw(x_data, y_data, labels, txt_width, txt_height, x_limits, vert_line, line_max,
                          axis, y_total_min, molfontize, moleculefont, color_label,
                          highlight_lines, fontweight, textplot, vlcolor):
    
    if type(x_data) != list:
        x_data = x_data.to_list()
    diffx_groups, diff_ind = group_by_difference(x_data, diff=txt_width * 1.0)
    diffy_groups = []
    labels_groups = []
    colors_groups = []
    for s,subgroup in enumerate(diff_ind):
        diffy_groups.append([])
        labels_groups.append([])
        colors_groups.append([])
        for index in subgroup:
            diffy_groups[s].append(y_data[index])
            labels_groups[s].append(labels[index])
            colors_groups[s].append(color_label[index])
    subgroups_lists =  list(zip(diffx_groups, diffy_groups, labels_groups, colors_groups))
    textpos_groups = []
    xtext = []
    ytext = []
    ltext = []
    ctext = []
    for s, (x_sub, y_sub, l_sub, c_sub) in enumerate(subgroups_lists):
        
        textpos_groups.append([])
        if len(x_sub)==1:
            for i, (x,y,l,c) in enumerate(list(zip(x_sub, y_sub, l_sub, c_sub))):
                line_col = c
                textpos_groups[s].append([x, y, l])
                xtext.append(x)
                ytext.append(y)
                ltext.append(l)
                ctext.append(c)
                if textplot:
                    axis.vlines(x, ymin=y_total_min, ymax=y*vert_line, color=vlcolor, linestyle='--', lw=1.0)
                    axis.plot([x, x], [y*vert_line, y*line_max], color='k', linestyle='--', lw=1.0)
                    axis.text(x,  y, l, ha='center', va='bottom',
                                 rotation='vertical', backgroundcolor='none', fontsize=molfontize,
                                 fontname=moleculefont, color=line_col, fontweight=fontweight)
        else:
            multi = 0.15
            if len(x_sub)==2:
                xsepmult = multi
                subdiff = np.abs(x_sub[0]-x_sub[-1])
                xlolim = x_sub[0]-txt_width*xsepmult
                xuplim = x_sub[-1]+txt_width*xsepmult
                if xlolim <= x_limits[0]:
                    xlolim = x_limits[0]*1.01
                if xuplim >= x_limits[1]:
                    xuplim = x_limits[1]*0.99
                lsp_sub = [xlolim, xuplim]
                # Forcing down $v5=1/v7=3 when close to v=0
                if l_sub == ['$v$=0', '$v_5$=1/$v_7$=3$\\,(-1,0)$']:
                    y_sub[1] = y_sub[1]*0.75
                ysp_sub = y_sub
            else:
                xsepmult = len(x_sub)*multi*(multi*2)
                xlolim = x_sub[0]-txt_width*xsepmult
                xuplim = x_sub[-1]+txt_width*xsepmult
                if xlolim <= x_limits[0]:
                    xlolim = x_limits[0]*1.03
                if xuplim >= x_limits[1]:
                    xuplim = x_limits[1]*0.97
                lsp_sub = np.sort(np.linspace(xlolim, xuplim, len(x_sub), endpoint=True))
                if len(y_sub)==11:
                    ysp_sub = [np.nanmean(y_sub)*1.01]*len(y_sub)
                else:
                    ysp_sub = [np.nanmin(y_sub)*1.01]*len(y_sub)
            for i, (x,lsp,y,l,c) in enumerate(list(zip(x_sub, lsp_sub, ysp_sub, l_sub, c_sub))):
                #if bool(highlight_lines):
                #    line_col = [highlight_lines[key] if key in l else color_label for key in highlight_lines][0]
                #else:
                #    line_col = color_label
                line_col = c
                textpos_groups[s].append([lsp, y, l])
                xtext.append(x)
                ytext.append(y)
                ltext.append(l)
                ctext.append(c)
                if textplot:
                    
                    
                    axis.vlines(x, ymin=y_total_min, ymax=y*vert_line, color=vlcolor, linestyle='--', lw=1.0)
                    axis.plot([x, lsp], [y*vert_line, y*line_max], color='k', linestyle='--', lw=1.0)
                    axis.text(lsp,  y, l, ha='center', va='bottom',
                                 rotation='vertical', backgroundcolor='none', fontsize=molfontize,
                                 fontname=moleculefont, color=line_col, fontweight=fontweight)
    return xtext, ytext, ltext


            
            
def line_annotation(all_species, data_files, m, n, spec_all_files, spec_sel_files, panel_names,
                    y_total_min, y_total_max, y_local_limits, y_limits, x_limits, axis_ind, y_max_list,
                    axis, molfontize, moleculefont,
                    highlight_lines, erase_molecule, erase_line, vel_shift, label_color, 
                    panel_spec_df_list, xsep, new_labelling,
                    move_lines, plot_as_one, txtypercen, txtxpercen, frequencies, molec_info):

    vert_line = 0.66
    line_max  = 0.95
    color_label = 'k'
    fontweight= 'bold'
    
    line_labels = highlight_lines.keys()
    line_colors = highlight_lines.values()
    
    move_lines_df = pd.DataFrame(move_lines)
    if len(move_lines_df):
        move_lines_df['old_label'] = deepcopy(move_lines_df['label'])
        move_lines_df.drop(['label'], axis=1, inplace = True)
    erase_lines_df = pd.DataFrame(erase_line)
    if all_species:
        sel_specs = spec_all_files
    else:
        sel_specs = spec_sel_files
    # Annotating line labels
    all_ytext = []
    all_xtext = []
    all_ltext = []
    for j in range(m*n):
        x_limits_spec = x_limits[j][0]
        xlen = x_limits_spec[1]-x_limits_spec[0]
        file_ind = x_limits[j][1]
        print('Panel: ' +str(j+1))
        # Reading asciis from SLIM
        data = pd.read_csv(data_files[file_ind], delim_whitespace= True, header=None)
        data_ncolumns = data.shape[1]
        data_cols = ['vel', 'int']
        # Appending if there is more than one fit available
        if data_ncolumns > 2:
            for f in range(data_ncolumns-2):
                data_cols.append('fit_'+str(f+1))
        data.columns = data_cols
        # Label position at 80% of panel height
        if y_limits != {}:
            y_label_pos_all = y_total_max
            y_label_pos_min = y_total_min
            yaxis_len = np.abs(y_local_limits[j][1]-y_local_limits[j][0])
            #yaxis_len = np.abs(y_label_pos_min-y_label_pos_all)
        else:
            #y_limits = [y_total_min, y_max_list[j]+y_max_list[j]*0.4]
            y_label_pos_all = y_max_list[j]
            y_label_pos_min = y_min_list[j]
            yaxis_len = np.abs(y_local_limits[j][1]-y_local_limits[j][0])
            #yaxis_len = np.abs(y_total_min-y_max_list[j]+y_max_list[j]*0.4)
        yaxis_total_len = y_label_pos_all-y_label_pos_min
        # X axis length
        xaxis_len = np.abs(x_limits_spec[0]-x_limits_spec[1])
        txt_height = txtypercen*(yaxis_len)
        txt_width  = txtxpercen*(xaxis_len)
        spec_filename = sel_specs[file_ind]
        spec = pd.read_csv(spec_filename, delim_whitespace= True, header=None)
        spec.columns = ['vel', 'label', 'transition', 'freq']
        spec = spec.sort_values(['vel'], ascending = True).reset_index(drop=True)
        spec = line_cleaner(spec, xlen)
        spec['old_label'] =  spec['label']
        # Trying to rename molecules
        for molec in new_labelling:
            spec.loc[spec['old_label'] == molec, 'label'] = new_labelling[molec]
        # Not plotting line nor label for molecule in erase_molecule
        spec = spec[~spec['label'].isin(erase_molecule)]
        spec = spec.reset_index(drop=True)
        spec = molec_rename(spec, molec_info)
        for l, line in spec.iterrows():
            if line['Frequency'] == 238053.9019 and line['label']=='HC3N':
                spec.loc[l, 'label'] = r'$v_7$=2$\,(-2)$'
        # Not plotting line nor label for transition in erase_lines
        for er,delrow in erase_lines_df.iterrows():
            panelnum = j+1
            if panelnum == delrow['panel']:
                inddel = np.where((spec['old_label'] == delrow['label']) & (spec['freq'] == delrow['freq']))
                spec.drop(inddel[0], inplace=True)
        spec = spec.reset_index(drop=True)
        # Shifting labels positions (different velocity than spec)
        if frequencies:
            spec['vel'] = vel_to_freq(spec['vel'], vel_shift)
        else:
            spec['vel'] = spec['vel']+vel_shift
        # Dropping different transitions but from the same molecule and same freq. (to avoid too many labels)
        spec = spec.drop_duplicates(subset=['label', 'freq'], keep="first")
        # Dropping lines outside plotted range
        spec = spec[(spec['vel'] >= x_limits_spec[0]) & (spec['vel'] <= x_limits_spec[1])]
        spec = spec.reset_index(drop=True)
        
        yratio = yaxis_total_len/yaxis_len
        # Adding y position based on local intensity
        for l, line in spec.iterrows():
            length_group = len(spec[spec['group'] == line['group']])
            channelsmin = 30
            if length_group*8>=channelsmin:
                if length_group >10:
                    chans = length_group*10
                else:
                    chans = length_group*8
            
            else:
                chans = channelsmin
            if x_limits != {}:
                subdata = data[(data['vel'] >= x_limits_spec[0]) & (data['vel'] <= x_limits_spec[1])]
                subdata = subdata.reset_index(drop=True)
                dfvel_sub = subdata.iloc[(subdata['vel']-line['vel']).abs().argsort()[:chans]]
                dfvel_localmax = subdata.iloc[(subdata['vel']-line['vel']).abs().argsort()[:int(len(subdata)/6)]]
            else:
                dfvel_sub = data.iloc[(data['vel']-line['vel']).abs().argsort()[:chans]]
                dfvel_localmax = data.iloc[(data['vel']-line['vel']).abs().argsort()[:int(len(data)/6)]]
            yprepos = np.nanmax([np.nanmax(dfvel_sub['int']), np.nanmax(dfvel_sub['fit_1'])])
            ylocalmax = np.nanmax([np.nanmax(dfvel_localmax['int']), np.nanmax(dfvel_localmax['fit_1'])])
            if ylocalmax < yaxis_len*yratio*0.33:
                ylocalmax_new = yaxis_len*yratio*0.33
            else:
                ylocalmax_new = ylocalmax
            if yprepos<=0.10*yaxis_len*yratio: # below 10% of axis
                multiplier = 2.0
            elif yprepos<=0.25*yaxis_len*yratio: # below 25% of axis
                multiplier = 1.33
            elif yprepos<=0.45*yaxis_len*yratio: # Below 45% of axis
                multiplier = 1.07
            elif yprepos<=0.66*yaxis_len*yratio: # Below 66% of axis
                multiplier = 1.03
            elif yprepos>=0.66*yaxis_len*yratio: # Above 66% of axis
                multiplier = 1.001
            else:
                multiplier = 1.2

            mult2 = 1
            ypos =  mult2*multiplier*yprepos
            if y_local_limits[j][1] > 0.55*y_label_pos_all:
                if ypos >= ylocalmax:
                    spec.loc[l,'ypos'] = ypos
                else:
                    spec.loc[l,'ypos'] = ylocalmax_new
            else:
                if ylocalmax < 0.25*y_label_pos_all:
                    spec.loc[l,'ypos'] = ylocalmax*yratio*1.0
                if ylocalmax < 0.33*y_label_pos_all:
                    spec.loc[l,'ypos'] = ylocalmax*yratio*0.8
                else:
                    spec.loc[l,'ypos'] = ylocalmax
            
        for l, line in spec.iterrows():
            # Label outside y axis limits
            if line['ypos'] >= y_label_pos_all:
                subylimok = spec[spec['ypos'] <= y_label_pos_all]
                if len(subylimok)>1:
                    # Setting it to same height if there is anyother label below yaxis lim
                    spec.loc[l,'ypos'] = np.nanmin(subylimok['ypos'])*2.15
                else:
                    # If not, setting it to 65% of y axis lim
                    spec.loc[l,'ypos'] = 0.65*y_label_pos_all
        # Separating manually moved lines
        if len(move_lines_df):
            spec_sub_linedf = pd.merge(spec, move_lines_df, on=['old_label','freq'], how='left', indicator='manmv')
            for i, row in spec_sub_linedf.iterrows():
                panelnum = j+1
                if (panelnum != row['panel']) and (row['manmv']=='both'): # Line is repeated in another panel than indicated
                    spec_sub_linedf.loc[i,'manmv'] = 'left_only'
            spec_manlinedf = spec_sub_linedf[spec_sub_linedf['manmv']=='both']
            spec_manlinedf.reset_index(inplace=True)
            spec = spec_sub_linedf[spec_sub_linedf['manmv'] == 'left_only']
            spec.reset_index(inplace=True)
            if len(spec_manlinedf):
                for i, row in spec_manlinedf.iterrows():
                    if row['yunit'] == 'mJy':
                        mult=1/1000
                    else:
                        mult=1
                    if bool(highlight_lines):
                        line_col = [highlight_lines[key] if key in row['label'] else color_label for key in highlight_lines][0] 
                    else:
                        line_col = color_label
                        
#                    if ('HC$_3$N' in row['label']) & (j<3):
#                        vline_col = 'g'
#                    elif ('HC$_3$N' in row['label']) & (j>=3):
#                        vline_col = 'b'
#                    else:
#                        vline_col = 'k'
                    axis[j].vlines(row['vel'], ymin=y_total_min, ymax=mult*row['new_y']*vert_line, color=row['vlcolor'] , linestyle='--', lw=1.0)
                    axis[j].plot([row['vel'], row['new_x']], [mult*row['new_y']*vert_line, mult*row['new_y']*line_max], color='k', linestyle='--', lw=1.0)
                    axis[j].text(row['new_x'],  mult*row['new_y'], row['label'], ha='center', va='bottom',
                                 rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont, color=row['color'], fontweight=fontweight)
               
        
        if plot_as_one[0]:
            spec['rnd_freqGHz'] = np.round(spec['freq']/1e9, plot_as_one[1])
            spec = spec.drop_duplicates(subset=['label', 'rnd_freqGHz'], keep='first')
            spec = spec.reset_index(drop=True)
        xtext, ytext, ltext = get_text_positions_vw(spec['vel'], spec['ypos'], spec['label'], txt_width, txt_height, x_limits_spec,  vert_line, line_max,
                                         axis[j], y_total_min, molfontize, moleculefont, spec['color'], highlight_lines, fontweight, True, spec['vlcolor'])

        #xtext, ytext, ltext = get_text_positions_vw(xtext, ytext, ltext, txt_width, txt_height, x_limits,  vert_line, line_max,
        #                                 axis[j], y_total_min, molfontize, moleculefont, color_label, highlight_lines, fontweight, False)

        #xtext, ytext, ltext = get_text_positions_vw(xtext, ytext, ltext, txt_width, txt_height, x_limits_spec, vert_line, line_max,
        #                                axis[j], y_total_min, molfontize, moleculefont, spec['color'], highlight_lines, fontweight, True)
        all_xtext.append(xtext)
        all_ytext.append(ytext)
        all_ltext.append(ltext)
    return all_xtext, all_ytext, all_ltext



def line_cleaner(spec,xlen):
    xtol = xlen/200
    # Cleans lines of the same species if they are two close, except HC3N
    spec['freq_dif'] = dif_btw_listelemnt(spec['freq']/1e9)
    spec['same_group'] = spec['freq_dif']<xtol
    spec['group'] = spec['same_group'].ne(spec['same_group'].shift()).cumsum()
    for i,row in spec.iterrows():
        if i<len(spec)-1:
            if row['same_group'] == True:# and spec.loc[i+1,'group_ok'] == False:
                spec.loc[i+1,'group'] = row['group']
    return spec
        
def dif_btw_listelemnt(lista):
    fff = [j-i for i, j in zip(lista[:-1], lista[1:])]
    fff.append(1)
    return fff#[j-i for i, j in zip(lista[:-1], lista[1:])]

def lab_btw_listelemnt(lista):
    fff = [i==j for i, j in zip(lista[:-1], lista[1:])]
    fff.append(False)
    return fff

def mean_btw_listelemnt(spec,xtol):
    spec['group'] = spec['same_lab'].ne(spec['same_lab'].shift()).cumsum()
    for i,row in spec.iterrows():
        if i<len(spec)-1:
            if row['same_lab'] == True:# and spec.loc[i+1,'group_ok'] == False:
                spec.loc[i+1,'group'] = row['group']
    sub_spec = spec[spec['label'].str.contains('HC3N')]
    no_spec = spec[~spec['label'].str.contains('HC3N')]
    new_dfs = [sub_spec]
    for g,group in no_spec.groupby('group'):
        if len(group)==1:
            new_dfs.append(group)
        else:
            ndict = {'vel': [np.nanmean(group['vel'])],
                     'label': [group['label'].tolist()[0]],
                     'transition': [group['label'].tolist()[0]],
                     'freq': [np.nanmean(group['freq'])],'freq_dif': [np.nanmean(group['freq_dif'])],
                      'same_lab': [True], 'group': [group['group'].tolist()[0]]
                    }
            new_dfs.append(pd.DataFrame(ndict))
    spec_new = pd.concat(new_dfs, ignore_index=True)
    return spec_new
        
    
def to_raw(string):
    return fr"{string}"

def molec_rename(spec, molec_info):
    hc3n_info = molec_info+'SLIM_HC3N_info'
    spec['Frequency'] = spec['freq']/1e6
    spec['Frequency_rnd'] = np.round(spec['Frequency'],4)
    spec['Formula'] = spec['label']
    spec['Jup'] = '0'
    spec['color'] = '0.7'#'0.55'
    spec['vlcolor'] = 'k'
    for l, trans in spec.iterrows():
        # Renaming HC3N lines properly
        ############### Erase if there is no need to rename lines  ##############################
        hc3ninfo_df = pd.read_csv(hc3n_info+'.csv', sep=',',header=0)# skiprows=0, header=1)
        hc3ninfo_df['Frequency_rnd'] = np.round(hc3ninfo_df['Frequency'],4)
        if 'HC3N' in trans['label']:
            formula_df = hc3ninfo_df[hc3ninfo_df['Frequency_rnd'] == trans['Frequency_rnd']]
            formula_df.reset_index(inplace=True)
            if len(formula_df)==1:
                spec.loc[l,'color'] = 'k'
                form = formula_df['Formula'][0].replace('HC3N', 'HC$_3$N')
                spec.loc[l,'Formula'] = form
                spec.loc[l,'Jup'] = str(int(formula_df['qn1'][0]))
                form = formula_df['Formula'][0].replace('HC3N,', '') # Removing molec name to include quant numb
                if spec.loc[l,'Jup'] == '24':
                    spec.loc[l,'vlcolor'] = 'k'
                elif spec.loc[l,'Jup'] == '26':
                    spec.loc[l,'vlcolor'] = 'k'
                if 'v=0' not in form:
                    form = form.replace('v', '$v_')
                    form = form.replace('=', '$=')
                else:
                    form = form.replace('v', '$v$')
                #form = form +',('+ str(int(formula_df['qn1'][0]))
                if ~np.isnan(formula_df['qn2'][0]):
                    form = form +'$\,('+ str(int(formula_df['qn2'][0]))
                if ~np.isnan(formula_df['qn3'][0]):
                    form = form +','+ str(int(formula_df['qn3'][0]))
                if '(' in form:
                    form = form+')$'
                if '^' in form:
                    spec.loc[l,'label'] = form.replace('^0', '$^{0}$')
                else:
                    spec.loc[l,'label'] = form
            elif len(formula_df)>1:
                print('WARNING!!! more than one line with same freq')
        else:
            newlabel = to_raw(trans['label'])
            newlabel_mol = newlabel.split(',')[0]
            if len(newlabel.split(','))>1:
                newlabel_vib = newlabel.split(',')[1]
                newlabel_vib = '$'+newlabel_vib+'$'
                newlabel_vib = ','+re.sub('(?<=v)+([\d])+(?==)', r'_\1', newlabel_vib)
            else:
                newlabel_vib = ''
            newlabel_mol = re.sub('(?<=-)+([\d]?[\d])+(?=-)', r'$^{\1}$', newlabel_mol)
            newlabel_mol = re.sub('(?<=[a-zA-z]-)+([\d]?[\d])', r'$^{\1}$', newlabel_mol)
            newlabel_mol = newlabel_mol.replace('-', '')
            newlabel_mol = newlabel_mol.replace('+', '$^{+}$')
            newlabel_mol = re.sub('([\d])+(?![\d}])', r'$_\1$', newlabel_mol)+newlabel_vib
            
            if '$^' in newlabel_mol:
                result = re.search(r'(.+)(\$\^.*?\$)(.+|$)', newlabel_mol)
                newlabel_mol = ''
                for gr in result.groups():
                    if '$^' in gr:
                        newlabel_mol = newlabel_mol[:-1] + gr + newlabel_mol[-1]
                    else:
                        newlabel_mol = newlabel_mol + gr
            
            spec.loc[l, 'label'] = newlabel_mol
    return spec
    
class Grouper:
    """simple class to perform comparison when called, storing last element given"""
    def __init__(self, diff):
        self.last = None
        self.diff = diff
    def predicate(self, item):
        if self.last is None:
            return True
        return abs(self.last - item) < self.diff
    def __call__(self, item):
        """called with each item by takewhile"""
        result = self.predicate(item)
        self.last = item
        return result

def vel_to_freq(restfreq, vel):
    """
    velocity in km/s to frequency
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    freq = restfreq*(1.-vel/c_si.value)
    return freq

def group_by_difference(items, diff=50):
    results = []
    start = 0
    remaining_items = items
    while remaining_items:
        g = Grouper(diff)
        group = [*itertools.takewhile(g, remaining_items)]
        results.append(group)
        start += len(group)
        remaining_items = items[start:]
    
    results_index = []
    for s,subgroup in enumerate(results):
        results_index.append([])
        for element in subgroup:
            indices = [i for i, x in enumerate(items) if x == element]
            results_index[s].append(indices[0])
    return results, results_index