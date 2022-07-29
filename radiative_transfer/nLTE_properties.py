# =============================================================================
# Modelling Data
# =============================================================================

from astrothesispy.utiles import u_conversion
from astrothesispy.utiles import utiles


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

import numpy as np
import os
from math import atan2,degrees
import scipy
import pandas as pd
import astropy.units as u
import inspect
from scipy import interpolate
import collections
from tqdm import tqdm

# Min size 0.039" 
# tam_min_pc = 0.039*17/2
# factor_tam_min = tam_min_pc**2

# # Max size 0.1" 
# tam_max_pc = 0.1*17/2
# factor_tam_max = tam_max_pc**2
        
# # Observed values SHC14
# hc3n_v0_24_23_obs = 54.08 #mJy/beam
# hc3n_v0_39_38_obs = 70.29 #mJy
# hc3n_v0_ratio_obs = hc3n_v0_39_38_obs/hc3n_v0_24_23_obs

# hc3n_v7_24_23_obs = 22.05 #mJy
# hc3n_v7_39_38_obs = 44.59 #mJy
# hc3n_v7_ratio_obs = hc3n_v7_39_38_obs/hc3n_v7_24_23_obs
# luminosity_obs = 2.928e9 # lsun



class data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def line_finder(lambda_mu, model_prof, unit='mJy'):
    """
    Returns the flux peak in mJy, the integrated flux in mJy km/s
    and the entire line profile from the nLTE models
    model_prof = model_name+'_.prof'
    """
    tqdm.write('\tSearching line at {0:1.2f} microns'.format(lambda_mu))
    profile_lim = []
    for i, row in model_prof.iterrows():
        if row['V_kms'] == 0 and abs(row['lambda_um']-lambda_mu)<0.01:#<0.1:
            for j in range(i-1, -1, -1):
                if model_prof['V_kms'][j]>0:
                    profile_lim.append(j+2)
                    break
            for k in range(i+1, model_prof.shape[0]):
                if model_prof['V_kms'][k] < 0:
                    profile_lim.append(k)
                    break
    profile = model_prof.loc[profile_lim[0]:profile_lim[1]]
    profile.reset_index(drop=True, inplace=True)
    peak = profile['I_Ic_mJy'][profile.V_kms == 0].max()
    int_flux = 0.0 # mJy km/s
    for i, row in profile.iterrows():
        if i < len(profile)-1:
            int_flux = int_flux + profile.loc[i, 'I_Ic_mJy']*(profile.loc[i+1, 'V_kms'] - profile.loc[i, 'V_kms'])
    if unit=='mJy':
        tqdm.write('\t\tIpeak='+'%1.3f' %peak+'mJy'+'\tI='+'%1.3f' % int_flux+'mJy km/s')
    if unit=='uJy':
        tqdm.write(f'\t\tIpeak={peak*1000:1.3e}uJy\tI={int_flux*1000:1.3e}uJy km/s')
    return profile, peak, int_flux

def profile_plotter(transitions_df, Jplot, filas = 2, columnas = 1,
                        colors = ['k', 'r', 'b'], factor=1,
                        linewidth = 0.7, labelsize = 12, velocity_limits=[-100, 100],
                        intensity_limits=['NA', 'NA'], first_last_xlabel = False):
    """
    Jplot list with J-J-1 transitions to plot
    """
    tqdm.write('\tPlotting line profiles')
    # Plotting line profiles
    label_y =  r"Flux density (mJy)" 
    label_x =  r"V (km/s)"
    m = filas    # rows
    n = columnas # columns
    dx, dy = 1.2, 1
    figsize = plt.figaspect(float(dy * m) / float(dx * n))
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    # Generating specified number of axis
    axis = []
    axis_ind = []
    for i in range(m*n):
        axis.append(fig.add_subplot(gs1[i]))
    # Generating axis index
    ind = 0
    axis_ind = []
    for i  in range(m):
        axis_ind.append([])
        for j in range(n):
            axis_ind[i].append(ind)
            ind += 1
    for v,vib in enumerate(['v=0', 'v7=1', 'v6=1']):
        y_st = 0.85
        for j, trans in enumerate(Jplot):
            # Plotting spectrum
            if vib == 'v=0':
                par = '1f'
            elif vib == 'v7=1':
                # v7=1 Jup=even  1e lines are blended with v6=1 1f
                # v7=1 Jup=odd  1f lines are blended with v6=1 1e
                if np.float(trans.split('-')[0])%2 == 0: # 
                    par = '1f'
                elif np.float(trans.split('-')[0]) == 39:
                    # only sim line
                     par = '1f'
                else:
                    par = '1e'
            elif vib == 'v6=1':
                if np.float(trans.split('-')[0])%2 == 0: # 
                    par = '1e'
                elif np.float(trans.split('-')[0]) == 39:
                    # only sim line
                    par = '1e'
                else:
                    par = '1f'
            v_plot = transitions_df.loc[(transitions_df['vib'] == vib)  & (transitions_df['J'] == trans) & (transitions_df['parity'] == par)]
            axis[v].plot(v_plot.profile.values[0]['V_kms'], v_plot.profile.values[0]['I_Ic_mJy'], linewidth=linewidth, color=colors[j])
            axis[v].text(0.1, y_st,
                        trans, ha='left', va='center',
                        transform=axis[v].transAxes, color = colors[j],
                        rotation='horizontal', fontsize=8)
            y_st = y_st - 0.05
        axis[v].text(0.1, 0.9,
                    r'HC$_3$N '+vib, ha='left', va='center',
                    transform=axis[v].transAxes,
                    rotation='horizontal', fontsize=8)

    # Left Figures
    left_ind = []
    for i in range(m):
        left_ind.append(axis_ind[i][0])
    # Max_value 
    all_profiles = pd.concat(transitions_df.profile.values, ignore_index=True)
    maxval = np.nanmax(all_profiles['I_Ic_mJy'])
    # Bottom Figures
    bottom_figures = axis_ind[-1]
    intensity_limits_total = [-0.5, maxval]
    for i, axw in enumerate(axis):
        axw.minorticks_on()
        axw.tick_params(axis='both', which='major', labelsize=labelsize, length=5, direction='in')
        axw.tick_params(axis='both', which='minor', labelsize=labelsize, direction='in')
        axw.xaxis.set_tick_params(top =True, which='both')
        axw.yaxis.set_tick_params(right=True, which='both', labelright=False)
        # Limite en el eje x
        if velocity_limits != ['NA', 'NA']: 
            axw.set_xlim(velocity_limits)
        # Limite en el eje y
        if intensity_limits != ['NA', 'NA']:
            if isinstance(intensity_limits[0], list):
                axw.set_ylim(intensity_limits[i])
            else:
                axw.set_ylim(intensity_limits)
        else:
            axw.set_ylim([intensity_limits_total[0], intensity_limits_total[1]+intensity_limits_total[1]*0.4]) 
        if i in left_ind:
            axw.set_ylabel(label_y, fontsize=14)
        else:
            axw.set_yticklabels([])
        if i in bottom_figures:
            axw.set_xlabel(label_x, fontsize=14)
            if first_last_xlabel == False:
                plt.setp(axw.get_xticklabels()[0], visible=False)    
                plt.setp(axw.get_xticklabels()[-1], visible=False)
        else:
            axw.set_xticklabels([])
    return fig, axis

def get_tau100(model_path, model_name):
    """
        Gets tau100 from the molecule models
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at 100 um
    tau100 = model_taudust['taudust'].loc[model_taudust['lambda_um_u'] == 100.0]
    return tau100.values[0]

def get_tau100_dustmod(model_path, model_name):
    """
        Gets tau100 from the dust models
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.tau', delim_whitespace= True, header=None)
    model_taudust.columns = ['lambda_um', 'taudust', 'otro']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at 100 um
    tau100 = model_taudust.iloc[(model_taudust['lambda_um']-100.0).abs().argsort()[:1]]
    tau100.reset_index(inplace=True, drop=True)
    return tau100.values[0]

def get_taulambda(model_path, model_name, lambda_um):
    """
        Gets tau for the specified wavelength lambda in microns
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at specified wavelength in um
    taulambda = model_taudust['taudust'].loc[np.abs(model_taudust['lambda_um']-lambda_um)  < 0.1]
    # Mean value if more than one tau is found
    tau = taulambda.mean()
    return tau


def SED_plotter(model_iso, model_spire, model_taudust, distance_pc, luminosity_total):
    """
        Plots the model Spectral Energy Distribution and SEDs from other papers
    """
    tqdm.write('\tPlotting SED and continuum opacity')
    # Observational SED from perez-beaupuits 2018
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    obs_SED_path = script_dir+'/data/NGC253/'
    obs_SED = pd.read_csv(obs_SED_path+'NGC253_SED.txt', delim_whitespace = True, comment='#')
    SED_model_pars =pd.read_csv(obs_SED_path+'NGC253_SED_fit_parameters.txt', delim_whitespace = True, comment='#')
    modeled_comp_cold =[]
    modeled_comp_warm =[]
    modeled_comp_hot =[]
    modeled_comp_all = []
    wave_mod = range(10, 1200+1)
    D = distance_pc*(1*u.pc).to(u.Mpc).value
    for i, wave in enumerate(wave_mod):
        # Models from Perez-Beaupuits 2018
        nu = (wave*u.um).to(u.GHz, equivalencies=u.spectral())
        df_cold = SED_model_pars.loc[SED_model_pars['Component'] == 'Cold']
        df_warm = SED_model_pars.loc[SED_model_pars['Component'] == 'Warm']
        df_hot = SED_model_pars.loc[SED_model_pars['Component'] == 'Hot']
        modeled_cold = utiles.SED_model(nu, df_cold['Tdust'][0], df_cold['Mdust_1e6'][0]*1e6, df_cold['filling_fact'][0], D)
        modeled_warm = utiles.SED_model(nu, df_warm['Tdust'][1], df_warm['Mdust_1e6'][1]*1e6, df_warm['filling_fact'][1], D)
        modeled_hot = utiles.SED_model(nu, df_hot['Tdust'][2], df_hot['Mdust_1e6'][2]*1e6, df_hot['filling_fact'][2], D)
        modeled_comp_cold.append(modeled_cold.value)
        modeled_comp_warm.append(modeled_warm.value)
        modeled_comp_hot.append(modeled_hot.value)
        modeled_comp_all.append(modeled_cold.value+modeled_warm.value+modeled_hot.value)
    m = 2
    n = 1
    fig = plt.figure()
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    # Generating specified number of axis
    axis = []
    for i in range(m*n):
        axis.append(fig.add_subplot(gs1[i]))
    for j in range(m*n):
        # SED
        if j == 0:
            # SED
            axis[j].plot(model_iso['lambda_um'], model_iso['Ic_Jy'], linewidth=0.7, color='k')
            axis[j].plot(model_spire['lambda_um'], model_spire['Ic_Jy'], linewidth=0.7, color='k')
            # Observed SED
            axis[j].errorbar(obs_SED['Wavlength'], obs_SED['Obs_flux'],
                            yerr=obs_SED['Obs_flux_err'], fmt='.',
                            elinewidth=0.7, capsize=0.5,
                            linewidth=0.7, color='k')
            # Modeled SED
            axis[j].plot(wave_mod, modeled_comp_cold, linewidth=0.7, color='blue')
            axis[j].plot(wave_mod, modeled_comp_warm, linewidth=0.7, color='green')
            axis[j].plot(wave_mod, modeled_comp_hot, linewidth=0.7, color='red')
            axis[j].plot(wave_mod, modeled_comp_all, linewidth=0.7, color='k', linestyle='--')
        # Optical depth
        elif j==1:
            axis[j].plot(model_taudust['lambda_um'], model_taudust['taudust'], linewidth=0.7, color='k')
            # Fit opt depth from SED
            axis[j].errorbar(obs_SED['Wavlength'], obs_SED['opt_depth'],
                            yerr=obs_SED['Obs_flux_err'], fmt='.',
                            elinewidth=0.7, capsize=0.5,
                            linewidth=0.7, color='k')
    for i, ax in enumerate(axis):
        majorLocator = MultipleLocator(20)
        ax.yaxis.set_major_locator(majorLocator)
        # Solo ponemos titulo a los ejes y si estan a la izquierda del todo
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12, length=5, direction='in')
        ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')
        ax.xaxis.set_tick_params(top =True, which='both')
        ax.yaxis.set_tick_params(right=True, which='both', labelright=False)
        ax.set_xlim([10,1200])
        ax.set_yscale('log')
        ax.set_xscale('log')
    axis[0].set_ylim([1e-2,1e4])
    axis[0].set_xticklabels([])
    axis[0].set_ylabel(r'Flux density (Jy)', fontsize=14)
    axis[1].set_ylabel(r'Optical depth $\tau$', fontsize=14)
    axis[1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    axis[0].text(0.1, 0.9,
                        'L='+ '% 1.2E' % luminosity_total + r' L$_\odot$', ha='left', va='center',
                        transform=axis[j].transAxes,
                        rotation='horizontal', fontsize=8)
    return fig, axis

def SED_plotter_only_mod(model_iso, model_spire, model_taudust, distance_pc, luminosity_total):
    """
        Plots the model Spectral Energy Distribution
    """
    print('\tPlotting SED and continuum opacity')
    # Observational SED from perez-beaupuits 2018
    script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    m = 2
    n = 1
    fig = plt.figure()
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    # Generating specified number of axis
    axis = []
    for i in range(m*n):
        axis.append(fig.add_subplot(gs1[i]))
    for j in range(m*n):
        # SED
        if j == 0:
            # SED
            axis[j].plot(model_iso['lambda_um'], model_iso['Ic_Jy'], linewidth=0.7, color='k')
            axis[j].plot(model_spire['lambda_um'], model_spire['Ic_Jy'], linewidth=0.7, color='k')
        # Optical depth
        elif j==1:
            axis[j].plot(model_taudust['lambda_um'], model_taudust['taudust'], linewidth=0.7, color='k')
    for i, ax in enumerate(axis):
        majorLocator = MultipleLocator(20)
        ax.yaxis.set_major_locator(majorLocator)
        # Solo ponemos titulo a los ejes y si estan a la izquierda del todo
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=12, length=5, direction='in')
        ax.tick_params(axis='both', which='minor', labelsize=12, direction='in')
        ax.xaxis.set_tick_params(top =True, which='both')
        ax.yaxis.set_tick_params(right=True, which='both', labelright=False)
        #ax.set_xlim([10,1200])
        ax.set_yscale('log')
        ax.set_xscale('log')
    #axis[0].set_ylim([1e-2,1e4])
    axis[0].set_xticklabels([])
    axis[0].set_ylabel(r'Flux density (Jy)', fontsize=14)
    axis[1].set_ylabel(r'Optical depth $\tau$', fontsize=14)
    axis[1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    
    axis[0].text(0.05, 0.9,
                        'L='+ '% 1.2E' % luminosity_total + r' L$_\odot$', ha='left', va='center',
                        transform=axis[0].transAxes,
                        rotation='horizontal', fontsize=8)
    return fig, axis

def tau_profiles(model_tau, v6, linewidth=0.7,
                 v0_color = 'k', v7_color='r', v6_color='b'):
    """
        Plots the tau profiles for the line transitions
    """
    tqdm.write('\tPlotting line opacities')
    m = 2    # rows
    n = 1    # columns
    dx, dy = 1.2, 1
    figsize = plt.figaspect(float(dy * m) / float(dx * n))
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    # Generating specified number of axis
    axis = []
    for i in range(m*n):
        axis.append(fig.add_subplot(gs1[i]))
        if i == 0:
            # Rotational Lines
            label_y = r'$\tau_{\rm{line}}$ ROT'
            # v0 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v0_24_23'], linewidth=linewidth, color=v0_color)
            # v0 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v0_39_38'], linewidth=linewidth, linestyle='--', color=v0_color)
            # v7 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v7_24_23'], linewidth=linewidth, color=v7_color)
            # v7 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v7_39_38'], linewidth=linewidth, linestyle='--', color=v7_color)
            if v6==True or v6 == 'nograd':
                # v6 24-23
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v6_24_23'], linewidth=linewidth, color=v6_color)
                # v6 39-38
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v6_39_38'], linewidth=linewidth, linestyle='--', color=v6_color)
            
            # Not xticks nor xlabel to top subfigure
            axis[i].set_xticklabels([])
            axis[i].set_ylabel(label_y, fontsize=14)
            axis[i].set_yscale('log')
        elif i==1:
            # Vibrational Lines
            label_y = r'$\tau_{\rm{line}}$ VIB'
            label_x = r'r/R'
             # v7 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v7_24_23'], linewidth=linewidth, color=v7_color)
            # v7 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v7_39_38'], linewidth=linewidth, linestyle='--', color=v7_color)
            if v6==True or v6 == 'nograd': 
                # v6 24-23
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v6_24_23'], linewidth=linewidth, color=v6_color)
                # v6 39-38
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v6_39_38'], linewidth=linewidth, linestyle='--', color=v6_color)
            axis[i].set_yscale('log')
            axis[i].set_ylabel(label_y, fontsize=14)
            axis[i].set_xlabel(label_x, fontsize=14)
    return fig, axis


def tau_profiles_new(model_tau, v6, linewidth=0.7,
                 v0_color = 'k', v7_color='r', v6_color='b'):
    """
        Plots the tau profiles for the line transitions
    """
    tqdm.write('\tPlotting line opacities')
    m = 2    # rows
    n = 1    # columns
    dx, dy = 1.2, 1
    figsize = plt.figaspect(float(dy * m) / float(dx * n))
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    # Generating specified number of axis
    axis = []
    for i in range(m*n):
        axis.append(fig.add_subplot(gs1[i]))
        if i == 0:
            # Rotational Lines
            label_y = r'$\tau_{\rm{line}}$ ROT'
            # v0 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v0_24_23'], linewidth=linewidth, color=v0_color)
            # v0 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v0_39_38'], linewidth=linewidth, linestyle='--', color=v0_color)
            # v7 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v7_24_23'], linewidth=linewidth, color=v7_color)
            # v7 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v7_39_38'], linewidth=linewidth, linestyle='--', color=v7_color)
            if v6==True or v6 == 'nograd':
                # v6 24-23
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v6_24_23'], linewidth=linewidth, color=v6_color)
                # v6 39-38
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_v6_39_38'], linewidth=linewidth, linestyle='--', color=v6_color)
            
            # Not xticks nor xlabel to top subfigure
            axis[i].set_xticklabels([])
            axis[i].set_ylabel(label_y, fontsize=14)
            axis[i].set_yscale('log')
        elif i==1:
            # Vibrational Lines
            label_y = r'$\tau_{\rm{line}}$ VIB'
            label_x = r'r/R'
             # v7 24-23
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v7_24_23'], linewidth=linewidth, color=v7_color)
            # v7 39-38
            axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v7_39_38'], linewidth=linewidth, linestyle='--', color=v7_color)
            if v6==True or v6 == 'nograd': 
                # v6 24-23
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v6_24_23'], linewidth=linewidth, color=v6_color)
                # v6 39-38
                axis[i].plot(model_tau['r/Rtot'], model_tau['tau_vib_v0_v6_39_38'], linewidth=linewidth, linestyle='--', color=v6_color)
            axis[i].set_yscale('log')
            axis[i].set_ylabel(label_y, fontsize=14)
            axis[i].set_xlabel(label_x, fontsize=14)
    return fig, axis
        

def only_luminosity_cal(distance_pc, model_path, model_name):
    """
        Calculates the model luminosity
    """
    factor = 1 # No more factor to change size
    # =============================================================================
    # SED and tau_continuum
    # =============================================================================
    ## ISO
    model_iso = pd.read_csv(model_path+'/'+model_name+'_.iso', delim_whitespace= True)
    model_iso.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_iso['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_iso['Ic_ergscm2um'], model_iso['lambda_um'])/1000
    # Multiplying intensities by factor
    model_iso['Ic_Jy'] = model_iso['Ic_Jy']*(factor**2.)
    model_iso['Ic_ergscm2um'] = model_iso['Ic_ergscm2um']*(factor**2.)
    ## Spire
    model_spire = pd.read_csv(model_path+'/'+model_name+'_.spire', delim_whitespace= True)
    model_spire.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_spire['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_spire['Ic_ergscm2um'], model_spire['lambda_um'])/1000
    # Multiplying intensities by factor
    model_spire['Ic_Jy'] = model_spire['Ic_Jy']*(factor**2.)
    model_spire['Ic_ergscm2um'] = model_spire['Ic_ergscm2um']*(factor**2.)
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # =============================================================================
    # Luminosity 10-1200um
    # =============================================================================
    tqdm.write('\tCalculating luminosities')
    luminosity_iso = 0
    for i, row in model_iso.iterrows():
        if i == 0:
            continue
        else:
            luminosity_iso += (model_iso['Ic_ergscm2um'][i]*
                              (model_iso['lambda_um'][i]-model_iso['lambda_um'][i-1])*
                              4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
    luminosity_spire = 0
    for i, row in model_spire.iterrows():
        if i == 0:
            continue
        else:
            luminosity_spire += (model_spire['Ic_ergscm2um'][i]*
                              (model_spire['lambda_um'][i]-model_spire['lambda_um'][i-1])*
                              4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
        
    luminosity_total = luminosity_iso+luminosity_spire # lsun
    tqdm.write('\t\tL='+'%1.2E' % luminosity_total+' Lsun')
    return luminosity_total

def only_luminosity_cal2(distance_pc, model_path, model_name, freq_Hz_list, tol = 0.005):
    """
        Calculates the model luminosity
    """
    factor = 1 # No more factor to change size
    # =============================================================================
    # SED and tau_continuum
    # =============================================================================
    ## ISO
    model_iso = pd.read_csv(model_path+'/'+model_name+'_.iso', delim_whitespace= True)
    model_iso.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_iso['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_iso['Ic_ergscm2um'], model_iso['lambda_um'])/1000
    # Multiplying intensities by factor
    model_iso['Ic_Jy'] = model_iso['Ic_Jy']*(factor**2.)
    model_iso['Ic_ergscm2um'] = model_iso['Ic_ergscm2um']*(factor**2.)
    ## Spire
    model_spire = pd.read_csv(model_path+'/'+model_name+'_.spire', delim_whitespace= True)
    model_spire.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_spire['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_spire['Ic_ergscm2um'], model_spire['lambda_um'])/1000
    # Multiplying intensities by factor
    model_spire['Ic_Jy'] = model_spire['Ic_Jy']*(factor**2.)
    model_spire['Ic_ergscm2um'] = model_spire['Ic_ergscm2um']*(factor**2.)
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # =============================================================================
    # Luminosity 10-1200um
    # =============================================================================
    tqdm.write('\tCalculating luminosities')
    luminosity_iso = 0
    for i, row in model_iso.iterrows():
        if i == 0:
            continue
        else:
            luminosity_iso += (model_iso['Ic_ergscm2um'][i]*
                              (model_iso['lambda_um'][i]-model_iso['lambda_um'][i-1])*
                              4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
    luminosity_spire = 0
    for i, row in model_spire.iterrows():
        if i == 0:
            continue
        else:
            luminosity_spire += (model_spire['Ic_ergscm2um'][i]*
                              (model_spire['lambda_um'][i]-model_spire['lambda_um'][i-1])*
                              4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
        
    luminosity_total = luminosity_iso+luminosity_spire # lsun
    tqdm.write('\t\tL='+'%1.2E' % luminosity_total+' Lsun')
    
    freqs_df_dict = {}
    for freq_Hz in freq_Hz_list:
        freq_microns = (np.array(freq_Hz)*u.Hz).to(u.um, equivalencies=u.spectral())
    
        a = model_spire.loc[(model_spire['lambda_um'] >= freq_microns*(1-tol)) & (model_spire['lambda_um'] <= freq_microns*(1+tol))]
        a.reset_index(inplace=True)
        freqs_df_dict[str(int(freq_Hz/1e9))] = a
    return luminosity_total, freqs_df_dict
    
        
def luminosity_calculator(calculate_luminosity, distance_pc, model_path, model_name, v6, 
                          inter_ratios, 
                          plot_profiles, plot_SED, figoutdir, linesinfo, factor = 1,  velocity_limits=[-100, 100],
                          intensity_limits=['NA', 'NA'], fig_format='png'):
    """
        Calculates the model luminosity, gets the line profiles and plots them, the SED and tau continuum
        v6=True includes v6 in the calculations
        Generally v7=1e is blended v6=1f
        inter_ratios = ['24-23', '16-15'] list with ratios to make and plot
    """
    # =============================================================================
    # # Line profiles
    # =============================================================================
    if plot_profiles == True or plot_SED==True:
        if not os.path.exists(figoutdir):
            os.makedirs(figoutdir)
    # Loading model line profile
    model_prof = pd.read_csv(model_path+'/'+model_name+'_.prof', delim_whitespace= True, comment='!', header=None)#, skiprows=[0])
    model_prof.columns = ['lambda_um', 'I_ergscm2um', 'I_Ic_ergscm2um', 'V_kms']
    # Conversion erg/s/cm^2/um to mJy
    model_prof['I_Ic_mJy'] = u_conversion.ergscmum_to_mjy(model_prof['I_Ic_ergscm2um'], model_prof['lambda_um'])
    # Multiplying intensities by factor^2
    model_prof['I_Ic_mJy'] = model_prof['I_Ic_mJy']*(factor)
    transitions_df = pd.read_csv(linesinfo, delim_whitespace= True, header=0, comment='#')
    transitions_df['alams'] =   (np.array(transitions_df['freq_Hz'])*u.Hz).to(u.um, equivalencies=u.spectral())
    transitions_df['V_kms'] = np.nan
    transitions_df['I_Ic_mJy'] = np.nan
    transitions_df['peak'] = np.nan
    transitions_df['integ_intens'] = np.nan
    transitions_df['parity'] = np.nan
    profiles = []
    for index, row in transitions_df.iterrows():
        print(f'Line: {row.vib} {row.J} {row.par}')
        profile, peak, integ_intens = line_finder(lambda_mu=row['alams'], model_prof=model_prof, unit='mJy') 
        profiles.append(profile)
        transitions_df.loc[index,'peak'] = peak
        transitions_df.loc[index,'integ_intens'] = integ_intens
        if row['lup'] == -1:
            transitions_df.loc[index,'parity'] = '1e'
        else:
            transitions_df.loc[index,'parity'] = '1f'
    transitions_df['profile'] = profiles
    ratios_dict = {}
    vib_states = transitions_df['vib'].unique().tolist()
    # Ratios between same vib. state
    ratios_dict = {}
    # Ratios between same vib. state
    str_ratios  = utiles.str_btw_listelemnt_asc(inter_ratios)
    for vib in vib_states:
        ratios_dict[vib] = {}
        for s, s_ratio in enumerate(str_ratios):
            int_ratio = s_ratio.split('_')
            vibs_df = transitions_df[transitions_df['vib'] == vib]
            # Generally v7=1e is blended v6=1f
            # We use 1f for v7=1 and 1e for v6=1 then
            if vib == 'v6=1':
                par = '1e'
            else:
                par = '1f'
            jup_peak = vibs_df.loc[(vibs_df['J'] == int_ratio[0]) & (vibs_df['parity'] == par), 'peak'].values[0]
            jup_integ = vibs_df.loc[(vibs_df['J'] == int_ratio[0]) & (vibs_df['parity'] == par), 'integ_intens'].values[0]
            jlow_peak = vibs_df.loc[(vibs_df['J'] == int_ratio[1]) & (vibs_df['parity'] == par), 'peak'].values[0]
            jlow_integ = vibs_df.loc[(vibs_df['J'] == int_ratio[1]) & (vibs_df['parity'] == par), 'integ_intens'].values[0]
            ratios_dict[vib]['peak_'+int_ratio[0]+'_'+int_ratio[1]] =  jup_peak/jlow_peak
            ratios_dict[vib]['integ_'+int_ratio[0]+'_'+int_ratio[1]] =  jup_integ/jlow_integ
    # Ratios between ground and v7=1
    ratios_dict['v0_v7'] = {}
    for ratio in inter_ratios:
        v7 = transitions_df.loc[(transitions_df['vib'] == 'v7=1') & (transitions_df['J'] == ratio) & (transitions_df['parity'] == '1f')]
        v0 = transitions_df.loc[(transitions_df['vib'] == 'v=0')  & (transitions_df['J'] == ratio) & (transitions_df['parity'] == '1f')]
        ratios_dict['v0_v7']['peak_'+ratio] = v0.peak.values[0]/v7.peak.values[0]
        ratios_dict['v0_v7']['integ_'+ratio] = v0.integ_intens.values[0]/v7.integ_intens.values[0]

    # =============================================================================
    #  Plotting profiles
    # =============================================================================
    if plot_profiles == True:
        fig, axis = profile_plotter(ratios_dict, transitions_df, Jplot=inter_ratios, filas = 3, columnas = 1,
                                        colors = ['k', 'r', 'b'], factor=1,
                                        linewidth = 0.7, labelsize = 12, velocity_limits=velocity_limits,
                                        intensity_limits=intensity_limits, first_last_xlabel = False)
        
        fig.savefig(figoutdir+'/'+model_name+'_f'+'%1.1f' % factor +'_profile.'+fig_format, bbox_inches='tight', transparent=True, dpi=400, format=fig_format)
        plt.close()
    
    # =============================================================================
    # SED and tau_continuum
    # =============================================================================
    ## ISO
    model_iso = pd.read_csv(model_path+'/'+model_name+'_.iso', delim_whitespace= True)
    model_iso.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_iso['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_iso['Ic_ergscm2um'], model_iso['lambda_um'])/1000
    # Multiplying intensities by factor
    model_iso['Ic_Jy'] = model_iso['Ic_Jy']*(factor**2.)
    model_iso['Ic_ergscm2um'] = model_iso['Ic_ergscm2um']*(factor**2.)
    ## Spire
    model_spire = pd.read_csv(model_path+'/'+model_name+'_.spire', delim_whitespace= True)
    model_spire.columns = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    # Conversion erg/s/cm^2/um to mJy
    model_spire['Ic_Jy'] = u_conversion.ergscmum_to_mjy(model_spire['Ic_ergscm2um'], model_spire['lambda_um'])/1000
    # Multiplying intensities by factor
    model_spire['Ic_Jy'] = model_spire['Ic_Jy']*(factor**2.)
    model_spire['Ic_ergscm2um'] = model_spire['Ic_ergscm2um']*(factor**2.)
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    
    # =============================================================================
    # Luminosity 10-1200um
    # =============================================================================
    if calculate_luminosity== True:
        tqdm.write('\tCalculating luminosities')
        luminosity_iso = 0
        for i, row in model_iso.iterrows():
            if i == 0:
                continue
            else:
                luminosity_iso += (model_iso['Ic_ergscm2um'][i]*
                                  (model_iso['lambda_um'][i]-model_iso['lambda_um'][i-1])*
                                  4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
        luminosity_spire = 0
        for i, row in model_spire.iterrows():
            if i == 0:
                continue
            else:
                luminosity_spire += (model_spire['Ic_ergscm2um'][i]*
                                  (model_spire['lambda_um'][i]-model_spire['lambda_um'][i-1])*
                                  4.*3.14159*(distance_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
            
        luminosity_total = luminosity_iso+luminosity_spire # lsun
        print('\t\tL='+'%1.2E' % luminosity_total+' Lsun')
    else:
        luminosity_total = -1
    # Plotting SED
    if plot_SED == True:
        fig, axis = SED_plotter(model_iso, model_spire, model_taudust, distance_pc, luminosity_total)  
        fig.savefig(figoutdir+'/'+model_name+'_f'+'%1.1f' % factor +'_SED.'+fig_format, bbox_inches='tight', transparent=True, dpi=400, format=fig_format)
        plt.close()
    return luminosity_total, transitions_df, ratios_dict

def labelLine_fit(xvals,yvals,xpos, ypos,ax, color='k', label=None,align=True,**kwargs):
    """
        Labeling lines
    """
    xdata = xvals.tolist()
    ydata = yvals.tolist()
    x = xpos
    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return
    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break
    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])
    if not label:
        label = ''
    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))
        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]
    else:
        trans_angle = 0
    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = color
    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'
    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'
    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()
    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True
    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5
    t = ax.text(x,ypos,label,rotation=trans_angle, fontsize=6, verticalalignment='center', horizontalalignment='center', **kwargs)
    t.set_bbox(dict(facecolor='white', alpha=0.0))
    

def labelLines_fit(xvals, yvals, ax, xpos, ypos, color, label, align=True,**kwargs):
    labelLine_fit(xvals,yvals,xpos,ypos,ax,color,label,align,**kwargs)

def plot_model_nHT_map(models_df, observed_df, results_path, d_slicer,
                       xvar, yvar, zvar, integrated,
                       legend = False
                       ):
    """
        Plots the model gas density in a 2D map
    """
    fig = plt.figure()
    ax = plt.axes()
    x = models_df[xvar]
    y = models_df[yvar]
    z = models_df[zvar]
    # Generating linearspace for maps
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xi, yi = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear', smooth=0.5)
    zi = rbf(xi, yi)
    # Check if all elements in z are the same -> cant make contours
    lev = np.linspace(z.min(), z.max(), 20)
    lev = np.arange(100, 600+50, 50)
    # Plotting Temperature contours and colorbar
    cs = ax.contourf(xi, yi, zi,  cmap="Reds", origin='lower', levels=lev)
    cbar = plt.colorbar(cs)
    # Getting unique values of densities
    unique_dens = models_df.central_nh.unique()
    s_unique_dens = np.sort(unique_dens)
    color_ndens_lim = 0
    # Selecting densities to plot contours
    aa_p = np.linspace(1e6, 1e7, 10).tolist()
    bb_p = [1e8]
    nppplot = aa_p+bb_p
    nppplot   = s_unique_dens
    # Solid line densities contours
    nnnp2 = [5E5, 1E6, 5E6, 1E7, 5E7, 1E8]
    cmap_blues = matplotlib.cm.get_cmap('winter_r')#('Blues')
    for n, ndens in enumerate(nppplot):
        color_ndens_lim += 1
    custom_lines_dens = []
    custom_lines_txt_dens = []
    models_df.drop_duplicates(inplace=True)
    s_nppplot = np.sort(nppplot)
    # Connecting models with same density
    if d_slicer == 'Xdust':
        color_list_b = np.linspace(0.1, 1,color_ndens_lim )#len(unique_dens))
        custom_lines_dens = []
        custom_lines_txt_dens = []
        for n, ndens in enumerate(s_unique_dens):
            if ndens in s_unique_dens:
                linestyle='-'
                models_df['cond_nH_'+'%1.1E' % ndens] = (models_df['central_nh']/ndens > 0.98) & (models_df['central_nh']/ndens < 1.02)
                custom_lines_dens.append(Line2D([0], [0], color=cmap_blues(color_list_b[n]), lw=0.5, linestyle=linestyle))
                ndens_sel = models_df[models_df['cond_nH_'+'%1.1E' % ndens] == True]['central_nh'].values[0]
                custom_lines_txt_dens.append('nH='+'%1.1E' % ndens_sel)
        for n, ndens in enumerate(s_nppplot):
            if ndens in s_nppplot:
                if ndens in nnnp2:
                    linestyle='-'
                    lw=0.6
                    color='0.3'
                else:
                    linestyle='--'
                    lw=0.6
                    color='grey'
                # df Mask for density
                models_df['cond_nH_'+'%1.1E' % ndens] = (models_df['central_nh']/ndens > 0.98) & (models_df['central_nh']/ndens < 1.02)
                # Applytig mask to df
                model_dens = models_df[models_df['cond_nH_'+'%1.1E' % ndens] == True]
                model_dens = model_dens.sort_values(by=['central_T'], ascending = True)
                #ax.plot(model_dens[xvar], model_dens[yvar], marker='None', linestyle=linestyle, color=color, markersize = 0.8, linewidth= lw, zorder=3)#=cmap_blues(color_list_b[n])
                ax.plot(model_dens[xvar], model_dens[yvar], marker='None', linestyle=linestyle, color=cmap_blues(color_list_b[n]), markersize = 0.8, linewidth= lw, zorder=3)#=cmap_blues(color_list_b[n])
                label_text = '%1.1E' % ndens
            # Plotting on selected density contours
            if ndens in nnnp2:
                if ndens == 5e5:
                    xpos=0.5
                    ypos=2.25
                    label_text = r'$5 \times 10^5$'
                if ndens == 1e6:
                    xpos=1.06
                    ypos=1.9
                    label_text = r'$1 \times 10^6$'
                elif ndens == 5e6:
                    xpos=2.5625
                    ypos=2.1
                    label_text = r'$5 \times 10^6$'
                elif ndens == 1e7:
                    xpos=2.5625
                    ypos=2.7
                    label_text = r'$1 \times 10^7$'
                elif ndens == 5e7:
                    xpos=2.5625
                    ypos=3.04
                    label_text = r'$5 \times 10^7$'
                elif ndens == 1e8:
                    xpos=2.5625
                    ypos=3.25
                    label_text = r'$1 \times 10^8$'
                labelLines(model_dens[xvar], model_dens[yvar], ax, xpos=xpos, ypos=ypos, color='0.3', label=label_text, align=True)
        legend_dens = plt.legend(custom_lines_dens, custom_lines_txt_dens, prop={'size': 4}, loc=1)
        ax.add_artist(legend_dens)
    elif d_slicer == 'Ndust':
        linestyle='-'
        lw=0.6
        color_list_b = np.linspace(0.1, 1,color_ndens_lim )#len(unique_dens))
        custom_lines_dcol = []
        custom_lines_txt_dcol = []
        unique_dcol = models_df.Ndust.unique()
        s_unique_col = np.sort(unique_dcol)
        for n, dcol in enumerate(s_unique_col):
            if dcol in s_unique_col:
                linestyle='-'
                custom_lines_dcol.append(Line2D([0], [0], color=cmap_blues(color_list_b[n]), lw=0.5, linestyle=linestyle))
                models_df['cond_Nd_'+'%1.1E' % dcol] = (models_df['Ndust']/dcol > 0.98) & (models_df['Ndust']/dcol < 1.02)
                dcol_sel = models_df[models_df['cond_Nd_'+'%1.1E' % dcol] == True]['Ndust'].values[0]
                custom_lines_txt_dcol.append('Nd='+'%1.1E' % dcol_sel)
        for n, dcol in enumerate(s_unique_col):
            # df Mask for density
            models_df['cond_Nd_'+'%1.1E' % dcol] = (models_df['Ndust']/dcol > 0.98) & (models_df['Ndust']/dcol < 1.02)
            # Applytig mask to df
            model_dcol = models_df[models_df['cond_Nd_'+'%1.1E' % dcol] == True]
            model_dcol = model_dcol.sort_values(by=['central_T'], ascending = True)
            #ax.plot(model_dens[xvar], model_dens[yvar], marker='None', linestyle=linestyle, color=color, markersize = 0.8, linewidth= lw, zorder=3)#=cmap_blues(color_list_b[n])
            ax.plot(model_dcol[xvar], model_dcol[yvar], marker='None', linestyle=linestyle, color=cmap_blues(color_list_b[n]), markersize = 0.8, linewidth= lw, zorder=3)#=cmap_blues(color_list_b[n])
            label_text = '%1.1E' % dcol
        legend_dcol = plt.legend(custom_lines_dcol, custom_lines_txt_dcol, prop={'size': 4}, loc=1)
        ax.add_artist(legend_dcol)
    # To get the same column from obsserved_df
    if xvar == 'central_T':
        xvar_obs = 'Tvib'
    else:
        xvar_obs = xvar
    if yvar == 'logN':
        yvar_obs = 'logN_vib'
        ax.set_ylim([14,17])
    else:
        yvar_obs = yvar        
    # Observed values
    observed_df['Ind_ok'].astype(float)
    observed_df = observed_df.sort_values(by=['Ind_ok'], ascending = True)
    for o, orow in observed_df.iterrows():
        if not pd.isnull(orow[xvar_obs]) and not pd.isnull(orow[yvar_obs]):
            if orow['Ind_ok'] not in [6,7,9,10,12]:#[9, 10, 11, 12]:
                ax.scatter(orow[xvar_obs], orow[yvar_obs], marker='.', c='k', s=2, zorder =10)
                ax.text(orow[xvar_obs], orow[yvar_obs],
                        str(int(orow['Ind_ok'])), ha='left', va='bottom', color='k',
                        rotation='horizontal', fontsize=9, zorder =11)
    # Checking if saving path exists
    out_dir = results_path+'/'+'Figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Axis labels    
    if xvar == 'ratio_v7':
        xlabel = 'v7=1 (39-38) / v7=1 (24-23)'
    else:
        xlabel = xvar
    if yvar == 'ratio_v724_23':
        ylabel = 'v=0 (24-23) / v7=1 (24-23)'
    elif yvar == 'ratio_v739_38':
        ylabel = 'v=0 (39-38) / v7=1 (39-38)'    
    else:
        ylabel = yvar
    if zvar == 'central_T':
        zlabel = r'T$_{\rm{dust}}$ (K)'
    else:
        zlabel = zvar
    # Setting labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar.ax.set_ylabel(zlabel)
    # Legend 
    if legend==True:
        custom_lines_dens = []
        custom_lines_txt_dens = []
        for n, ndens in enumerate(s_nppplot):
            if ndens in nnnp2:
                linestyle='-'
                custom_lines_dens.append(Line2D([0], [0], color='0.4', lw=0.5, linestyle=linestyle))
                custom_lines_txt_dens.append('nH='+'%1.1E' % ndens)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='both', top =True)
    ax.yaxis.set_tick_params(which='both', right=True, labelright=False)
    return fig, ax

def model_slicer(models_df, slicer, only_one_value, cmap,
                 fig, ax, xvar, yvar,  s_order, linestyle, linewidth, integrated,xlimits, ylimist, plotvalue=False):
    """
        Connects models with same parameter value
    """
    if integrated == True:
        xvar_p = xvar + '_int'
        yvar_p = yvar + '_int'
    else:
        xvar_p = xvar
        yvar_p = yvar
    if s_order == 'x':
        sort = xvar_p
    elif s_order == 'y':
        sort = yvar_p
        
    # Rounding esponential numbers
    models_df[slicer+'_round'] = models_df[slicer].apply(lambda x: utiles.rounding_exp(x, 1))
    # Unique values
    sliced_list_u = models_df[slicer+'_round'].unique()
    sliced_list = np.sort(sliced_list_u)
    if slicer == 'Tdust' or slicer == 'central_T' :
        sliced_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    elif slicer == 'nH_u':
        sliced_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
    if cmap[0] == 'color':
        color_list = [cmap[1]]*len(sliced_list)
    else:
    # colormap
        cmap_plot = matplotlib.cm.get_cmap(cmap)#('Blues')
        color_list = np.linspace(0.1, 1, len(sliced_list))
    if only_one_value != False:
        sliced_list = [only_one_value]
    # Subsetting for each unique value
    plotted_colors = []
    for i, sli in enumerate(sliced_list):
        models_df['cond_'+slicer+'_'+'%1.1E' % sli] = (models_df[slicer+'_round']/sli > 0.95) & (models_df[slicer+'_round']/sli < 1.05)
        model_sel = models_df[models_df['cond_'+slicer+'_'+'%1.1E' % sli] == True]
        model_sel = model_sel.sort_values(by=[sort], ascending = True)    
        if cmap[0] == 'color':
            ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='None',
                    linestyle=linestyle, color=cmap[1],
                    markersize = 0.8, linewidth= linewidth)
        else:
            ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='None',
                    linestyle=linestyle, color=cmap_plot(color_list[i]),
                    markersize = 0.8, linewidth= linewidth)
            plotted_colors.append(cmap_plot(color_list[i]))
            if plotvalue == True:
                model_sel['cond_nh_'+'%1.1E' % sli] = (model_sel['nH_u']/2e6 > 0.95) & (model_sel['nH_u']/2e6 < 1.05)
                model_sel_nh = model_sel[model_sel['cond_nh_'+'%1.1E' % sli] == True]
                model_sel_nh = model_sel_nh.sort_values(by=[sort], ascending = True)   
                ax.text(model_sel_nh[xvar_p], model_sel_nh[yvar_p], r'T='+'%1.d' % sli, ha='center', va='center', color='k',
                                rotation='horizontal', fontsize=6, zorder =18)
                ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='None',
                    linestyle=linestyle, color=cmap_plot(color_list[i]),
                    markersize = 0.8, linewidth= linewidth, zorder =18)
    print('\t'+slicer+'\tMin.:'+'%1.2E' % np.min(sliced_list)+'\tMax:'+'%1.2E' % np.max(sliced_list))
    return plotted_colors

def model_slicer_T(models_df, slicer, only_one_value, cmap,
                 fig, ax, xvar, yvar,  s_order, linestyle, linewidth, integrated):
    if integrated == True:
        xvar_p = xvar + '_int'
        yvar_p = yvar + '_int'
    else:
        xvar_p = xvar
        yvar_p = yvar
    if s_order == 'x':
        sort = xvar_p
    elif s_order == 'y':
        sort = yvar_p
    ### Connecting models with same parameter value
    # Rounding esponential numbers
    models_df[slicer+'_round'] = models_df[slicer].apply(lambda x: utiles.rounding_exp(x, 1))
    # Unique values
    sliced_list_u = models_df[slicer+'_round'].unique()
    sliced_list = np.sort(sliced_list_u)
    if slicer == 'Tdust' or slicer == 'central_T' :
        sliced_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    elif slicer == 'nH_u':
        sliced_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7, 2e7, 5e7]
    if cmap[0] == 'color':
        color_list = [cmap[1]]*len(sliced_list)
    else:
    # colormap
        cmap_plot = matplotlib.cm.get_cmap(cmap)#('Blues')
        color_list = np.linspace(0.1, 1, len(sliced_list))
    if only_one_value != False:
        sliced_list = [only_one_value]
    # Subsetting for each unique value
    plotted_colors = []
    models_df['cond_nh_max'] = (models_df['nH_u']==models_df['nH_u'].max())
    model_sel_nh = models_df[models_df['cond_nh_max'] == True]
    
    for i, sli in enumerate(sliced_list[::-1]):
        models_df['cond_'+slicer+'_'+'%1.1E' % sli] = (models_df[slicer+'_round']/sli > 0.95) & (models_df[slicer+'_round']/sli < 1.05)
        model_sel = models_df[models_df['cond_'+slicer+'_'+'%1.1E' % sli] == True]
        model_sel = model_sel.sort_values(by=[sort], ascending = True)  
        if i != 0:
            x = model_sel[xvar_p].tolist()
            y = model_sel[yvar_p].tolist()
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            xp = np.linspace(0,5,100)
            if cmap[0] == 'color':
                ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='None',
                        linestyle=linestyle, color=cmap[1],
                        markersize = 0.8, linewidth= linewidth)
                ax.fill_between(xp, y_prev, y, color=cmap[1], alpha=1.0)
            else:
                ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='None',
                        linestyle=linestyle, color=cmap_plot(color_list[i]),
                        markersize = 0.8, linewidth= linewidth)
                plotted_colors.append(cmap_plot(color_list[i]))
                ax.fill_between(xp, p(xp), p_prev(xp), color=cmap_plot(color_list[i]), alpha=1.0)
        x_prev = model_sel[xvar_p].tolist()
        y_prev = model_sel[yvar_p].tolist()
        z_prev = np.polyfit(x_prev, y_prev, 3)
        p_prev = np.poly1d(z_prev)
    print('\t'+slicer+'\tMin.:'+'%1.2E' % np.min(sliced_list)+'\tMax:'+'%1.2E' % np.max(sliced_list))
    return plotted_colors

def model_slicer_nH(models_df, slicer, only_one_value, cmap,
                 fig, ax, xvar, yvar,  s_order, linestyle, linewidth, integrated, plot_slicer):
    if integrated == True:
        xvar_p = xvar + '_int'
        yvar_p = yvar + '_int'
    else:
        xvar_p = xvar
        yvar_p = yvar
    if s_order == 'x':
        sort = xvar_p
    elif s_order == 'y':
    
    ### Connecting models with same parameter value
    # Rounding esponential numbers
    models_df[slicer+'_round'] = models_df[slicer].apply(lambda x: utiles.rounding_exp(x, 1))
    # Unique values
    sliced_list_u = models_df[slicer+'_round'].unique()
    sliced_list = np.sort(sliced_list_u)
    if slicer == 'Tdust' or slicer == 'central_T' :
        sliced_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    elif slicer == 'nH_u':
        import_dens = [1e6,  5e6, 1e7, 5e7]
    if cmap[0] == 'color':
        color_list = [cmap[1]]*len(sliced_list)
    else:
    # colormap
        cmap_plot = matplotlib.cm.get_cmap(cmap)
        color_list = np.linspace(0.1, 1, len(sliced_list))
    if only_one_value != False:
        sliced_list = [only_one_value]
    # Subsetting for each unique value
    plotted_colors = []
    data_y = {}
    data_x = {}
    for i, sli in enumerate(sliced_list):
        if sli in import_dens:
            linestyle = '-'
            color = 'k'
        else:
            linestyle = '--'
            color='grey'
        models_df['cond_'+slicer+'_'+'%1.1E' % sli] = (models_df[slicer+'_round']/sli > 0.95) & (models_df[slicer+'_round']/sli < 1.05)
        model_sel = models_df[models_df['cond_'+slicer+'_'+'%1.1E' % sli] == True]
        model_sel = model_sel.sort_values(by=[sort], ascending = True)    
        if plot_slicer == True:
            if cmap[0] == 'color':
                ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='o',
                        linestyle=linestyle, color=color,
                        markersize = 0.8, linewidth= linewidth)
            else:
                ax.plot(model_sel[xvar_p], model_sel[yvar_p], marker='o',
                        linestyle=linestyle, color=cmap_plot(color_list[i]),
                        markersize = 0.8, linewidth= linewidth)
                plotted_colors.append(cmap_plot(color_list[i]))
                
        data_x[slicer+'_x_'+'%1.1E' % sli] = model_sel[xvar_p]
        data_y[slicer+'_y_'+'%1.1E' % sli] = model_sel[yvar_p]
            
    print('\t'+slicer+'\tMin.:'+'%1.2E' % np.min(sliced_list)+'\tMax:'+'%1.2E' % np.max(sliced_list))
    return data_x, data_y, plotted_colors

    
def plot_observed_ratios(fig, ax, observed_df, results_path,
                         xvar, yvar, integrated, print_ylabel, print_xlabel, fontsize, color,
                         new_intens=False 
                         ):
    """
        Plot the observed ratios
    """
    
    
    # To get the same column from obsserved_df
    if xvar == 'central_T':
        xvar_obs = 'Tvib'
    else:
        xvar_obs = xvar
    if yvar == 'logN':
        yvar_obs = 'logN_vib'
        ax.set_ylim([14,17])
    else:
        yvar_obs = yvar        
    # Seetting limits depending on vars
    if xvar == 'ratio_v7':
        ax.set_xlim([1,3])
    elif xvar == 'ratio_v724_23':
        ax.set_xlim([1,6])
    elif xvar == 'ratio_v739_38':
        ax.set_xlim([0,15])
    elif xvar == 'ratio_v0':
        ax.set_xlim([0.5,2.5])
    ax.set_ylim([0,5.5])
        
    
    if integrated == True:
        # Axis labels    
        if xvar == 'ratio_v7':
            xvar = 'ratio_v7_int'
            xlabel = 'v7=1 (39-38) / v7=1 (24-23)'
            if new_intens == True:
                xvar = 'ratio_v7_int_new'
                #ax.set_xlim([1.0,2.5])
        elif xvar == 'ratio_v6':
            xvar = 'ratio_v6_int'
            xlabel = 'v6=1 (39-38) / v6=1 (24-23)'
        elif xvar == 'ratio_v0':
            xvar = 'ratio_v0_int'
            xlabel = 'v=0 (39-38) / v=0 (24-23)'
            #ax.set_xlim([0.5,1.5])
        else:
            xlabel = xvar
        if yvar == 'ratio_v724_23':
            yvar = 'ratio_v724_23_int'
            ylabel = 'v=0 (24-23) / v7=1 (24-23)'
        elif yvar == 'ratio_v739_38':
            yvar = 'ratio_v739_38_int'
            ylabel = 'v=0 (39-38) / v7=1 (39-38)'    
            if new_intens == True:
                yvar = 'ratio_v739_38_int_new'
        elif yvar == 'ratio_v624_23':
            yvar = 'ratio_v624_23_int'
            ylabel = 'v=0 (24-23) / v6=1 (24-23)'
        elif yvar == 'ratio_v639_38':
            yvar = 'ratio_v639_38_int'
            ylabel = 'v=0 (39-38) / v6=1 (39-38)'
        else:
            ylabel = yvar
    else: 
        # Axis labels    
        if xvar == 'ratio_v7':
            xlabel = 'v7=1 (39-38) / v7=1 (24-23)'
            if new_intens == True:
                ax.set_xlim([1.0,2.5])
                xvar = 'ratio_v7_new'
        elif xvar == 'ratio_v0':
            xvar = 'ratio_v0'
            xlabel = 'v=0 (39-38) / v=0 (24-23)'
            ax.set_xlim([0.5,1.5])
        else:
            xlabel = xvar
        if yvar == 'ratio_v724_23':
            ylabel = 'v=0 (24-23) / v7=1 (24-23)'
        elif yvar == 'ratio_v739_38':
            ylabel = 'v=0 (39-38) / v7=1 (39-38)'   
            if new_intens == True:
                yvar = 'ratio_v739_38_new'
        else:
            ylabel = yvar
            
    # No SHCs 
    noSHC = [6,7,9,10,11,12]
    # Observed values
    if xvar == 'ratio_v0' or xvar == 'ratio_v0_int':
        exclude_list = [6,7]
    elif xvar == 'ratio_v7' or xvar == 'ratio_v7_int':
        exclude_list = [6,7,9,10,11,12]
    elif xvar == 'ratio_v6' or xvar == 'ratio_v6_int':
        exclude_list = [6,7,8,9,10,11,12]
    else:
        exclude_list = [6,7,9,10,11,12]
    observed_df['Ind_ok'].astype(float)
    observed_df = observed_df.sort_values(by=['Ind_ok'], ascending = True)
    for o, orow in observed_df.iterrows():
        if not pd.isnull(orow[xvar_obs]) and not pd.isnull(orow[yvar_obs]):
            if orow['Ind_ok'] == 2 and 'ratio_v0' in xvar:
               alignment = 'right'
            elif orow['Ind_ok'] in [9,11,14] and 'ratio_v0' in xvar:
               alignment = 'right'
            else:
               alignment = 'left'
            if orow['Ind_ok'] not in noSHC:#[9, 10, 11, 12]:
                if orow['Ind_ok'] in [2,14] and 'ratio_v0' in xvar:
                    xlab = orow[xvar]-0.02
                else:
                    xlab = orow[xvar]+0.02
                if 'v6' in xvar and orow['Ind_ok']==8:
                    continue
                else:
                    ax.scatter(orow[xvar], orow[yvar], marker='o', edgecolor=color, facecolor='white', s=22, zorder =22)
                    label_add = ''
                    yplot = orow[yvar]
                    ax.text(xlab, yplot,
                            str(int(orow['Ind_ok']))+label_add, ha=alignment, va='bottom', color=color,
                            rotation='horizontal', fontsize=fontsize, zorder =22,  weight='bold')
                        
            elif orow['Ind_ok'] not in exclude_list and 'ratio_v0' in xvar:
                if orow['Ind_ok'] in [10,12]:
                    # Plotting their upper limit (3sigma), not their actual value
                    yplot = orow['ratio_v724_23_int_uplims']
                    xlab = orow[xvar]+0.02
                elif orow['Ind_ok'] in [9,11]:
                    # Plotting their upper limit (3sigma), not their actual value
                    #print(orow['Ind_ok'])
                    yplot = orow['ratio_v724_23_int_uplims']
                    #print(yplot)
                    xlab = orow[xvar]-0.02
                else: 
                    xlab = orow[xvar]+0.02
                label_add = ''
                if orow['Ind_ok'] in [12]:
                    yplot = 4.95
                    
                ax.errorbar(orow[xvar], yplot,
                            lolims=True,
                            yerr=0.17,
                            capsize=1.3,
                            capthick=1.3,
                            marker='', color=color, ecolor=color,
                            markeredgecolor=color, markerfacecolor=color,
                            linewidth=0.8, linestyle='--', alpha=1, zorder =22)
            
                ax.text(xlab, yplot,
                        str(int(orow['Ind_ok']))+label_add, ha=alignment, va='bottom', color=color,
                        rotation='horizontal', fontsize=fontsize, zorder =22,  weight='bold')
                        
    # Checking if saving path exists
    out_dir = results_path+'/'+'Figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ax.tick_params(axis='both', which='both', direction='in', zorder=30)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='both', top =True, zorder=30)
    ax.yaxis.set_tick_params(which='both', right=True, labelright=False, zorder=30)
    # Setting labels
    if print_xlabel==True:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    else:
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    if print_ylabel==True:
        ax.set_ylabel(ylabel)
    return fig, ax

def plot_observed_ratios_NGC1068(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                 xvar, yvar, integrated, print_ylabel, print_xlabel, fontsize, color,
                                 man_obs = []):
    # Seetting limits depending on vars
    if xvar == 'ratio_v7':
        ax.set_xlim([1,3])
    elif xvar == 'ratio_v724_23':
        ax.set_xlim([1,6])
    elif xvar == 'ratio_v739_38':
        ax.set_xlim([0,15])
    elif xvar == 'ratio_v0':
        ax.set_xlim([0.5,2.5])
    if yvar == 'ratio_v724_23':
        ax.set_ylim([1.5,5])
    elif yvar == 'ratio_v739_38':
        ax.set_ylim([0,15])
        

    if integrated == True:
        ints = 'integ_'
    else:
        ints = 'peak_'
    jplot = j_ratio.split('_')
    # Axis labels    
    if xvar == 'ratio_v7':
        xvar = 'ratio_v7_int'
        xvar_plot = obs_ratios_dict['v7=1'][ints+j_ratio]
        xlabel = 'v7=1 ('+jplot[0]+') / v7=1 ('+jplot[1]+')'
        ax.set_xlim([1.5, 5.5])
    elif xvar == 'ratio_v6':
        xvar = 'ratio_v6_int'
        xvar_plot = obs_ratios_dict['v6=1'][ints+j_ratio]
        xlabel = 'v6=1 ('+jplot[0]+') / v6=1 ('+jplot[1]+')'
        ax.set_xlim([1.5, 5.5])
    elif xvar == 'ratio_v0':
        xvar = 'ratio_v0_int'
        xvar_plot = obs_ratios_dict['v=0'][ints+j_ratio]
        xlabel = 'v=0 ('+jplot[0]+') / v=0 ('+jplot[1]+')'
        ax.set_xlim([0.5,2.5])
        #ax.set_xlim([0.5,1.5])
    else:
        xlabel = xvar
    #if yvar == 'ratio_v724_23':
    #    yvar = 'ratio_v724_23_int'
    #    ylabel = 'v=0 (24-23) / v7=1 (24-23)'
    #yvar = 'ratio_v724_23'
    yvar_plot = obs_ratios_dict['v0_v7'][ints+vib_ratio]
    ylabel = 'v=0 ('+vib_ratio+') / v7=1 ('+vib_ratio+')' 
    
            
    #ax.scatter(xvar_plot, yvar_plot, marker='o', edgecolor=color, facecolor='white', s=22, zorder =22)
    if len(man_obs) != 0:
        print('Man ratios')
        if 'v7' in xvar:
            xvar_plot_man = man_obs[2]
        elif 'v6' in xvar:
            xvar_plot_man = man_obs[3]
        elif 'v0' in xvar:
            xvar_plot_man = man_obs[1]
        yvar_plot_man = man_obs[0]
        ax.scatter(xvar_plot_man, yvar_plot_man, marker='o', edgecolor=color, facecolor='white', s=22, zorder =22)

        #manobs = [v0_v7_1615_ratio, v0_ratio, v7_ratio, v6_ratio]
        # Plotting manual ratios
        
    # Checking if saving path exists
    out_dir = results_path+'/'+'Figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ax.tick_params(axis='both', which='both', direction='in', zorder=30)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='both', top =True, zorder=30)
    ax.yaxis.set_tick_params(which='both', right=True, labelright=False, zorder=30)
    # Setting labels
    if print_xlabel==True:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    else:
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    if print_ylabel==True:
        ax.set_ylabel(ylabel)
    return fig, ax


def plot_observed_ratios_NGC1068_new(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                 xvar, yvar, integrated, print_ylabel, print_xlabel, fontsize, ylimits, xlimits, color
                                 ):
    # Seetting limits depending on vars
    # if xvar == 'ratio_v7':
    #     ax.set_xlim([1,3])
    # elif xvar == 'ratio_v724_23':
    #     ax.set_xlim([1,6])
    # elif xvar == 'ratio_v739_38':
    #     ax.set_xlim([0,15])
    # elif xvar == 'ratio_v0':
    #     ax.set_xlim([0.5,2.5])
    # if yvar == 'ratio_v724_23':
    #     ax.set_ylim([1.5,5])
    # elif yvar == 'ratio_v739_38':
    #     ax.set_ylim([0,15])
        
    # Axis labels 
    if xvar == 'v0_2416':
        xlabel = 'v=0 (24-23) / v=0 (16-15)'
    elif xvar == 'v7_2416':
        xlabel = 'v7=1 (24-23) / v7=1 (16-15)'
    elif xvar == 'v6_2416':
        xlabel = 'v6=1 (24-23) / v6=1 (16-15)'

       

    if yvar == 'v0_v7_2423':
        ylabel = 'v=0 (24-23) / v7=1 (24-23)'
    elif yvar == 'v0_v7_1615':
        ylabel = 'v=0 (16-15) / v7=1 (16-15)'
    
    marker = 'o'
    markersize = 8
    colormarker = 'w'
    edgemarker = 'k'
    line_width_err = 0.7
    y_err_plot = (ylimits[-1]-ylimits[0])*0.05
    x_err_plot= (xlimits[-1]-xlimits[0])*0.07
    for location in obs_ratios_dict:
        x_ratio_plot = obs_ratios_dict[location][xvar]
        y_ratio_plot = obs_ratios_dict[location][yvar]
        if x_ratio_plot[-1] == 'False' and y_ratio_plot[-1] == 'False':
            # No upper/lower limit in x or y
            ax.errorbar(x_ratio_plot[0], y_ratio_plot[0], yerr=y_ratio_plot[1], xerr=x_ratio_plot[1],
                     marker='None', markersize=markersize,
                     markerfacecolor=colormarker,
                     markeredgecolor='k', markeredgewidth=0.01,
                     ecolor='k',
                     color = colormarker,
                     elinewidth= line_width_err,
                     barsabove= True,
                     zorder=1)
            ax.plot(x_ratio_plot[0], y_ratio_plot[0], marker=marker, markersize=markersize,
                         markerfacecolor=colormarker,
                         markeredgecolor=edgemarker, markeredgewidth=0.8,
                         linestyle = None,zorder=3)
        elif x_ratio_plot[-1] == 'False' and y_ratio_plot[-1] != 'False':
            # upper/lower limit in y
            if y_ratio_plot[-1] == 'uplim':
                uplims = True
                lolims = False
            elif y_ratio_plot[-1] == 'lolim':
                uplims = False
                lolims = True
            elif y_ratio_plot[-1] == 'both':
                uplims = True
                lolims = True
            ax.errorbar(x_ratio_plot[0], y_ratio_plot[0], 
                        uplims=uplims, lolims=lolims,
                        yerr=y_err_plot,
                        marker='None', markersize=markersize,
                        markerfacecolor=colormarker,
                        markeredgecolor='k', markeredgewidth=0.01,
                        ecolor='k',
                        color = colormarker,
                        elinewidth= line_width_err,
                        barsabove= True,
                        zorder=1)
            ax.plot(x_ratio_plot[0], y_ratio_plot[0], marker=marker, markersize=markersize,
                         markerfacecolor=colormarker,
                         markeredgecolor=edgemarker, markeredgewidth=0.8,
                         linestyle = None,zorder=3)
        elif x_ratio_plot[-1] != 'False' and y_ratio_plot[-1] == 'False':
            # upper/lower limit in x
            if x_ratio_plot[-1] == 'uplim':
                uplims = True
                lolims = False
            elif x_ratio_plot[-1] == 'lolim':
                uplims = False
                lolims = True
            elif x_ratio_plot[-1] == 'both':
                uplims = True
                lolims = True
                
            ax.errorbar(x_ratio_plot[0], y_ratio_plot[0], 
                        xuplims=uplims, xlolims=lolims,
                        xerr=x_err_plot,
                        marker='None', markersize=markersize,
                        markerfacecolor=colormarker,
                        markeredgecolor='k', markeredgewidth=0.01,
                        ecolor='k',
                        color = colormarker,
                        elinewidth= line_width_err,
                        barsabove= True,
                        zorder=1)
            ax.plot(x_ratio_plot[0], y_ratio_plot[0], marker=marker, markersize=markersize,
                         markerfacecolor=colormarker,
                         markeredgecolor=edgemarker, markeredgewidth=0.8,
                         linestyle = None,zorder=3)
        elif x_ratio_plot[-1] != 'False' and y_ratio_plot[-1] != 'False':
            # upper/lower limit in x
            if x_ratio_plot[-1] == 'uplim':
                xuplims = True
                xlolims = False
            elif x_ratio_plot[-1] == 'lolim':
                xuplims = False
                xlolims = True
            elif x_ratio_plot[-1] == 'both':
                xuplims = True
                xlolims = True
            # upper/lower limit in y
            if y_ratio_plot[-1] == 'uplim':
                yuplims = True
                ylolims = False
            elif y_ratio_plot[-1] == 'lolim':
                yuplims = False
                ylolims = True
            elif y_ratio_plot[-1] == 'both':
                yuplims = True
                ylolims = True
            ax.errorbar(y_ratio_plot[0], x_ratio_plot[0], 
                        xuplims=xuplims, xlolims=xlolims,
                        uplims=yuplims, lolims=ylolims,
                        yerr=y_err_plot,
                        xerr=x_err_plot,
                        marker='None', markersize=markersize,
                        markerfacecolor=colormarker,
                        markeredgecolor='k', markeredgewidth=0.01,
                        ecolor='k',
                        color = colormarker,
                        elinewidth= line_width_err,
                        barsabove= True,
                        zorder=1)
            ax.plot(y_ratio_plot[0], x_ratio_plot[0], marker=marker, markersize=markersize,
                         markerfacecolor=colormarker,
                         markeredgecolor=edgemarker, markeredgewidth=0.8,
                         linestyle = None,zorder=3)
        
    # Checking if saving path exists
    out_dir = results_path+'/'+'Figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ax.tick_params(axis='both', which='both', direction='in', zorder=30)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='both', top =True, zorder=30)
    ax.yaxis.set_tick_params(which='both', right=True, labelright=False, zorder=30)
    # Setting labels
    if print_xlabel==True:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    else:
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    if print_ylabel==True:
        ax.set_ylabel(ylabel)
    return fig, ax

#Label line with line2D label data
def labelLine(xvals,yvals,xpos, ypos,ax, color='k', label=None, align=True, angle=False, **kwargs):
    from math import atan2,degrees
    xdata = xvals.tolist()
    ydata = yvals.tolist()
    if len(xdata)!= 0 and len(ydata)!=0:
            
        x = xpos
        #if (x < xdata[0]) or (x > xdata[-1]):
        #    print('x label location is outside data range!')
        #    return
        #Find corresponding y co-ordinate and angle of the line
        ip = 1
        for i in range(len(xdata)):
            if x < xdata[i]:
                ip = i
                break
        y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])
        if not label:
            label = ''
        if align:
            #Compute the slope
            dx = xdata[ip] - xdata[ip-1]
            dy = ydata[ip] - ydata[ip-1]
            ang = degrees(atan2(dy,dx))
            #Transform to screen co-ordinates
            pt = np.array([x,y]).reshape((1,2))
            trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]
        else:
            trans_angle = 0
        #Set a bunch of keyword arguments
        if 'color' not in kwargs:
            kwargs['color'] = color
        if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
            kwargs['ha'] = 'center'
        if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
            kwargs['va'] = 'center'
        if 'backgroundcolor' not in kwargs:
            kwargs['backgroundcolor'] = ax.get_facecolor()
        if 'clip_on' not in kwargs:
            kwargs['clip_on'] = True
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 20
        if angle== False:
            angulo = trans_angle
        else:
            angulo = angle
        weight='bold'
        t = ax.text(x,ypos,label,rotation=angulo, weight=weight, fontsize=6, verticalalignment='center', horizontalalignment='center',**kwargs)
        t.set_bbox(dict(facecolor='white', alpha=0.0))
        

    
def labelLines(df, ndens, row, column, xvar, yvar, ax, color, align=True,**kwargs):
    slicer = 'nH_u'
    df['cond_'+'%1.1E' % ndens] = (df[slicer+'_round']/ndens > 0.995) & (df[slicer+'_round']/ndens < 1.005)
    model_sel = df[df['cond_'+'%1.1E' % ndens] == True]
    model_sel = model_sel.sort_values(by=[yvar], ascending = True)    
    # Top left plot
    angle = False
    if row==0 and column ==0:
        noprint = False
        if ndens == 1e6:
            xpos=0.82
            ypos=2.25
            label_text = r'$\boldsymbol{1 \times 10^6}$'
            angle = -60
        elif ndens == 2e6:
            xpos=1.25
            ypos=3.25
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -45
        elif ndens == 3e6:
            xpos=1.4
            ypos=4.2
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -55
        elif ndens == 4e6:
            noprint = True
        elif ndens == 5e6:
            noprint = True
        else:
            noprint = True
    # Bottom middle plot
    elif row==1 and column ==0:
        noprint = False
        if ndens == 1e6:
            xpos=0.57
            ypos=3.25
            label_text = r'$\boldsymbol{1 \times 10^6}$'
            angle = -65
        elif ndens == 2e6:
            xpos=1.40
            ypos=2.2
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -48
        elif ndens == 3e6:
            xpos=1.38
            ypos=3.12
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -40
        elif ndens == 4e6:
            xpos=1.39
            ypos=4.10
            label_text = r'$\boldsymbol{4 \times 10^6}$'
            angle = -60
        elif ndens == 5e6:
            xpos=1.6
            ypos=4.4
            label_text = r'$\boldsymbol{5 \times 10^6}$'
            angle = -57
        else:
            xpos=0
            ypos=0
            label_text = ''
            noprint = True
    # Bottom left plot
    elif row==2 and column ==0:
        noprint = False
        if ndens == 1e6:
            xpos=0.53
            ypos=3.0
            label_text = r'$\boldsymbol{1 \times 10^6}$'
            angle = -75
        elif ndens == 2e6:
            xpos=0.9
            ypos=3.4
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -62
        elif ndens == 3e6:
            xpos=1.3
            ypos=2.9
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -58
        elif ndens == 4e6:
            xpos=1.45
            ypos=3.2
            label_text = r'$\boldsymbol{4 \times 10^6}$'
            angle = -52
        elif ndens == 5e6:
            xpos=1.6
            ypos=3.3
            label_text = r'$\boldsymbol{5 \times 10^6}$'
            angle = -48
        elif ndens == 1e7:
            xpos=1.56
            ypos=4.4
            label_text = r'$\boldsymbol{1 \times 10^7}$'
            angle = -55
        else:
            xpos=0
            ypos=0
            label_text = ''
            noprint = True
    # Top right plot
    elif row==0 and column ==1:
        noprint = False
        if ndens == 1e6:
            xpos=0.91
            ypos=2.06
            label_text = r'$\boldsymbol{1 \times 10^6}$'
            angle = -60
        elif ndens == 2e6:
            xpos=1.53
            ypos=2.07
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -45
        elif ndens == 3e6:
            xpos=2.06
            ypos=1.95
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -38
        elif ndens == 4e6:
            xpos=2.08
            ypos=2.47
            label_text = r'$\boldsymbol{4 \times 10^6}$'
            angle = -40
        elif ndens == 5e6:
            xpos=2.10
            ypos=3.00
            label_text = r'$\boldsymbol{5 \times 10^6}$'
            angle = -39
        elif ndens == 1e7:
            xpos=2.10
            ypos=3.70
            label_text = r'$\boldsymbol{1 \times 10^7}$'
            angle = -39
        else:
            noprint = True
    # Middle right plot
    elif row==1 and column ==1:
        noprint = False
        if ndens == 1e6:
            noprint = True
        elif ndens == 2e6:
            xpos=1.04
            ypos=3.06
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -55
        elif ndens == 3e6:
            xpos=1.72
            ypos=2.20
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -47
        elif ndens == 4e6:
            xpos=1.94
            ypos=2.37
            label_text = r'$\boldsymbol{4 \times 10^6}$'
            angle = -43
        elif ndens == 5e6:
            xpos=1.94
            ypos=3.27
            label_text = r'$\boldsymbol{5 \times 10^6}$'
            angle = -43
        elif ndens == 1e7:
            xpos=1.92
            ypos=4.00
            label_text = r'$\boldsymbol{1 \times 10^7}$'
            angle = -43
            
        else:
            noprint = True
    # Bottom right plot
    elif row==2 and column ==1:
        noprint = False
        if ndens == 1e6:
            noprint = True
        elif ndens == 2e6:
            xpos=0.96
            ypos=2.4
            label_text = r'$\boldsymbol{2 \times 10^6}$'
            angle = -63
        elif ndens == 3e6:
            xpos=1.15
            ypos=3.0
            label_text = r'$\boldsymbol{3 \times 10^6}$'
            angle = -55
        elif ndens == 4e6:
            xpos=1.7
            ypos=2.3
            label_text = r'$\boldsymbol{4 \times 10^6}$'
            angle = -48
        elif ndens == 5e6:
            xpos=1.76
            ypos=2.84
            label_text = r'$\boldsymbol{5 \times 10^6}$'
            angle = -50
        elif ndens == 1e7:
            xpos=2.1
            ypos=3.2
            label_text = r'$\boldsymbol{1 \times 10^7}$'
            angle = -40
            
        else:
            noprint = True
    
    # Bottom right plot
    if column ==2:
        noprint = True
        
    xvals= model_sel[xvar]
    yvals= model_sel[yvar]
    if noprint != True:
        print(row)
        print(column)
        print('%1.1E' % ndens)
        
        labelLine(xvals,yvals,xpos,ypos,ax,color,label_text,align, angle)


        



def HC3N_modelsub_new_NGC1068(rnube_list, hc3n_model, integrated, dens_profile, results_path,
                              hc3n_selection, tau_selection,
                              color_nh, color_obs,
                              xvar, yvar, out_dir, fig_format, vib_ratio, j_ratio, obs_ratios_dict, suffix_name='',
                              nline='tau100', plot_lines = False,
                              columnas = 3, write_temps = False, fillcolors = False,
                              plot_linetemps=False, plot_all_dens=False,
                              cont_map = False, use_all_dens =False, use_all_temps=False,
                              labelling = False, nh_e17_list = [1e6, 2.5e6],
                              dens_colorbar = False):
    """
    Plotting models parameter space
    vib_ratio = '16-15'
    j_ratio = '24-23_16-15'
    """
    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib as mpl
    profile_path = 'r_'+'%1.1f' % dens_profile
    
    # Selecting by radius
    for rad in rnube_list:
        filas = len(tau_selection)
        #columnas = 2
        fig, axarr = plt.subplots(filas, columnas, sharex=False, figsize=(10,8))
        print(len(axarr))
        fig.subplots_adjust(hspace=0)
        hc3n_model['cond_sel_r'] = (hc3n_model['rnube_pc_u']==rad)
        hc3n_model_rad1 = hc3n_model[hc3n_model['cond_sel_r'] == True]
        chunks_row = [[tau_selection[0],tau_selection[0], tau_selection[0]],
                      [tau_selection[1],tau_selection[1], tau_selection[1]],
                      [tau_selection[2],tau_selection[2], tau_selection[2]]]
        # Plotting each row
        # one row for each ratio
        for r,row_fig in enumerate(chunks_row):
            # Only bottom subplots with X label
            if r ==(len(chunks_row)-1):
                print_xlabel = True
            else:
                print_xlabel = False
            # Selecting by tau100
            for nl, tau in enumerate(row_fig):
                hc3n_model_rad1['cond_sel_tau'] = (hc3n_model_rad1['tau100_u']==tau)
                hc3n_model_radtau = hc3n_model_rad1[hc3n_model_rad1['cond_sel_tau'] == True]
                if filas == 1:
                    ax = axarr[nl]
                else:
                    ax = axarr[r, nl]
                # Only left subplots with Y label
                if nl == 0:
                    print_ylabel = False
                    xvar = 'ratio_v0'
                    xv_pre = 'v=0_'
                elif nl == 1:
                    print_ylabel = False
                    xvar = 'ratio_v7'
                    xv_pre = 'v7=1_'
                elif nl == 2:
                    print_ylabel = False
                    xvar = 'ratio_v6'
                    xv_pre = 'v6=1_'
                # Limits
                if xvar == 'ratio_v0':
                    xlimits = [0.4,1.8]
                elif xvar == 'ratio_v7':
                    xlimits =  [1.4, 5.6]
                elif xvar == 'ratio_v6':
                    xlimits =  [1.4, 5.6]
                if yvar == 'ratio_v724_23':
                    ylimits = [0,5]
                elif yvar == 'ratio_v716_15':
                    ylimits = [0,5]
                    
                # Writting model parameters                            
                # tau100
                ax.text(0.925, 0.91, r'$\boldsymbol{\tau_{100}='+'%1.d' % tau+r'}$', ha='right', va='center', color='k',
                                rotation='horizontal', fontsize=6, zorder =11,transform = ax.transAxes)
                # Selecting by Xline or Nline
                Nline_linestyles = ['-', '--']
                if integrated == True:
                    xvar_p = xv_pre +'integ_'+ j_ratio
                    yvar_p = 'v0_v7_integ_' + vib_ratio
                else:
                    xvar_p = xv_pre +'peak_'+ j_ratio
                    yvar_p = 'v0_v7_peak_' + vib_ratio
                fillbtw_lines = {}
                for l, line in enumerate(hc3n_selection):
                    if line > 1:
                        if nline == 'tau100':
                            selection = 'Nline_tau100'
                        else:
                            selection = 'Nline'
                        sel_lab = 'N'
                        #other_lab = 'X'
                        #other_selection = 'Xline'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.95)& (hc3n_model_radtau[selection]/line < 1.05)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    else:
                        selection = 'Xline'
                        sel_lab = 'X'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.995)& (hc3n_model_radtau[selection]/line < 1.005)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    
                    # Dropping Tdust = 125
                    hc3n_model_radtauNline['cond_sel_t1'] = (hc3n_model_radtauNline['Tdust'] !=  125)
                    hc3n_model_radtauNline = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_sel_t1'] == True]
                    hc3n_model_nline = hc3n_model_radtauNline
                    if len(hc3n_model_nline) != 0:
                        fillbtw_lines[line] = {}
                        print('Plotting')
                        
                        if nline == 'tau100':
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_tau100_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        else:
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        
                        if not use_all_dens:
                            #Dropping values with densities outside 5e5 XXXX ver como hago esto parametro
                            if tau != 16:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                            else:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                           
                            hc3n_model_nline['cond_sel_nrange1'] = (hc3n_model_nline['nH_u'] !=  8E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange1'] == True]
                            
                            hc3n_model_nline['cond_sel_nrange2'] = (hc3n_model_nline['nH_u'] !=  9E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange2'] == True]
                        
                            hc3n_model_nline['cond_sel_nrange3'] = (hc3n_model_nline['nH_u'] >5e7)  & (hc3n_model_nline['nH_u']  <1e8)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange3'] == False]
                            
                            hc3n_model_nline['cond_sel_nrange4'] = (hc3n_model_nline['nH_u'] >5e6)  & (hc3n_model_nline['nH_u']  <1e7)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange4'] == False]
                        
                        if nline == 'tau100':
                            # Nline from tau100
                            Nl_list_rad  = hc3n_model_nline['Nline_tau100_u'].unique() 
                        else:
                            # Original Nline from Xline *rnube*nH
                            Nl_list_rad  = hc3n_model_nline['Nline_u'].unique() 
                        
                        X_list_rad  = hc3n_model_nline['Xline_u'].unique() 
                        tau100_list_rad  = hc3n_model_nline['tau100'].unique()
                        nh_list_rad = hc3n_model_nline['nH_u'].unique()
                        Nh2_list_rad = hc3n_model_nline['NH2_tau100_u'].unique()
                        Tdust_list_rad = hc3n_model_nline['Tdust'].unique()
                        
                        print('\ttau100='+'%1.1f' % tau100_list_rad[0])
                        print('\tN(H2)='+'%1.1E' % Nh2_list_rad[0])
                        print('\tN(HC3N)='+'%1.1E' % Nl_list_rad[0])
                        print('\tX(HC3N)='+'%1.1E' % X_list_rad[0])
                   
                        if cont_map:
                            # Plotting continuos dustmap
                            hc3n_model_nline_sort_plot = hc3n_model_nline.sort_values(by=[xvar_p, yvar_p])
    
                            xi = np.linspace(xlimits[0], xlimits[1], 1000)
                            yi = np.linspace(ylimits[0], ylimits[1], 1000)
                            xi, yi = np.meshgrid(xi, yi)
                            rbf = scipy.interpolate.Rbf(hc3n_model_nline_sort_plot[xvar_p], hc3n_model_nline_sort_plot[yvar_p],
                                                        hc3n_model_nline_sort_plot['Tdust'],
                                                        function='linear', smooth=0.0)#20.0
                            zi = rbf(xi, yi)
                            #f = interpolate.interp2d(x,y,test,kind='cubic')
                            cmap_new = sns.light_palette('pale red', input="xkcd", as_cmap=True)
                            ax.contourf(xi, yi, zi,  cmap=cmap_new, origin='lower', levels=400, zorder=0.002)
                    
                        if nl == 1 and tau == 16:
                            nh_list_1e17 = nh_e17_list
                        else:
                            nh_list_1e17 = nh_e17_list
                        # Grouping by density
                        if line == 1E17:
                            nh_list_plot = nh_list_1e17
                        else:
                            if plot_all_dens:
                                nh_list_plot = nh_list_rad
                            else:
                                nh_list_plot = [5e5, 1e6, 2.5e6, 5e6, 1e7]
                        nh_text = [1e6, 5e6, 1e7]
                        
                        redpink     = sns.xkcd_palette(['red pink'])[0]
                        oblue       = sns.xkcd_palette(['ocean blue'])[0]
                        elime       = sns.xkcd_palette(['electric lime'])
                        melon       = sns.xkcd_palette(['melon'])[0]
                        aquamarine  = sns.xkcd_palette(['aquamarine'])[0]
                        aquagreen   = sns.xkcd_palette(['aqua green'])[0]
                        turqoise    = sns.xkcd_palette(['turquoise'])[0]
                        water       = sns.xkcd_palette(['water blue'])[0]
                        brown       =sns.xkcd_palette(['terracota'])[0]
                        purple      =sns.xkcd_palette(['purple'])[0]
                        orange      = sns.xkcd_palette(['blood orange'])[0]
                        green       = sns.xkcd_palette(['kelly green'])[0]
                        azure       = sns.xkcd_palette(['azure'])[0]
                        
                        #ncolors = {5e5: 'darkviolet', 1e6: 'dodgerblue', 2.5e6: 'cyan', 5e6: 'limegreen', }#1e7:'darkred'}
                        #ncolors = {5e5: 'darkviolet', 1e6: 'dodgerblue', 2.5e6: 'cyan', 5e6: 'limegreen', 1e7: 'xkcd:red pink'}#1e7:'darkred'}
                        
                        ncolors = {1e4: 'darkviolet', 5e4: 'dodgerblue', 1e5: 'cyan', 5e5: 'limegreen', 1e6: 'xkcd:red pink', 5e6: 'xkcd:terracota', 1e7: 'xkcd:blood orange'}
                        for dd, dens in enumerate(nh_list_plot):
                            hc3n_model_nline['cond_selnh'] = (hc3n_model_nline['nH_u'] ==  dens)
                            hc3n_model_nline_nH = hc3n_model_nline[hc3n_model_nline['cond_selnh'] == True]
                            
                            # Sorting values for interpolation and plotting
                            hc3n_model_nline_nH_plot = hc3n_model_nline_nH.sort_values(by=[xvar_p, yvar_p])
                            if line == 1E17:
                                smooth = 0.002
                            else:
                                smooth = 0.01
                            # Interpolation
                            #nH_interp = interpolate.interp1d(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], fill_value='extrapolate')
                            # Splines
                            tck = interpolate.splrep(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], s=smooth)
                            xvals_nH = np.linspace(xlimits[0], xlimits[1],1000)
                            yvals_nH = interpolate.splev(xvals_nH, tck, der=0)
                            
                            if dens in nh_list_1e17:
                                fillbtw_lines[line][dens] = {'xvals': xvals_nH, 'yvals': yvals_nH}
                            # Plotting nH color
                            color_dens = True
                            if color_dens:
                                if dens in ncolors.keys():
                                    ax.plot(xvals_nH, yvals_nH,
                                            marker='None', markerfacecolor ='None', markersize = 4,
                                            color = ncolors[dens], linestyle = Nline_linestyles[l], linewidth = 1.2, zorder=1/10.)
                                    plot_true_values = False
                                    if plot_true_values:
                                        # Plotting true values without interp to compare
                                        ax.plot(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = 'k', linestyle = ':', linewidth = 1.2, zorder=1/10.)
                                else:
                                    if plot_all_dens:
                                        ax.plot(xvals_nH, yvals_nH,
                                            marker='None', markerfacecolor ='None', markersize = 4,
                                            color = 'k', linestyle = ':', linewidth = 0.65, zorder=1/10.)
                            else:
                                ax.plot(xvals_nH, yvals_nH,
                                    marker='None', markerfacecolor ='None', markersize = 4,
                                    color = 'k', linestyle = Nline_linestyles[l], linewidth = 0.5, zorder=1/10.)
                            
                            if labelling:
                                if dens in nh_text:
                                    if r==0:
                                        if nl ==0:
                                            if dens == 1e6:
                                                xpos = 1
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                        elif nl == 1:
                                            if dens == 1e7:
                                                xpos = 2
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                        elif nl == 2:
                                            if dens == 1e7:
                                                xpos = 2
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                    elif r==1:
                                        if nl ==0:
                                            if dens == 1e6:
                                                xpos = 1.5
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                        elif nl == 1:
                                            if dens == 1e6:
                                                xpos = 1.3
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                        elif nl == 2:
                                            if dens == 1e6:
                                                xpos = 1.1
                                                ypos = interpolate.splev(xpos, tck, der=0)
                                                labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)

                        if line != 1E17:
                            # Grouping by tdust
                            # I use a dictionary to group all densities given a Temperature.
                            # Then I use this interpolated curves to color the space between them
                            curve_temps_dictio = {}
                            # For temperatures i use all the data available i.e. hc3n_model_radtauNline
                            if use_all_temps:
                                temp_list_plot = np.sort(Tdust_list_rad)
                            else:
                                temp_list_plot = np.arange(100, 650, 50)
                            colors_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=False)
                            for tind, tdus in enumerate(temp_list_plot):
                                print('\t\t'+str(tdus))
                                hc3n_model_radtauNline['cond_seltdust'] = (hc3n_model_radtauNline['Tdust'] ==  tdus)
                                hc3n_model_nline_Td = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_seltdust'] == True]
                                if integrated == True:
                                    xvar_p = xv_pre +'integ_'+ j_ratio
                                    yvar_p = 'v0_v7_integ_' + vib_ratio
                                else:
                                    xvar_p = xv_pre +'peak_'+ j_ratio
                                    yvar_p = 'v0_v7_peak_' + vib_ratio
                                hc3n_model_nline_Td_plot = hc3n_model_nline_Td.sort_values(by=[yvar_p, xvar_p])
                                
                                tdust_interp = interpolate.interp1d(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p], fill_value='extrapolate')
                                
                                xvals = np.linspace(xlimits[0], xlimits[1],100)
                                yvals = tdust_interp(xvals)
                                curve_temps_dictio[tdus] = {'xvals': xvals, 'yvals': yvals, 'Tdust': tdus, 'Color': colors_tdust[tind], 'ind': tind}
                                if plot_linetemps:
                                    if tdus in [125, 400, 450, 500, 550, 600]:
                                        if tdus ==350:
                                            temp_color = 'lime'
                                        elif tdus == 400:
                                            temp_color = 'b'
                                        elif tdus == 450:
                                            temp_color = 'purple'
                                        elif tdus == 500:
                                            temp_color = 'g'
                                        elif tdus == 550:
                                            temp_color = 'r'
                                        elif tdus == 600:
                                            temp_color = 'yellow'
                                        else:
                                            temp_color = 'k'
                                        ax.plot(xvals, yvals,
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = temp_color, linestyle = '-', linewidth = 0.5, zorder=2/10.)
                                
                                # Plotting T color
                                dens_color = [1e6, 5e6, 1e7]
                                dcolors = ['b', 'g', 'r']
                                ax.plot(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p],
                                        marker='None', markerfacecolor ='None', markersize = 4,
                                        color = 'k', linestyle = 'None', linewidth = 0.5, zorder=1/10.)
                                for d, dds in enumerate(dens_color):
                                    hc3n_model_nline_Td['cond_selnH'] = (hc3n_model_nline_Td['nH_u'] ==  dds)
                                    hc3n_model_nline_TdnH = hc3n_model_nline_Td[hc3n_model_nline_Td['cond_selnH'] == True]
                                    ax.plot(hc3n_model_nline_TdnH[xvar_p], hc3n_model_nline_TdnH[yvar_p],
                                            marker='None', markerfacecolor ='None', markersize = 4,
                                            color = dcolors[d], linestyle =  '-', linewidth = 0.5, zorder=3/10.)
                            
                            od_curve_temps_dictio = collections.OrderedDict(sorted(curve_temps_dictio.items()))
                            #od_curve_temps_dictio = {k: curve_temps_dictio[k] for k in sorted(curve_temps_dictio)}
                            for item, next_item in utiles.iterate(od_curve_temps_dictio.items()):
                                if next_item != 0:
                                    # to check if iteration is ok
                                    # print(item[1]['ind'], item[1]['Tdust'], next_item[1]['ind'], next_item[1]['Tdust'])
                                    ax.fill_between(item[1]['xvals'], item[1]['yvals'],
                                                    next_item[1]['yvals'], facecolor=item[1]['Color'], zorder=0)
                
                if fillcolors:
                    for density in nh_list_1e17:
                        fill_color = ncolors[density]
                        ax.fill_between(fillbtw_lines[5E16][density]['xvals'],
                                        fillbtw_lines[5E16][density]['yvals'], fillbtw_lines[1E17][density]['yvals'],
                                        facecolor = fill_color, zorder = 0.001, alpha=0.25)
            
                fontsize = 7
                plot_observed_ratios_NGC1068(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                             xvar, yvar, integrated, print_ylabel, print_xlabel,
                                             fontsize=fontsize, color=color_obs
                                             )
                    
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
                                                 
                if r == (len(chunks_row)-1):
                    #ax.set_xlabel(xlabel)
                    ax.tick_params(axis='x', labeltop=False, labelbottom=True, labelsize='8',zorder=30)
                else:
                    ax.tick_params(axis='x', labeltop=False, labelbottom=False, labelsize='8', zorder=30)
                ax.tick_params(axis='both', zorder=30)
                print('-------------')
        # Position of the axes
        ax1 = axarr[0, 1].get_position().get_points().flatten()
        ax2 = axarr[2, 1].get_position().get_points().flatten()
        # Temperature colorbar
        if use_all_temps:
            temp_list_plot = np.sort(Tdust_list_rad)
        else:
            temp_list_plot = np.arange(100, 650, 50)
            
        #Tdust_list_rad_sort = sorted(Tdust_list_rad)
        cmap_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=True)
        cbar_ax = fig.add_axes([0.92, ax2[1], 0.025, ax1[3]-ax2[1]])
        #temp_list = np.arange(100, 700, 100)
        temp_list = np.arange(150, 600, 50)
        temp_norm_new = mpl.colors.BoundaryNorm(temp_list_plot, cmap_tdust.N)
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_tdust,
                                norm=temp_norm_new,
                                spacing='uniform',
                                orientation='vertical',
                                extend='neither',
                                ticks=temp_list)
        cb.set_label(r'$T_{\rm{dust}}$ (K)')
        cb.ax.tick_params(axis='y', direction='in')
        # This is for plotting horizontal lines on the cbar
        #cb.ax.plot([0, 1], [0.2, 0.2], 'b')
        #cb.ax.plot([0, 1], [0.4, 0.4], 'g')
        #cb.ax.plot([0, 1], [0.6, 0.6], 'r')
        #for j, lab in enumerate(Tdust_list_rad):
        #    lab_int = int(lab)
        #    cb.ax.text(1, (j) / (len(Tdust_list_rad)-1), lab_int, ha='center', va='center')
        ylabel = 'v=0 (24-23) / v7=1 (24-23)'
        fig.text(0.075,0.5, ylabel, ha="center", va="center", rotation=90)
        
        if dens_colorbar:
            #dens_colors = ['darkviolet', 'dodgerblue', 'cyan', 'limegreen', 'darkred']
            dens_colors = ['darkviolet', 'dodgerblue',  'cyan', 'limegreen', 'xkcd:red pink']

            dmaps_sns = LinearSegmentedColormap.from_list( 'dens_colors', dens_colors, N=len(dens_colors))
            # Position of the axes
            ax1 = axarr[0, 0].get_position().get_points().flatten()
            ax2 = axarr[0, 2].get_position().get_points().flatten()
            # Density colorbar
            cbardens_ax = fig.add_axes([ax1[0], 0.90, ax2[2]-ax1[0], 0.015])
            dens_list = [0.5, 1, 2.5, 5, 10] 
            #dmaps = matplotlib.cm.get_cmap('winter')
            #dmaps_sns = matplotlib.colors.LinearSegmentedColormap.from_list('cblues', dens_sns_map, N=len(dens_colors)+1)
            #dens_list2 = np.array([1, 2, 3, 4, 5])-0.5
            #dens_norm = mpl.colors.BoundaryNorm(dens_list, dmaps.N)
                
            #tick_locs = (np.arange(len(dens_list)+1) + 0.(5)*(len(dens_list)-1)/len(dens_list))
            tick_locs = np.arange(0,np.max(dens_list), np.max(dens_list)/(len(dens_list)+1))
            tick_locs2 = []
            for t, tick in enumerate(tick_locs):
                if t != 4:
                    if t < 4:
                        tick_locs2.append(tick+np.max(dens_list)/(len(dens_list)+1)/2)
                    else:
                        tick_locs2.append(tick-np.max(dens_list)/(len(dens_list)+1)/2)
            dens_norm_sns = mpl.colors.BoundaryNorm(tick_locs, dmaps_sns.N)
            cbdens = mpl.colorbar.ColorbarBase(cbardens_ax, cmap=dmaps_sns,
                                    norm=dens_norm_sns,
                                    #boundaries=tick_locs,
                                    spacing='uniform',
                                    orientation='horizontal',
                                    extend='neither',
                                    ticks=tick_locs2
                                    )
            cbdens.ax.set_xticklabels(['0.5', '1.0', '2.5', '5.0', '10'])
            cbdens.ax.tick_params(axis='x', direction='in',labeltop=True, labelbottom=False, bottom=False, top=True, labelsize='8', zorder=30)
            cbdens.set_label(r'$n(\rm{H}_2)$ ($10^6\,$cm$^{-3}$)', labelpad=-40)
        
        if write_temps:
            # Plotting temperature indicators (by hand!!)
            # T=200
            axarr[0, 0].text(1.25, 3.2,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            #axarr[1, 0].text(0.7, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            #axarr[2, 0].text(0.6, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 1].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 2].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 2].text(2.25, 3.55,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 2].text(2.2, 3.6,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            
        fig_name = 'TEST_HC3N_models'
        for formato in fig_format:
            fig.savefig(out_dir+fig_name+suffix_name+'.'+formato, bbox_inches='tight', transparent=False, dpi=400)


def HC3N_modelsub_new_NGC1068_chek(rnube_list, hc3n_model, integrated, dens_profile, results_path,
                                   hc3n_selection, tau_selection, 
                                   color_nh, color_obs, 
                                   xvar, yvar, out_dir, fig_format, vib_ratio, j_ratio, obs_ratios_dict, suffix_name='',
                                  nline='tau100', plot_lines = False, plot_only_conts=True,
                                  columnas = 3, write_temps = False, fillcolors = False,
                                  plot_linetemps=False, plot_all_dens=False,
                                  cont_map = False, use_all_dens =False, use_all_temps=False,
                                  labelling = False, nh_e17_list = [1e6, 2.5e6],
                                  dens_colorbar = False, fig_name = 'test_model', man_obs =[]):
    """
    Plotting models parameter space
    vib_ratio = '16-15'
    j_ratio = '24-23_16-15'
    """
    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib as mpl
    profile_path = 'r_'+'%1.1f' % dens_profile
    
    # Selecting by radius
    for rad in rnube_list:
        filas = len(tau_selection)
        #columnas = 2
        fig, axarr = plt.subplots(filas, columnas, sharex=False, figsize=(10,8))
        fig.subplots_adjust(hspace=0)
        hc3n_model['cond_sel_r'] = (hc3n_model['rnube_pc_u']==rad)
        hc3n_model_rad1 = hc3n_model[hc3n_model['cond_sel_r'] == True]
        chunks_row = []
        for tt, tau in enumerate(tau_selection):
            chunks_row.append([tau_selection[tt],tau_selection[tt], tau_selection[tt]])
        #chunks_row = [[tau_selection[0],tau_selection[0], tau_selection[0]],
        #              [tau_selection[1],tau_selection[1], tau_selection[1]],
        #              [tau_selection[2],tau_selection[2], tau_selection[2]]]
        # Plotting each row
        # one row for each ratio
        for r,row_fig in enumerate(chunks_row):
            # Only bottom subplots with X label
            if r ==(len(chunks_row)-1):
                print_xlabel = True
            else:
                print_xlabel = False
            # Selecting by tau100
            for nl, tau in enumerate(row_fig):
                hc3n_model_rad1['cond_sel_tau'] = (hc3n_model_rad1['tau100_u']==tau)
                hc3n_model_radtau = hc3n_model_rad1[hc3n_model_rad1['cond_sel_tau'] == True]
                if filas == 1:
                    ax = axarr[nl]
                else:
                    ax = axarr[r, nl]
                # Only left subplots with Y label
                if nl == 0:
                    print_ylabel = False
                    xvar = 'ratio_v0'
                    xv_pre = 'v=0_'
                elif nl == 1:
                    print_ylabel = False
                    xvar = 'ratio_v7'
                    xv_pre = 'v7=1_'
                elif nl == 2:
                    print_ylabel = False
                    xvar = 'ratio_v6'
                    xv_pre = 'v6=1_'
                # Limits
                if xvar == 'ratio_v0':
                    if j_ratio == '24-23_16-15':
                        xlimits = [0.1,1.5]
                    elif j_ratio == '39-38_24-23':
                        xlimits =  [1, 5]
                elif xvar == 'ratio_v7':
                    if j_ratio == '24-23_16-15':
                        xlimits =  [0.5, 2.5]
                    elif j_ratio == '39-38_24-23':
                        xlimits =  [1, 5]
                    
                elif xvar == 'ratio_v6':
                    if j_ratio == '24-23_16-15':
                        xlimits =  [0.5, 4.5]
                    elif j_ratio == '39-38_24-23':
                        xlimits =  [3, 6]
                if yvar == 'ratio_v724_23':
                    ylimits = [1.5,8]
                elif yvar == 'ratio_v716_15':
                    ylimits = [1.5,8]
                    
                # Writting model parameters                            
                # tau100
                ax.text(0.925, 0.91, r'$\boldsymbol{\tau_{100}='+'%1.d' % tau+r'}$', ha='right', va='center', color='k',
                                rotation='horizontal', fontsize=6, zorder =11,transform = ax.transAxes)
                # Selecting by Xline or Nline
                Nline_linestyles = ['-', '--']
                if integrated == True:
                    xvar_p = xv_pre +'integ_'+ j_ratio
                    yvar_p = 'v0_v7_integ_' + vib_ratio
                else:
                    xvar_p = xv_pre +'peak_'+ j_ratio
                    yvar_p = 'v0_v7_peak_' + vib_ratio
                fillbtw_lines = {}
                for l, line in enumerate(hc3n_selection):
                    if line > 1:
                        if nline == 'tau100':
                            selection = 'Nline_tau100'
                        else:
                            selection = 'Nline'
                        sel_lab = 'N'
                        #other_lab = 'X'
                        #other_selection = 'Xline'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.95)& (hc3n_model_radtau[selection]/line < 1.05)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    else:
                        selection = 'Xline'
                        sel_lab = 'X'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.995)& (hc3n_model_radtau[selection]/line < 1.005)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    
                    # Dropping Tdust = 125
                    # hc3n_model_radtauNline['cond_sel_t1'] = (hc3n_model_radtauNline['Tdust'] !=  125)
                    # hc3n_model_radtauNline = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_sel_t1'] == True]
                    hc3n_model_nline = hc3n_model_radtauNline
                    if len(hc3n_model_nline) != 0:
                        # just continue of selected models is not epty
                        fillbtw_lines[line] = {}
                        print('Plotting')
                        
                        if nline == 'tau100':
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_tau100_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        else:
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        
                        if not use_all_dens:
                            #Dropping values with densities outside 5e5 XXXX ver como hago esto parametro
                            if tau != 16:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                            else:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                           
                            hc3n_model_nline['cond_sel_nrange1'] = (hc3n_model_nline['nH_u'] !=  8E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange1'] == True]
                            
                            hc3n_model_nline['cond_sel_nrange2'] = (hc3n_model_nline['nH_u'] !=  9E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange2'] == True]
                        
                            hc3n_model_nline['cond_sel_nrange3'] = (hc3n_model_nline['nH_u'] >5e7)  & (hc3n_model_nline['nH_u']  <1e8)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange3'] == False]
                            
                            hc3n_model_nline['cond_sel_nrange4'] = (hc3n_model_nline['nH_u'] >5e6)  & (hc3n_model_nline['nH_u']  <1e7)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange4'] == False]
                        
                        if nline == 'tau100':
                            # Nline from tau100
                            Nl_list_rad  = hc3n_model_nline['Nline_tau100_u'].unique() 
                        else:
                            # Original Nline from Xline *rnube*nH
                            Nl_list_rad  = hc3n_model_nline['Nline_u'].unique() 
                        
                        
                        # Selecting only dens below 1e7
                        hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] <= 5e7)
                        hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                        X_list_rad  = hc3n_model_nline['Xline_u'].unique() 
                        tau100_list_rad  = hc3n_model_nline['tau100'].unique()
                        nh_list_rad = np.sort(hc3n_model_nline['nH_u'].unique())
                        Nh2_list_rad = hc3n_model_nline['NH2_tau100_u'].unique()
                        Tdust_list_rad = np.sort(hc3n_model_nline['Tdust'].unique())
                        
                        print('\ttau100='+'%1.1f' % tau100_list_rad[0])
                        print('\tN(H2)='+'%1.1E' % Nh2_list_rad[0])
                        print('\tN(HC3N)='+'%1.1E' % Nl_list_rad[0])
                        print('\tX(HC3N)='+'%1.1E' % X_list_rad[0])
                   
                        
                            
                        if cont_map:
                            # Plotting continuos dustmap
                            hc3n_model_nline_sort_plot = hc3n_model_nline.sort_values(by=[xvar_p, yvar_p])
    
                            xi = np.linspace(xlimits[0], xlimits[1], 1000)
                            yi = np.linspace(ylimits[0], ylimits[1], 1000)
                            xi, yi = np.meshgrid(xi, yi)
                            rbf = scipy.interpolate.Rbf(hc3n_model_nline_sort_plot[xvar_p], hc3n_model_nline_sort_plot[yvar_p],
                                                        hc3n_model_nline_sort_plot['Tdust'],
                                                        function='linear', smooth=0.0)#20.0
                            zi = rbf(xi, yi)
                            #f = interpolate.interp2d(x,y,test,kind='cubic')
                            cmap_new = sns.light_palette('pale red', input="xkcd", as_cmap=True)
                            ax.contourf(xi, yi, zi,  cmap=cmap_new, origin='lower', levels=400, zorder=0.002)
                    
                        if nl == 1 and tau == 16:
                            nh_list_1e17 = nh_e17_list
                        else:
                            nh_list_1e17 = nh_e17_list
                        # Grouping by density
                        if line == 1E17:
                            nh_list_plot = nh_list_1e17
                        else:
                            if plot_all_dens:
                                nh_list_plot = nh_list_rad
                            else:
                                nh_list_plot = [5e5, 1e6, 2.5e6, 5e6, 1e7]
                        nh_text = [1e6, 5e6, 1e7]
                        
                        redpink     = sns.xkcd_palette(['red pink'])[0]
                        oblue       = sns.xkcd_palette(['ocean blue'])[0]
                        elime       = sns.xkcd_palette(['electric lime'])
                        melon       = sns.xkcd_palette(['melon'])[0]
                        aquamarine  = sns.xkcd_palette(['aquamarine'])[0]
                        aquagreen   = sns.xkcd_palette(['aqua green'])[0]
                        turqoise    = sns.xkcd_palette(['turquoise'])[0]
                        water       = sns.xkcd_palette(['water blue'])[0]
                        brown       =sns.xkcd_palette(['terracota'])[0]
                        purple      =sns.xkcd_palette(['purple'])[0]
                        orange      = sns.xkcd_palette(['blood orange'])[0]
                        green       = sns.xkcd_palette(['kelly green'])[0]
                        azure       = sns.xkcd_palette(['azure'])[0]
                        
                        #ncolors = {5e5: 'darkviolet', 1e6: 'dodgerblue', 2.5e6: 'cyan', 5e6: 'limegreen', }#1e7:'darkred'}
                        #ncolors = {5e5: 'darkviolet', 1e6: 'dodgerblue', 2.5e6: 'cyan', 5e6: 'limegreen', 1e7: 'xkcd:red pink'}#1e7:'darkred'}
                        # Plot only lines with no interpolation
                        if plot_only_conts:
                            fillcolors = False
                            
                            # Plotting by same nH density (i.e. changing Temperature)
                            desn_color_palete = sns.color_palette("Blues", len(nh_list_rad))
                            dens_ticklab = []
                            for dd, denss in enumerate(nh_list_rad):
                                hc3n_model_nline['cond_selnh'] = (hc3n_model_nline['nH_u'] ==  denss)
                                hc3n_model_nline_nH = hc3n_model_nline[hc3n_model_nline['cond_selnh'] == True]
                                # Sorting values for plotting
                                hc3n_model_nline_nH_plot = hc3n_model_nline_nH.sort_values(by=[xvar_p, yvar_p])
                                ax.plot(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = desn_color_palete[dd], linestyle =  Nline_linestyles[l], linewidth = 1.2, zorder=1/10.)
                                dens_ticklab.append(f'{denss/1e5:1.2f}')
                            dmaps_sns = LinearSegmentedColormap.from_list( 'dens_colors', desn_color_palete, N=len(desn_color_palete))
                            ax1 = axarr[0, 0].get_position().get_points().flatten()
                            ax2 = axarr[0, 2].get_position().get_points().flatten()
                            # Density colorbar
                            cbardens_ax = fig.add_axes([ax1[0], 0.90, ax2[2]-ax1[0], 0.015])
                            dens_list = nh_list_rad
                                
                            #tick_locs = (np.arange(len(dens_list)+1) + 0.(5)*(len(dens_list)-1)/len(dens_list))
                            tick_locs = np.arange(0,np.max(dens_list), np.max(dens_list)/(len(dens_list)+1))
                            tick_locs2 = []
                            for t, tick in enumerate(tick_locs):
                                if t != 4:
                                    if t < 4:
                                        tick_locs2.append(tick+np.max(dens_list)/(len(dens_list)+1)/2)
                                    else:
                                        tick_locs2.append(tick-np.max(dens_list)/(len(dens_list)+1)/2)
                            dens_norm_sns = mpl.colors.BoundaryNorm(tick_locs, dmaps_sns.N)
                            cbdens = mpl.colorbar.ColorbarBase(cbardens_ax, cmap=dmaps_sns,
                                                    norm=dens_norm_sns,
                                                    #boundaries=tick_locs,
                                                    spacing='uniform',
                                                    orientation='horizontal',
                                                    extend='neither',
                                                    ticks=tick_locs2
                                                    )
                            
                            cbdens.ax.set_xticklabels(dens_ticklab)
                            cbdens.ax.tick_params(axis='x', direction='in',labeltop=True, labelbottom=False, bottom=False, top=True, labelsize='8', zorder=30)
                            cbdens.set_label(r'$n(\rm{H}_2)$ ($10^5\,$cm$^{-3}$)', labelpad=-40)
                                    
                            # Plotting by same Tdust (i.e. changing density) 
                            tdust_color_palete = sns.color_palette("Reds", len(Tdust_list_rad))
                            for tt, tdust in enumerate(Tdust_list_rad):
                                hc3n_model_nline['cond_seltd'] = (hc3n_model_nline['Tdust'] ==  tdust)
                                hc3n_model_nline_td = hc3n_model_nline[hc3n_model_nline['cond_seltd'] == True]
                                # Sorting values for plotting
                                hc3n_model_nline_td_plot = hc3n_model_nline_td.sort_values(by=[xvar_p, yvar_p])
                                if tdust == 200:
                                    lstyle = '--'
                                else:
                                    lstyle = '-'
                                ax.plot(hc3n_model_nline_td_plot[xvar_p], hc3n_model_nline_td_plot[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = tdust_color_palete[tt], linestyle =  lstyle, linewidth = 1.2, zorder=1/10.)
                            # 
                        else:
                            ncolors = {1e4: 'darkviolet', 5e4: 'dodgerblue', 1e5: 'cyan', 5e5: 'limegreen', 1e6: 'xkcd:red pink', 5e6: 'xkcd:terracota', 1e7: 'xkcd:blood orange'}
                            for dd, dens in enumerate(nh_list_plot):
                                hc3n_model_nline['cond_selnh'] = (hc3n_model_nline['nH_u'] ==  dens)
                                hc3n_model_nline_nH = hc3n_model_nline[hc3n_model_nline['cond_selnh'] == True]
                                
                                # Sorting values for interpolation and plotting
                                hc3n_model_nline_nH_plot = hc3n_model_nline_nH.sort_values(by=[xvar_p, yvar_p])
                                if line == 1E17:
                                    smooth = 0.002
                                else:
                                    smooth = 0.01
                                # Interpolation
                                #nH_interp = interpolate.interp1d(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], fill_value='extrapolate')
                                # Splines
                                tck = interpolate.splrep(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], s=smooth)
                                xvals_nH = np.linspace(xlimits[0], xlimits[1],1000)
                                yvals_nH = interpolate.splev(xvals_nH, tck, der=0)
                                
                                if dens in nh_list_1e17:
                                    fillbtw_lines[line][dens] = {'xvals': xvals_nH, 'yvals': yvals_nH}
                                # Plotting nH color
                                color_dens = True
                                if color_dens:
                                    if dens in ncolors.keys():
                                        ax.plot(xvals_nH, yvals_nH,
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = ncolors[dens], linestyle = Nline_linestyles[l], linewidth = 1.2, zorder=1/10.)
                                        plot_true_values = False
                                        if plot_true_values:
                                            # Plotting true values without interp to compare
                                            ax.plot(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p],
                                                    marker='None', markerfacecolor ='None', markersize = 4,
                                                    color = 'k', linestyle = ':', linewidth = 1.2, zorder=1/10.)
                                    else:
                                        if plot_all_dens:
                                            ax.plot(xvals_nH, yvals_nH,
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = 'k', linestyle = ':', linewidth = 0.65, zorder=1/10.)
                                else:
                                    ax.plot(xvals_nH, yvals_nH,
                                        marker='None', markerfacecolor ='None', markersize = 4,
                                        color = 'k', linestyle = Nline_linestyles[l], linewidth = 0.5, zorder=1/10.)
                                
                                if labelling:
                                    if dens in nh_text:
                                        if r==0:
                                            if nl ==0:
                                                if dens == 1e6:
                                                    xpos = 1
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 1:
                                                if dens == 1e7:
                                                    xpos = 2
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                            elif nl == 2:
                                                if dens == 1e7:
                                                    xpos = 2
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                        elif r==1:
                                            if nl ==0:
                                                if dens == 1e6:
                                                    xpos = 1.5
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 1:
                                                if dens == 1e6:
                                                    xpos = 1.3
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 2:
                                                if dens == 1e6:
                                                    xpos = 1.1
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
    
                            if line != 1E17:
                                # Grouping by tdust
                                # I use a dictionary to group all densities given a Temperature.
                                # Then I use this interpolated curves to color the space between them
                                curve_temps_dictio = {}
                                # For temperatures i use all the data available i.e. hc3n_model_radtauNline
                                if use_all_temps:
                                    temp_list_plot = np.sort(Tdust_list_rad)
                                else:
                                    temp_list_plot = np.arange(100, 650, 50)
                                colors_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=False)
                                for tind, tdus in enumerate(temp_list_plot):
                                    print('\t\t'+str(tdus))
                                    hc3n_model_radtauNline['cond_seltdust'] = (hc3n_model_radtauNline['Tdust'] ==  tdus)
                                    hc3n_model_nline_Td = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_seltdust'] == True]
                                    if integrated == True:
                                        xvar_p = xv_pre +'integ_'+ j_ratio
                                        yvar_p = 'v0_v7_integ_' + vib_ratio
                                    else:
                                        xvar_p = xv_pre +'peak_'+ j_ratio
                                        yvar_p = 'v0_v7_peak_' + vib_ratio
                                    hc3n_model_nline_Td_plot = hc3n_model_nline_Td.sort_values(by=[yvar_p, xvar_p])
                                    
                                    tdust_interp = interpolate.interp1d(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p], fill_value='extrapolate')
                                    
                                    xvals = np.linspace(xlimits[0], xlimits[1],100)
                                    yvals = tdust_interp(xvals)
                                    curve_temps_dictio[tdus] = {'xvals': xvals, 'yvals': yvals, 'Tdust': tdus, 'Color': colors_tdust[tind], 'ind': tind}
                                    if plot_linetemps:
                                        if tdus in [125, 400, 450, 500, 550, 600]:
                                            if tdus ==350:
                                                temp_color = 'lime'
                                            elif tdus == 400:
                                                temp_color = 'b'
                                            elif tdus == 450:
                                                temp_color = 'purple'
                                            elif tdus == 500:
                                                temp_color = 'g'
                                            elif tdus == 550:
                                                temp_color = 'r'
                                            elif tdus == 600:
                                                temp_color = 'yellow'
                                            else:
                                                temp_color = 'k'
                                            ax.plot(xvals, yvals,
                                                    marker='None', markerfacecolor ='None', markersize = 4,
                                                    color = temp_color, linestyle = '-', linewidth = 0.5, zorder=2/10.)
                                    
                                    # Plotting T color
                                    dens_color = [1e6, 5e6, 1e7]
                                    dcolors = ['b', 'g', 'r']
                                    ax.plot(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p],
                                            marker='None', markerfacecolor ='None', markersize = 4,
                                            color = 'k', linestyle = 'None', linewidth = 0.5, zorder=1/10.)
                                    for d, dds in enumerate(dens_color):
                                        hc3n_model_nline_Td['cond_selnH'] = (hc3n_model_nline_Td['nH_u'] ==  dds)
                                        hc3n_model_nline_TdnH = hc3n_model_nline_Td[hc3n_model_nline_Td['cond_selnH'] == True]
                                        ax.plot(hc3n_model_nline_TdnH[xvar_p], hc3n_model_nline_TdnH[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = dcolors[d], linestyle =  '-', linewidth = 0.5, zorder=3/10.)
                                
                                od_curve_temps_dictio = collections.OrderedDict(sorted(curve_temps_dictio.items()))
                                #od_curve_temps_dictio = {k: curve_temps_dictio[k] for k in sorted(curve_temps_dictio)}
                                for item, next_item in utiles.iterate(od_curve_temps_dictio.items()):
                                    if next_item != 0:
                                        # to check if iteration is ok
                                        # print(item[1]['ind'], item[1]['Tdust'], next_item[1]['ind'], next_item[1]['Tdust'])
                                        ax.fill_between(item[1]['xvals'], item[1]['yvals'],
                                                        next_item[1]['yvals'], facecolor=item[1]['Color'], zorder=0)
                
                if fillcolors:
                    for density in nh_list_1e17:
                        fill_color = ncolors[density]
                        ax.fill_between(fillbtw_lines[5E16][density]['xvals'],
                                        fillbtw_lines[5E16][density]['yvals'], fillbtw_lines[1E17][density]['yvals'],
                                        facecolor = fill_color, zorder = 0.001, alpha=0.25)
            
                fontsize = 7
                plot_observed_ratios_NGC1068(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                             xvar, yvar, integrated, print_ylabel, print_xlabel,
                                             fontsize=fontsize, color=color_obs, man_obs=man_obs
                                             )
                    
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
                                                 
                if r == (len(chunks_row)-1):
                    #ax.set_xlabel(xlabel)
                    ax.tick_params(axis='x', labeltop=False, labelbottom=True, labelsize='8',zorder=30)
                else:
                    ax.tick_params(axis='x', labeltop=False, labelbottom=False, labelsize='8', zorder=30)
                ax.tick_params(axis='both', zorder=30)
                print('-------------')
        # Position of the axes
        ax1 = axarr[0, 1].get_position().get_points().flatten()
        ax2 = axarr[2, 1].get_position().get_points().flatten()
        # Temperature colorbar
        if use_all_temps:
            temp_list_plot = np.sort(Tdust_list_rad)#np.array([100, 125, 150, 175, 200, 250, 300, 400])#
        else:
            temp_list_plot = np.arange(100, 650, 50)
            
        #Tdust_list_rad_sort = sorted(Tdust_list_rad)
        cmap_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=True)
        cbar_ax = fig.add_axes([0.92, ax2[1], 0.025, ax1[3]-ax2[1]])
        #temp_list = np.arange(100, 700, 100)
        temp_list = np.arange(150, 600, 50)
        temp_norm_new = mpl.colors.BoundaryNorm(temp_list_plot, cmap_tdust.N)
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_tdust,
                                norm=temp_norm_new,
                                spacing='uniform',
                                orientation='vertical',
                                extend='neither',
                                ticks=temp_list)
        cb.set_label(r'$T_{\rm{dust}}$ (K)')
        cb.ax.tick_params(axis='y', direction='in')
        # This is for plotting horizontal lines on the cbar
        #cb.ax.plot([0, 1], [0.2, 0.2], 'b')
        #cb.ax.plot([0, 1], [0.4, 0.4], 'g')
        #cb.ax.plot([0, 1], [0.6, 0.6], 'r')
        #for j, lab in enumerate(Tdust_list_rad):
        #    lab_int = int(lab)
        #    cb.ax.text(1, (j) / (len(Tdust_list_rad)-1), lab_int, ha='center', va='center')
        ylabel = 'v=0 ('+vib_ratio+') / v7=1 ('+vib_ratio+')'
        fig.text(0.075,0.5, ylabel, ha="center", va="center", rotation=90)
        
        if dens_colorbar:
            #dens_colors = ['darkviolet', 'dodgerblue', 'cyan', 'limegreen', 'darkred']
            dens_colors = ['darkviolet', 'dodgerblue',  'cyan', 'limegreen', 'xkcd:red pink']

            dmaps_sns = LinearSegmentedColormap.from_list( 'dens_colors', dens_colors, N=len(dens_colors))
            # Position of the axes
            ax1 = axarr[0, 0].get_position().get_points().flatten()
            ax2 = axarr[0, 2].get_position().get_points().flatten()
            # Density colorbar
            cbardens_ax = fig.add_axes([ax1[0], 0.90, ax2[2]-ax1[0], 0.015])
            dens_list = [0.5, 1, 2.5, 5, 10] 
            #dmaps = matplotlib.cm.get_cmap('winter')
            #dmaps_sns = matplotlib.colors.LinearSegmentedColormap.from_list('cblues', dens_sns_map, N=len(dens_colors)+1)
            #dens_list2 = np.array([1, 2, 3, 4, 5])-0.5
            #dens_norm = mpl.colors.BoundaryNorm(dens_list, dmaps.N)
                
            #tick_locs = (np.arange(len(dens_list)+1) + 0.(5)*(len(dens_list)-1)/len(dens_list))
            tick_locs = np.arange(0,np.max(dens_list), np.max(dens_list)/(len(dens_list)+1))
            tick_locs2 = []
            for t, tick in enumerate(tick_locs):
                if t != 4:
                    if t < 4:
                        tick_locs2.append(tick+np.max(dens_list)/(len(dens_list)+1)/2)
                    else:
                        tick_locs2.append(tick-np.max(dens_list)/(len(dens_list)+1)/2)
            dens_norm_sns = mpl.colors.BoundaryNorm(tick_locs, dmaps_sns.N)
            cbdens = mpl.colorbar.ColorbarBase(cbardens_ax, cmap=dmaps_sns,
                                    norm=dens_norm_sns,
                                    #boundaries=tick_locs,
                                    spacing='uniform',
                                    orientation='horizontal',
                                    extend='neither',
                                    ticks=tick_locs2
                                    )
            cbdens.ax.set_xticklabels(['0.5', '1.0', '2.5', '5.0', '10'])
            cbdens.ax.tick_params(axis='x', direction='in',labeltop=True, labelbottom=False, bottom=False, top=True, labelsize='8', zorder=30)
            cbdens.set_label(r'$n(\rm{H}_2)$ ($10^6\,$cm$^{-3}$)', labelpad=-40)
        
        if write_temps:
            # Plotting temperature indicators (by hand!!)
            # T=200
            axarr[0, 0].text(1.25, 3.2,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            #axarr[1, 0].text(0.7, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            #axarr[2, 0].text(0.6, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 1].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 2].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 2].text(2.25, 3.55,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 2].text(2.2, 3.6,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            
        #fig_name = 'TEST_HC3N_models_check'
        for formato in fig_format:
            fig.savefig(out_dir+fig_name+suffix_name+'_'+j_ratio+'.'+formato, bbox_inches='tight', transparent=False, dpi=400)



def HC3N_modelsub_new_NGC1068_chek2(rnube_list, hc3n_model, integrated, dens_profile, results_path,
                                   hc3n_selection, tau_selection, 
                                   color_nh, color_obs, 
                                   xvar, yvar, out_dir, fig_format, vib_ratio, j_ratio, obs_ratios_dict, suffix_name='',
                                  nline='tau100', plot_lines = False, plot_only_conts=True,
                                  columnas = 3, write_temps = False, fillcolors = False,
                                  plot_linetemps=False, plot_all_dens=False,
                                  cont_map = False, use_all_dens =False, use_all_temps=False,
                                  labelling = False, nh_e17_list = [1e6, 2.5e6],
                                  dens_colorbar = False, fig_name = 'test_model', man_obs =[]):
    """
    Plotting models parameter space
    vib_ratio = '16-15'
    j_ratio = '24-23_16-15'
    """
    import seaborn as sns
    import matplotlib.ticker as ticker
    import matplotlib as mpl
    profile_path = 'r_'+'%1.1f' % dens_profile
    
    
    integrated = True
    yvar = 'v0_v7_1615'
    use_all_dens = True
    # Selecting by radius
    for rad in rnube_list:
        filas = len(tau_selection)
        #columnas = 2
        fig, axarr = plt.subplots(filas, columnas, sharex=False, figsize=(10,8))
        fig.subplots_adjust(hspace=0)
        hc3n_model['cond_sel_r'] = (hc3n_model['rnube_pc_u']==rad)
        hc3n_model_rad1 = hc3n_model[hc3n_model['cond_sel_r'] == True]
        chunks_row = []
        for tt, tau in enumerate(tau_selection):
            chunks_row.append([tau_selection[tt],tau_selection[tt], tau_selection[tt]])
        #chunks_row = [[tau_selection[0],tau_selection[0], tau_selection[0]],
        #              [tau_selection[1],tau_selection[1], tau_selection[1]],
        #              [tau_selection[2],tau_selection[2], tau_selection[2]]]
        # Plotting each row
        # one row for each ratio
        for r,row_fig in enumerate(chunks_row):
            # Only bottom subplots with X label
            if r ==(len(chunks_row)-1):
                print_xlabel = True
            else:
                print_xlabel = False
            # Selecting by tau100
            for nl, tau in enumerate(row_fig):
                hc3n_model_rad1['cond_sel_tau'] = (hc3n_model_rad1['tau100_u']==tau)
                hc3n_model_radtau = hc3n_model_rad1[hc3n_model_rad1['cond_sel_tau'] == True]
                if filas == 1:
                    ax = axarr[nl]
                else:
                    ax = axarr[r, nl]
                # Only left subplots with Y label
                if nl == 0:
                    print_ylabel = False
                    #xvar = 'ratio_v0'
                    xvar = 'v0_2416'
                    xv_pre = 'v=0_'
                elif nl == 1:
                    print_ylabel = False
                    xvar = 'ratio_v7'
                    xvar = 'v7_2416'
                    xv_pre = 'v7=1_'
                elif nl == 2:
                    print_ylabel = False
                    xvar = 'ratio_v6'
                    xvar = 'v6_2416'
                    xv_pre = 'v6=1_'
                # Limits
                if xvar == 'v0_2416':
                    xlimits = [0.1,1.5]
                elif xvar == 'v7_2416':
                    xlimits =  [0.5, 5]
                elif xvar == 'v6_2416':
                    xlimits =  [0.5, 5]
                if yvar == 'v0_v7_1615':
                    ylimits = [0,15]
                elif yvar == 'v0_v7_2423':
                    ylimits = [0,15]
                    
                # Writting model parameters                            
                # tau100
                ax.text(0.925, 0.91, r'$\boldsymbol{\tau_{100}='+'%1.d' % tau+r'}$', ha='right', va='center', color='k',
                                rotation='horizontal', fontsize=6, zorder =11,transform = ax.transAxes)
                # Selecting by Xline or Nline
                Nline_linestyles = ['-', '--']
                if integrated == True:
                    xvar_p = xvar
                    yvar_p = yvar
                else:
                    xvar_p = xv_pre +'peak_'+ j_ratio
                    yvar_p = 'v0_v7_peak_' + vib_ratio
                fillbtw_lines = {}
                for l, line in enumerate(hc3n_selection):
                    if line > 1:
                        if nline == 'tau100':
                            selection = 'Nline_tau100'
                        else:
                            selection = 'Nline'
                        sel_lab = 'N'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.95)& (hc3n_model_radtau[selection]/line < 1.05)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    else:
                        selection = 'Xline'
                        sel_lab = 'X'
                        hc3n_model_radtau['cond_sel_'+selection] = (hc3n_model_radtau[selection]/line > 0.995)& (hc3n_model_radtau[selection]/line < 1.005)
                        hc3n_model_radtauNline = hc3n_model_radtau[hc3n_model_radtau['cond_sel_'+selection] == True]
                    
                    # Dropping Tdust = 125
                    # hc3n_model_radtauNline['cond_sel_t1'] = (hc3n_model_radtauNline['Tdust'] !=  125)
                    # hc3n_model_radtauNline = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_sel_t1'] == True]
                    hc3n_model_nline = hc3n_model_radtauNline
                    if len(hc3n_model_nline) != 0:
                        # just continue if selected models is not epty
                        fillbtw_lines[line] = {}
                        print('Plotting')
                        
                        if nline == 'tau100':
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_tau100_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        else:
                            hc3n_model_nline = hc3n_model_nline.drop_duplicates(subset=['Xline_u', 'Nline_u', 'Ndust_u', 'tau100', 'Tdust', 'nH_u'], keep="last")
                        
                        if not use_all_dens:
                            #Dropping values with densities outside 5e5 XXXX ver como hago esto parametro
                            if tau != 16:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                            else:
                                hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] >=  5e5)& (hc3n_model_nline['nH_u'] <= 1e8)
                                hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                           
                            hc3n_model_nline['cond_sel_nrange1'] = (hc3n_model_nline['nH_u'] !=  8E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange1'] == True]
                            
                            hc3n_model_nline['cond_sel_nrange2'] = (hc3n_model_nline['nH_u'] !=  9E5)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange2'] == True]
                        
                            hc3n_model_nline['cond_sel_nrange3'] = (hc3n_model_nline['nH_u'] >5e7)  & (hc3n_model_nline['nH_u']  <1e8)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange3'] == False]
                            
                            hc3n_model_nline['cond_sel_nrange4'] = (hc3n_model_nline['nH_u'] >5e6)  & (hc3n_model_nline['nH_u']  <1e7)
                            hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange4'] == False]
                        
                        if nline == 'tau100':
                            # Nline from tau100
                            Nl_list_rad  = hc3n_model_nline['Nline_tau100_u'].unique() 
                        else:
                            # Original Nline from Xline *rnube*nH
                            Nl_list_rad  = hc3n_model_nline['Nline_u'].unique() 
                        
                        
                        # Selecting only dens below 1e7
                        hc3n_model_nline['cond_sel_nrange'] = (hc3n_model_nline['nH_u'] <= 5e7)
                        hc3n_model_nline = hc3n_model_nline[hc3n_model_nline['cond_sel_nrange'] == True]
                        X_list_rad  = hc3n_model_nline['Xline_u'].unique() 
                        tau100_list_rad  = hc3n_model_nline['tau100'].unique()
                        nh_list_rad = np.sort(hc3n_model_nline['nH_u'].unique())
                        Nh2_list_rad = hc3n_model_nline['NH2_tau100_u'].unique()
                        Tdust_list_rad = np.sort(hc3n_model_nline['Tdust'].unique())
                        
                        print('\ttau100='+'%1.1f' % tau100_list_rad[0])
                        print('\tN(H2)='+'%1.1E' % Nh2_list_rad[0])
                        print('\tN(HC3N)='+'%1.1E' % Nl_list_rad[0])
                        print('\tX(HC3N)='+'%1.1E' % X_list_rad[0])
                   
                        
                            
                        if cont_map:
                            # Plotting continuos dustmap
                            hc3n_model_nline_sort_plot = hc3n_model_nline.sort_values(by=[xvar_p, yvar_p])
    
                            xi = np.linspace(xlimits[0], xlimits[1], 1000)
                            yi = np.linspace(ylimits[0], ylimits[1], 1000)
                            xi, yi = np.meshgrid(xi, yi)
                            rbf = scipy.interpolate.Rbf(hc3n_model_nline_sort_plot[xvar_p], hc3n_model_nline_sort_plot[yvar_p],
                                                        hc3n_model_nline_sort_plot['Tdust'],
                                                        function='linear', smooth=0.0)#20.0
                            zi = rbf(xi, yi)
                            #f = interpolate.interp2d(x,y,test,kind='cubic')
                            cmap_new = sns.light_palette('pale red', input="xkcd", as_cmap=True)
                            ax.contourf(xi, yi, zi,  cmap=cmap_new, origin='lower', levels=400, zorder=0.002)
                    
                        if nl == 1 and tau == 16:
                            nh_list_1e17 = nh_e17_list
                        else:
                            nh_list_1e17 = nh_e17_list
                        # Grouping by density
                        if line == 1E17:
                            nh_list_plot = nh_list_1e17
                        else:
                            if plot_all_dens:
                                nh_list_plot = nh_list_rad
                            else:
                                nh_list_plot = [5e5, 1e6, 2.5e6, 5e6, 1e7]
                        nh_text = [1e6, 5e6, 1e7]
                        
                        # Plot only lines with no interpolation
                        if plot_only_conts:
                            fillcolors = False
                            
                            # Plotting by same nH density (i.e. changing Temperature)
                            desn_color_palete = sns.color_palette("Blues", len(nh_list_rad))
                            dens_ticklab = []
                            for dd, denss in enumerate(nh_list_rad):
                                hc3n_model_nline['cond_selnh'] = (hc3n_model_nline['nH_u'] ==  denss)
                                hc3n_model_nline_nH = hc3n_model_nline[hc3n_model_nline['cond_selnh'] == True]
                                # Sorting values for plotting
                                hc3n_model_nline_nH_plot = hc3n_model_nline_nH.sort_values(by=[xvar_p, yvar_p])
                                ax.plot(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = desn_color_palete[dd], linestyle =  Nline_linestyles[l], linewidth = 1.2, zorder=1/10.)
                                dens_ticklab.append(f'{denss/1e5:1.2f}')
                            dmaps_sns = LinearSegmentedColormap.from_list( 'dens_colors', desn_color_palete, N=len(desn_color_palete))
                            ax1 = axarr[0, 0].get_position().get_points().flatten()
                            ax2 = axarr[0, 2].get_position().get_points().flatten()
                            # Density colorbar
                            cbardens_ax = fig.add_axes([ax1[0], 0.90, ax2[2]-ax1[0], 0.015])
                            dens_list = nh_list_rad
                            tick_locs = np.arange(0,np.max(dens_list), np.max(dens_list)/(len(dens_list)+1))
                            tick_locs2 = []
                            for t, tick in enumerate(tick_locs):
                                if t != 4:
                                    if t < 4:
                                        tick_locs2.append(tick+np.max(dens_list)/(len(dens_list)+1)/2)
                                    else:
                                        tick_locs2.append(tick-np.max(dens_list)/(len(dens_list)+1)/2)
                            dens_norm_sns = mpl.colors.BoundaryNorm(tick_locs, dmaps_sns.N)
                            cbdens = mpl.colorbar.ColorbarBase(cbardens_ax, cmap=dmaps_sns,
                                                    norm=dens_norm_sns,
                                                    #boundaries=tick_locs,
                                                    spacing='uniform',
                                                    orientation='horizontal',
                                                    extend='neither',
                                                    ticks=tick_locs2
                                                    )
                            
                            cbdens.ax.set_xticklabels(dens_ticklab)
                            cbdens.ax.tick_params(axis='x', direction='in',labeltop=True, labelbottom=False, bottom=False, top=True, labelsize='8', zorder=30)
                            cbdens.set_label(r'$n(\rm{H}_2)$ ($10^5\,$cm$^{-3}$)', labelpad=-40)
                                    
                            # Plotting by same Tdust (i.e. changing density) 
                            tdust_color_palete = sns.color_palette("Reds", len(Tdust_list_rad))
                            for tt, tdust in enumerate(Tdust_list_rad):
                                hc3n_model_nline['cond_seltd'] = (hc3n_model_nline['Tdust'] ==  tdust)
                                hc3n_model_nline_td = hc3n_model_nline[hc3n_model_nline['cond_seltd'] == True]
                                # Sorting values for plotting
                                hc3n_model_nline_td_plot = hc3n_model_nline_td.sort_values(by=[xvar_p, yvar_p])
                                if tdust == 200:
                                    lstyle = '--'
                                else:
                                    lstyle = '-'
                                ax.plot(hc3n_model_nline_td_plot[xvar_p], hc3n_model_nline_td_plot[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = tdust_color_palete[tt], linestyle =  lstyle, linewidth = 1.2, zorder=1/10.)
                        else:
                            ncolors = {1e4: 'darkviolet', 5e4: 'dodgerblue', 1e5: 'cyan', 5e5: 'limegreen', 1e6: 'xkcd:red pink', 5e6: 'xkcd:terracota', 1e7: 'xkcd:blood orange'}
                            for dd, dens in enumerate(nh_list_plot):
                                hc3n_model_nline['cond_selnh'] = (hc3n_model_nline['nH_u'] ==  dens)
                                hc3n_model_nline_nH = hc3n_model_nline[hc3n_model_nline['cond_selnh'] == True]
                                
                                # Sorting values for interpolation and plotting
                                hc3n_model_nline_nH_plot = hc3n_model_nline_nH.sort_values(by=[xvar_p, yvar_p])
                                if line == 1E17:
                                    smooth = 0.002
                                else:
                                    smooth = 0.01
                                # Interpolation
                                #nH_interp = interpolate.interp1d(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], fill_value='extrapolate')
                                # Splines
                                tck = interpolate.splrep(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p], s=smooth)
                                xvals_nH = np.linspace(xlimits[0], xlimits[1],1000)
                                yvals_nH = interpolate.splev(xvals_nH, tck, der=0)
                                
                                if dens in nh_list_1e17:
                                    fillbtw_lines[line][dens] = {'xvals': xvals_nH, 'yvals': yvals_nH}
                                # Plotting nH color
                                color_dens = True
                                if color_dens:
                                    if dens in ncolors.keys():
                                        ax.plot(xvals_nH, yvals_nH,
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = ncolors[dens], linestyle = Nline_linestyles[l], linewidth = 1.2, zorder=1/10.)
                                        plot_true_values = False
                                        if plot_true_values:
                                            # Plotting true values without interp to compare
                                            ax.plot(hc3n_model_nline_nH_plot[xvar_p], hc3n_model_nline_nH_plot[yvar_p],
                                                    marker='None', markerfacecolor ='None', markersize = 4,
                                                    color = 'k', linestyle = ':', linewidth = 1.2, zorder=1/10.)
                                    else:
                                        if plot_all_dens:
                                            ax.plot(xvals_nH, yvals_nH,
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = 'k', linestyle = ':', linewidth = 0.65, zorder=1/10.)
                                else:
                                    ax.plot(xvals_nH, yvals_nH,
                                        marker='None', markerfacecolor ='None', markersize = 4,
                                        color = 'k', linestyle = Nline_linestyles[l], linewidth = 0.5, zorder=1/10.)
                                
                                if labelling:
                                    if dens in nh_text:
                                        if r==0:
                                            if nl ==0:
                                                if dens == 1e6:
                                                    xpos = 1
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 1:
                                                if dens == 1e7:
                                                    xpos = 2
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                            elif nl == 2:
                                                if dens == 1e7:
                                                    xpos = 2
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e7', align=True, angle=False)
                                        elif r==1:
                                            if nl ==0:
                                                if dens == 1e6:
                                                    xpos = 1.5
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 1:
                                                if dens == 1e6:
                                                    xpos = 1.3
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
                                            elif nl == 2:
                                                if dens == 1e6:
                                                    xpos = 1.1
                                                    ypos = interpolate.splev(xpos, tck, der=0)
                                                    labelLine(xvals_nH,yvals_nH, xpos, ypos+ypos*0.01, ax, color='k', label='nH=1e6', align=True, angle=False)
    
                            if line != 1E17:
                                # Grouping by tdust
                                # I use a dictionary to group all densities given a Temperature.
                                # Then I use this interpolated curves to color the space between them
                                curve_temps_dictio = {}
                                # For temperatures i use all the data available i.e. hc3n_model_radtauNline
                                if use_all_temps:
                                    temp_list_plot = np.sort(Tdust_list_rad)
                                else:
                                    temp_list_plot = np.arange(100, 650, 50)
                                colors_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=False)
                                for tind, tdus in enumerate(temp_list_plot):
                                    print('\t\t'+str(tdus))
                                    hc3n_model_radtauNline['cond_seltdust'] = (hc3n_model_radtauNline['Tdust'] ==  tdus)
                                    hc3n_model_nline_Td = hc3n_model_radtauNline[hc3n_model_radtauNline['cond_seltdust'] == True]
                                    if integrated == True:
                                        xvar_p = xv_pre +'integ_'+ j_ratio
                                        yvar_p = 'v0_v7_integ_' + vib_ratio
                                    else:
                                        xvar_p = xv_pre +'peak_'+ j_ratio
                                        yvar_p = 'v0_v7_peak_' + vib_ratio
                                    hc3n_model_nline_Td_plot = hc3n_model_nline_Td.sort_values(by=[yvar_p, xvar_p])
                                    
                                    tdust_interp = interpolate.interp1d(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p], fill_value='extrapolate')
                                    
                                    xvals = np.linspace(xlimits[0], xlimits[1],100)
                                    yvals = tdust_interp(xvals)
                                    curve_temps_dictio[tdus] = {'xvals': xvals, 'yvals': yvals, 'Tdust': tdus, 'Color': colors_tdust[tind], 'ind': tind}
                                    if plot_linetemps:
                                        if tdus in [125, 400, 450, 500, 550, 600]:
                                            if tdus ==350:
                                                temp_color = 'lime'
                                            elif tdus == 400:
                                                temp_color = 'b'
                                            elif tdus == 450:
                                                temp_color = 'purple'
                                            elif tdus == 500:
                                                temp_color = 'g'
                                            elif tdus == 550:
                                                temp_color = 'r'
                                            elif tdus == 600:
                                                temp_color = 'yellow'
                                            else:
                                                temp_color = 'k'
                                            ax.plot(xvals, yvals,
                                                    marker='None', markerfacecolor ='None', markersize = 4,
                                                    color = temp_color, linestyle = '-', linewidth = 0.5, zorder=2/10.)
                                    
                                    # Plotting T color
                                    dens_color = [1e6, 5e6, 1e7]
                                    dcolors = ['b', 'g', 'r']
                                    ax.plot(hc3n_model_nline_Td_plot[xvar_p], hc3n_model_nline_Td_plot[yvar_p],
                                            marker='None', markerfacecolor ='None', markersize = 4,
                                            color = 'k', linestyle = 'None', linewidth = 0.5, zorder=1/10.)
                                    for d, dds in enumerate(dens_color):
                                        hc3n_model_nline_Td['cond_selnH'] = (hc3n_model_nline_Td['nH_u'] ==  dds)
                                        hc3n_model_nline_TdnH = hc3n_model_nline_Td[hc3n_model_nline_Td['cond_selnH'] == True]
                                        ax.plot(hc3n_model_nline_TdnH[xvar_p], hc3n_model_nline_TdnH[yvar_p],
                                                marker='None', markerfacecolor ='None', markersize = 4,
                                                color = dcolors[d], linestyle =  '-', linewidth = 0.5, zorder=3/10.)
                                
                                od_curve_temps_dictio = collections.OrderedDict(sorted(curve_temps_dictio.items()))
                                #od_curve_temps_dictio = {k: curve_temps_dictio[k] for k in sorted(curve_temps_dictio)}
                                for item, next_item in utiles.iterate(od_curve_temps_dictio.items()):
                                    if next_item != 0:
                                        # to check if iteration is ok
                                        # print(item[1]['ind'], item[1]['Tdust'], next_item[1]['ind'], next_item[1]['Tdust'])
                                        ax.fill_between(item[1]['xvals'], item[1]['yvals'],
                                                        next_item[1]['yvals'], facecolor=item[1]['Color'], zorder=0)
                
                if fillcolors:
                    for density in nh_list_1e17:
                        fill_color = ncolors[density]
                        ax.fill_between(fillbtw_lines[5E16][density]['xvals'],
                                        fillbtw_lines[5E16][density]['yvals'], fillbtw_lines[1E17][density]['yvals'],
                                        facecolor = fill_color, zorder = 0.001, alpha=0.25)
            
                fontsize = 7
                color_obs = 'k'
                plot_observed_ratios_NGC1068_new(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                 xvar, yvar, integrated, print_ylabel, print_xlabel, fontsize, ylimits, xlimits, color=color_obs
                                 )
                
                ax.set_xlim(xlimits)
                ax.set_ylim(ylimits)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
                                                 
                if r == (len(chunks_row)-1):
                    #ax.set_xlabel(xlabel)
                    ax.tick_params(axis='x', labeltop=False, labelbottom=True, labelsize='8',zorder=30)
                else:
                    ax.tick_params(axis='x', labeltop=False, labelbottom=False, labelsize='8', zorder=30)
                ax.tick_params(axis='both', zorder=30)
                print('-------------')
        # Position of the axes
        ax1 = axarr[0, 1].get_position().get_points().flatten()
        ax2 = axarr[2, 1].get_position().get_points().flatten()
        # Temperature colorbar
        if use_all_temps:
            temp_list_plot = np.sort(Tdust_list_rad)#np.array([100, 125, 150, 175, 200, 250, 300, 400])#
        else:
            temp_list_plot = np.arange(100, 650, 50)
            
        #Tdust_list_rad_sort = sorted(Tdust_list_rad)
        cmap_tdust = sns.light_palette('pale red', input="xkcd", n_colors=len(temp_list_plot), as_cmap=True)
        cbar_ax = fig.add_axes([0.92, ax2[1], 0.025, ax1[3]-ax2[1]])
        #temp_list = np.arange(100, 700, 100)
        temp_list = np.arange(150, 600, 50)
        temp_norm_new = mpl.colors.BoundaryNorm(temp_list_plot, cmap_tdust.N)
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_tdust,
                                norm=temp_norm_new,
                                spacing='uniform',
                                orientation='vertical',
                                extend='neither',
                                ticks=temp_list)
        cb.set_label(r'$T_{\rm{dust}}$ (K)')
        cb.ax.tick_params(axis='y', direction='in')
        # This is for plotting horizontal lines on the cbar
        #cb.ax.plot([0, 1], [0.2, 0.2], 'b')
        #cb.ax.plot([0, 1], [0.4, 0.4], 'g')
        #cb.ax.plot([0, 1], [0.6, 0.6], 'r')
        #for j, lab in enumerate(Tdust_list_rad):
        #    lab_int = int(lab)
        #    cb.ax.text(1, (j) / (len(Tdust_list_rad)-1), lab_int, ha='center', va='center')
        ylabel = 'v=0 ('+vib_ratio+') / v7=1 ('+vib_ratio+')'
        fig.text(0.075,0.5, ylabel, ha="center", va="center", rotation=90)
        
        if dens_colorbar:
            #dens_colors = ['darkviolet', 'dodgerblue', 'cyan', 'limegreen', 'darkred']
            dens_colors = ['darkviolet', 'dodgerblue',  'cyan', 'limegreen', 'xkcd:red pink']

            dmaps_sns = LinearSegmentedColormap.from_list( 'dens_colors', dens_colors, N=len(dens_colors))
            # Position of the axes
            ax1 = axarr[0, 0].get_position().get_points().flatten()
            ax2 = axarr[0, 2].get_position().get_points().flatten()
            # Density colorbar
            cbardens_ax = fig.add_axes([ax1[0], 0.90, ax2[2]-ax1[0], 0.015])
            dens_list = [0.5, 1, 2.5, 5, 10] 
            #dmaps = matplotlib.cm.get_cmap('winter')
            #dmaps_sns = matplotlib.colors.LinearSegmentedColormap.from_list('cblues', dens_sns_map, N=len(dens_colors)+1)
            #dens_list2 = np.array([1, 2, 3, 4, 5])-0.5
            #dens_norm = mpl.colors.BoundaryNorm(dens_list, dmaps.N)
                
            #tick_locs = (np.arange(len(dens_list)+1) + 0.(5)*(len(dens_list)-1)/len(dens_list))
            tick_locs = np.arange(0,np.max(dens_list), np.max(dens_list)/(len(dens_list)+1))
            tick_locs2 = []
            for t, tick in enumerate(tick_locs):
                if t != 4:
                    if t < 4:
                        tick_locs2.append(tick+np.max(dens_list)/(len(dens_list)+1)/2)
                    else:
                        tick_locs2.append(tick-np.max(dens_list)/(len(dens_list)+1)/2)
            dens_norm_sns = mpl.colors.BoundaryNorm(tick_locs, dmaps_sns.N)
            cbdens = mpl.colorbar.ColorbarBase(cbardens_ax, cmap=dmaps_sns,
                                    norm=dens_norm_sns,
                                    #boundaries=tick_locs,
                                    spacing='uniform',
                                    orientation='horizontal',
                                    extend='neither',
                                    ticks=tick_locs2
                                    )
            cbdens.ax.set_xticklabels(['0.5', '1.0', '2.5', '5.0', '10'])
            cbdens.ax.tick_params(axis='x', direction='in',labeltop=True, labelbottom=False, bottom=False, top=True, labelsize='8', zorder=30)
            cbdens.set_label(r'$n(\rm{H}_2)$ ($10^6\,$cm$^{-3}$)', labelpad=-40)
        
        if write_temps:
            # Plotting temperature indicators (by hand!!)
            # T=200
            axarr[0, 0].text(1.25, 3.2,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            #axarr[1, 0].text(0.7, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            #axarr[2, 0].text(0.6, 4.0,r'T=200', ha='center', va='center', color='k',
            #                        rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 1].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 1].text(2.2, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            axarr[0, 2].text(2.25, 3.5,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[1, 2].text(2.25, 3.55,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            axarr[2, 2].text(2.2, 3.6,r'T=200', ha='center', va='center', color='k',
                                    rotation='horizontal', fontsize=5, zorder =13)
            
            
        #fig_name = 'TEST_HC3N_models_check'
        for formato in fig_format:
            fig.savefig(out_dir+fig_name+suffix_name+'_'+j_ratio+'.'+formato, bbox_inches='tight', transparent=False, dpi=400)





def plot_observed_ratios_NGC1068_2(fig, ax, results_path, j_ratio, vib_ratio, obs_ratios_dict,
                                 xvar, yvar, integrated, print_ylabel, print_xlabel, fontsize, color,
                                 man_obs = []):
    
    
    # Manually integrated with gaus fit on MADCUBA
    line_intens_manpath = '/Users/frico/Documents/data/NGC1068/Results_v1/hc3n_SHC_line_intens.txt'
    hc3nv_linten = pd.read_csv(line_intens_manpath, sep='\t', header=0)
    
    mv0_16_15 = hc3nv_linten[(hc3nv_linten['J']=='16-15') & (hc3nv_linten['vib']=='v=0')]
    mv0_24_23 = hc3nv_linten[(hc3nv_linten['J']=='24-23') & (hc3nv_linten['vib']=='v=0')]
    mv7_16_15 = hc3nv_linten[(hc3nv_linten['J']=='16-15') & (hc3nv_linten['vib']=='v7=1')]
    mv7_24_23 = hc3nv_linten[(hc3nv_linten['J']=='24-23') & (hc3nv_linten['vib']=='v7=1')] 
    mv6_16_15 = hc3nv_linten[(hc3nv_linten['J']=='16-15') & (hc3nv_linten['vib']=='v6=1')]
    mv6_24_23 = hc3nv_linten[(hc3nv_linten['J']=='24-23') & (hc3nv_linten['vib']=='v6=1')]
    mv0_39_38 = hc3nv_linten[(hc3nv_linten['J']=='39-38') & (hc3nv_linten['vib']=='v=0')]
    # Blended v7=1 and v6=1 lines
    mv7v6_16_15 = hc3nv_linten[(hc3nv_linten['J']=='16-15') & (hc3nv_linten['vib']=='v7=1_v6=1')]
    mv7v6_24_23 = hc3nv_linten[(hc3nv_linten['J']=='24-23') & (hc3nv_linten['vib']=='v7=1_v6=1')]
    mv7v6_39_38 = hc3nv_linten[(hc3nv_linten['J']=='39-38') & (hc3nv_linten['vib']=='v7=1_v6=1')]
    
    mv0_v7_1615_ratio = mv0_16_15['Gauss_Jy_beam_kms'].tolist()[0]/mv7_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    mv0_v7_2423_ratio = mv0_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv7_24_23['Gauss_Jy_beam_kms'].tolist()[0]
    mv0_bv7_1615_ratio = mv0_16_15['Gauss_Jy_beam_kms'].tolist()[0]/mv7v6_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    mv0_bv7_2423_ratio = mv0_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv7v6_24_23['Gauss_Jy_beam_kms'].tolist()[0]
    mv0_bv7_3938_ratio = mv0_39_38['Gauss_Jy_beam_kms'].tolist()[0]/mv7v6_39_38['Gauss_Jy_beam_kms'].tolist()[0]
    mv7v6_ratio = mv7v6_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv7v6_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    mv0_ratio = mv0_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv0_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    mv6_ratio = mv6_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv6_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    mv7_ratio = mv7_24_23['Gauss_Jy_beam_kms'].tolist()[0]/mv7_16_15['Gauss_Jy_beam_kms'].tolist()[0]
    
#    # Seetting limits depending on vars
#    if xvar == 'ratio_v7':
#        ax.set_xlim([1,3])
#    elif xvar == 'ratio_v724_23':
#        ax.set_xlim([1,6])
#    elif xvar == 'ratio_v739_38':
#        ax.set_xlim([0,15])
#    elif xvar == 'ratio_v0':
#        ax.set_xlim([0.5,2.5])
#    if yvar == 'ratio_v724_23':
#        ax.set_ylim([1.5,5])
#    elif yvar == 'ratio_v739_38':
#        ax.set_ylim([0,15])
        

    if integrated == True:
        ints = 'integ_'
    else:
        ints = 'peak_'
    jplot = j_ratio.split('_')
    # Axis labels    
    if xvar == 'ratio_v7':
        xvar = 'ratio_v7_int'
        xvar_plot = mv7v6_ratio
        xlabel = 'v7=1 ('+jplot[0]+') / v7=1 ('+jplot[1]+')'
        ax.set_xlim([1.5, 5.5])
    elif xvar == 'ratio_v6':
        xvar = 'ratio_v6_int'
        xvar_plot = mv6_ratio
        xlabel = 'v6=1 ('+jplot[0]+') / v6=1 ('+jplot[1]+')'
        ax.set_xlim([1.5, 5.5])
    elif xvar == 'ratio_v0':
        xvar = 'ratio_v0_int'
        xvar_plot = mv0_ratio
        xlabel = 'v=0 ('+jplot[0]+') / v=0 ('+jplot[1]+')'
        ax.set_xlim([0.5,2.5])
        #ax.set_xlim([0.5,1.5])
    else:
        xlabel = xvar
    #if yvar == 'ratio_v724_23':
    #    yvar = 'ratio_v724_23_int'
    #    ylabel = 'v=0 (24-23) / v7=1 (24-23)'
    #yvar = 'ratio_v724_23'
    if vib_ratio == '39-38':
        yvar_plot = mv0_bv7_3938_ratio
    elif vib_ratio == '24-23':
        yvar_plot = mv0_bv7_1615_ratio
    elif vib_ratio == '16-15':
        yvar_plot = mv0_bv7_2423_ratio
        
    ylabel = 'v=0 ('+vib_ratio+') / v7=1 ('+vib_ratio+')' 
    
            
    ax.scatter(xvar_plot, yvar_plot, marker='o', edgecolor=color, facecolor='white', s=22, zorder =22)
#    if len(man_obs) != 0:
#        print('Man ratios')
#        if 'v7' in xvar:
#            xvar_plot_man = man_obs[2]
#        elif 'v6' in xvar:
#            xvar_plot_man = man_obs[3]
#        elif 'v0' in xvar:
#            xvar_plot_man = man_obs[1]
#        yvar_plot_man = man_obs[0]
#        ax.scatter(xvar_plot_man, yvar_plot_man, marker='s', edgecolor=color, facecolor='white', s=22, zorder =22)

        #manobs = [v0_v7_1615_ratio, v0_ratio, v7_ratio, v6_ratio]
        # Plotting manual ratios
        
    # Checking if saving path exists
    out_dir = results_path+'/'+'Figures/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ax.tick_params(axis='both', which='both', direction='in', zorder=30)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='both', top =True, zorder=30)
    ax.yaxis.set_tick_params(which='both', right=True, labelright=False, zorder=30)
    # Setting labels
    if print_xlabel==True:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    else:
        ax.xaxis.set_tick_params(which='both', top =True, labelbottom=False, zorder=30)
    if print_ylabel==True:
        ax.set_ylabel(ylabel)
    return fig, ax

