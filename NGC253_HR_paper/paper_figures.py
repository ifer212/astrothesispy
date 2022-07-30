import os

from astrothesispy.scripts import NGC253HR_contfigs
from astrothesispy.scripts import NGC253HR_SLIMfigs


cont219_plot = False
zoom_cont219_plot = False
moments_plot = False
ringspectra_plot = False
LTE2D_plot = True
LTEprofiles_plot = True

# =============================================================================
# Global vars & paths
# =============================================================================
D_Mpc = 3.5
source = 'SHC_13'
molecule = 'HC3Nvib_J24J26'
NGC253_path = '/mnt/c/Users/Usuario/Documents/CAB/NGC253_HR/'
results_path = f'{NGC253_path}Results_v2/'
cont_path = 'Continuums/'
moments_path = f'{NGC253_path}/SHC/{source}/moments/'
location_path = ''
fig_path = f'{NGC253_path}new_figs/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
# =============================================================================
# Continuum & Moments figures
# =============================================================================
if cont219_plot:
    # Figure 1
    NGC253HR_contfigs.plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path, D_Mpc = D_Mpc)
    
if zoom_cont219_plot:
    # Figure 2
    NGC253HR_contfigs.plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path, D_Mpc = D_Mpc)
    
if moments_plot:
    # Figure 3
    NGC253HR_contfigs.plot_moments(NGC253_path, cont_path, location_path, moments_path, fig_path, D_Mpc = D_Mpc, source = source)
    
if ringspectra_plot:
    # Figure 4
    # spectra from averaged ring, this figure is done inside Madcuba spectra
    cont = True
    
# =============================================================================
# SLIM LTE figures
# =============================================================================
if LTE2D_plot:
    # Figure 5
    NGC253HR_SLIMfigs.plot_SLIM2D(NGC253_path,  cont_path, location_path, fig_path, molecule = molecule, source = source, D_Mpc = D_Mpc)

if LTEprofiles_plot:
    # Figure 6
    NGC253HR_SLIMfigs.plot_SLIMprofiles(NGC253_path, fig_path)
    
# =============================================================================
# Radiative transfer modelling figures
# =============================================================================



# =============================================================================
# Comparisson figures btw HCs, SHCs and AGNs
# =============================================================================

SHC_compfig_helper.plot_LIR_comp_ALL(modsum_df, hc_df, rolffs_df, bgn_df, results_path
                                    , pfalzner_df, lada_df, portout_df,portin_df, Lmod_err=0.5, only_HC = True)

SHC_compfig_helper.plot_LIR_comp_ALL_big(modsum_df, hc_df, rolffs_df, bgn_df, results_path
                                    , pfalzner_df, lada_df, portout_df,portin_df, Lmod_err=0.5, only_HC = True)
