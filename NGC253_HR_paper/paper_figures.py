import os

from astrothesispy.scripts import NGC253HR_contfigs
from astrothesispy.scripts import NGC253HR_SLIMfigs
from astrothesispy.scripts import NGC253HR_nLTEfigs
from astrothesispy.scripts import NGC253HR_compfigs

cont219_plot = True
zoom_cont219_plot = False
moments_plot = False
ringspectra_plot = False
LTE2D_plot = False
LTEprofiles_plot = False
SB_models_plot = False
AGN_models_plot = False
LTEvelprofile_plot = False
cloudcloud_plot = False
comp_models_plot = False

# =============================================================================
# Global vars & paths
# =============================================================================
D_Mpc = 3.5
Rcrit = 0.85
source = 'SHC_13'
molecule = 'HC3Nvib_J24J26'
NGC253_path = '/mnt/c/Users/Usuario/Documents/CAB/NGC253_HR/'
NGC253_path = 'data/NGC253_HR/'
results_path = f'{NGC253_path}Results/'
cont_path = 'Continuums/'
moments_path = f'{NGC253_path}/SHC/{source}/moments/'
location_path = ''
fig_path = f'{results_path}Figures/{source}'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
rad_transf_path = '/mnt/c/Users/Usuario/Documents/CAB/radtransf/program/'
rad_transf_path = f'{results_path}/radtransf/program/'
figure_format = '.pdf'
# =============================================================================
# Continuum & Moments figures
# =============================================================================
if cont219_plot:
    # Figure 1
    fig_name = 'Figure_1_'
    NGC253HR_contfigs.plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path, D_Mpc = D_Mpc, fig_name = fig_name, fig_format = figure_format)
    
if zoom_cont219_plot:
    # Figure 2
    fig_name = 'Figure_2_'
    NGC253HR_contfigs.plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path, D_Mpc = D_Mpc, fig_name = fig_name)
    
if moments_plot:
    # Figure 3
    fig_name = 'Figure_3_'
    NGC253HR_contfigs.plot_moments(NGC253_path, cont_path, location_path, moments_path, fig_path, D_Mpc = D_Mpc, source = source, fig_name = fig_name)
    
if ringspectra_plot:
    # Figure 4
    fig_name = 'Figure_4_'
    # spectra from averaged ring, this figure is done inside Madcuba spectra
    skip_part = True
    
# =============================================================================
# SLIM LTE figures
# =============================================================================
if LTE2D_plot:
    # Figure 5
    fig_name = 'Figure_5_'
    NGC253HR_SLIMfigs.plot_SLIM2D(NGC253_path,  cont_path, location_path, fig_path, molecule = molecule, source = source, D_Mpc = D_Mpc, fig_name = fig_name)

if LTEprofiles_plot:
    # Figure 6
    fig_name = 'Figure_6_'
    NGC253HR_SLIMfigs.plot_SLIMprofiles(NGC253_path, fig_path, fig_name = fig_name)
    
if LTEvelprofile_plot:
    # Figure 12
    fig_name = 'Figure_12_'
    NGC253HR_SLIMfigs.plot_velprofiles(NGC253_path, source, fig_path, rad_transf_path, results_path, molecule = 'HC3Nvib_J24J26', modelname = 'model2', Rcrit = 0.85, D_Mpc = 3.5, style = 'onepanel', fig_name = fig_name)

if cloudcloud_plot:
    # Figure 13
    fig_name = 'Figure_13_'
    NGC253HR_SLIMfigs.plot_pvdiagram(NGC253_path, source, fig_path, moments_path, molecule = 'HC3Nvib_J24J26', D_Mpc = 3.5, style = 'onecol', fig_name = fig_name)

# =============================================================================
# Radiative transfer modelling figures
# =============================================================================
if SB_models_plot:
    # Figures 7, 8 and 9
    fig_name = ['Figure_7_', 'Figure_8_', 'Figure_9_']
    NGC253HR_nLTEfigs.nLTE_model_plot(NGC253_path, source, results_path, fig_path, rad_transf_path, D_Mpc = D_Mpc, Rcrit = Rcrit, plot_type = 'SBmods', paper_figs = True, presen_figs = False, fig_name = fig_name)

if AGN_models_plot:
    # Figure 10 and 11
    fig_name = ['Figure_10_', '', 'Figure_11_']
    NGC253HR_nLTEfigs.nLTE_model_plot(NGC253_path, source, results_path, fig_path, rad_transf_path, D_Mpc = D_Mpc, Rcrit = Rcrit, plot_type = 'AGNmods', paper_figs = True, presen_figs = False, fig_name = fig_name)

# =============================================================================
# Comparisson figures btw HCs, SHCs and AGNs
# =============================================================================
if comp_models_plot:
    # Figure 14
    fig_name = 'Figure_14_'
    NGC253HR_compfigs.plot_LIR_comp_ALL_big(fig_path, results_path, source, D_Mpc=D_Mpc, fig_name = fig_name)
