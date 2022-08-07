import os

from astrothesispy.scripts import NGC253HR_contfigs
from astrothesispy.scripts import NGC253HR_SLIMfigs
from astrothesispy.scripts import NGC253HR_nLTEfigs
from astrothesispy.scripts import NGC253HR_compfigs
from astrothesispy.scripts import NGC253HR_ringfigs

cont219_plot = False
zoom_cont219_plot = False
moments_plot = False
LTE2D_plot = False
LTEprofiles_plot = False
SB_models_plot = False
AGN_models_plot = False
LTEvelprofile_plot = False
cloudcloud_plot = False
comp_models_plot = False
ringspectra_plot = True

# =============================================================================
# Global vars & paths
# =============================================================================
figure_format = '.pdf'
D_Mpc = 3.5
Rcrit = 0.85
source = 'SHC_13'
molecule = 'HC3Nvib_J24J26'
NGC253_path = 'data/NGC253_HR/'
results_path = f'{NGC253_path}Results/'
cont_path = 'Continuums/'
moments_path = f'{NGC253_path}/Spectra/{source}/moments/'
location_path = ''
rad_transf_path = f'{NGC253_path}Radiative_Transfer/'
fig_path = f'Figures/PaperNGC253HR/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# =============================================================================
# Continuum & Moments figures
# =============================================================================
if cont219_plot:
    # Figure 1, saved in data/NHC253_HR/Results/Figures/NGC253/
    fig_name = 'Figure_1_'
    NGC253HR_contfigs.plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path,
                                   D_Mpc = D_Mpc, fig_name = fig_name, fig_format = figure_format)
    
if zoom_cont219_plot:
    # Figure 2, saved in data/NHC253_HR/Results/Figures/NGC253/
    fig_name = 'Figure_2_'
    ind_fig = False # Make individual plots
    NGC253HR_contfigs.plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path,
                                        ind_fig = ind_fig,
                                        D_Mpc = D_Mpc, fig_name = fig_name, fig_format = figure_format)
    
if moments_plot:
    # Figure 3, saved in data/NHC253_HR/Results/Figures/SHC_13/
    fig_name = 'Figure_3_'
    NGC253HR_contfigs.plot_moments(NGC253_path, cont_path, location_path, moments_path, fig_path,
                                   D_Mpc = D_Mpc, source = source, fig_name = fig_name, fig_format = figure_format)
    
# =============================================================================
# SLIM LTE figures
# =============================================================================
if LTE2D_plot:
    # Figure 4, saved in data/NHC253_HR/Results/Figures/SHC_13/
    fig_name = 'Figure_4_'
    NGC253HR_SLIMfigs.plot_SLIM2D(NGC253_path, results_path,  moments_path, cont_path, location_path, fig_path,
                                  molecule = molecule, source = source, D_Mpc = D_Mpc, fig_name = fig_name, fig_format = figure_format)

if LTEprofiles_plot:
    # Figure 5, saved in data/NHC253_HR/Results/Figures/SHC_13/
    fig_name = 'Figure_5_'
    NGC253HR_SLIMfigs.plot_SLIMprofiles(NGC253_path, results_path, fig_path, fig_name = fig_name, fig_format = figure_format)
    
if LTEvelprofile_plot:
    # Figure 11
    fig_name = 'Figure_11_'
    NGC253HR_SLIMfigs.plot_velprofiles(NGC253_path, source, fig_path, rad_transf_path, results_path, molecule = 'HC3Nvib_J24J26',
                                       modelname = 'model2', Rcrit = 0.85, D_Mpc = 3.5, style = 'onepanel',
                                       fig_name = fig_name, fig_format = figure_format)

if cloudcloud_plot:
    # Figure 12
    fig_name = 'Figure_12_'
    NGC253HR_SLIMfigs.plot_pvdiagram(NGC253_path, results_path, source, fig_path, moments_path, molecule = 'HC3Nvib_J24J26',
                                     D_Mpc = 3.5, style = 'onecol', fig_name = fig_name, fig_format = figure_format)

# =============================================================================
# Radiative transfer modelling figures
# =============================================================================
if SB_models_plot:
    # Figures 6, 7 and 8, saved in data/NHC253_HR/Results/Figures/SHC_13/
    fig_name = ['Figure_6_', 'Figure_7_', 'Figure_8_']
    NGC253HR_nLTEfigs.nLTE_model_plot(NGC253_path, source, results_path, fig_path, rad_transf_path,
                                      D_Mpc = D_Mpc, Rcrit = Rcrit, plot_type = 'SBmods',
                                      paper_figs = True, presen_figs = False, fig_name = fig_name,
                                      fig_format = figure_format)

if AGN_models_plot:
    # Figure 9 and 10, saved in data/NHC253_HR/Results/Figures/SHC_13/
    fig_name = ['Figure_9_', '', 'Figure_10_']
    NGC253HR_nLTEfigs.nLTE_model_plot(NGC253_path, source, results_path, fig_path, rad_transf_path,
                                      D_Mpc = D_Mpc, Rcrit = Rcrit, plot_type = 'AGNmods',
                                      paper_figs = True, presen_figs = False, fig_name = fig_name,
                                      fig_format = figure_format)

# =============================================================================
# Comparisson figures btw HCs, SHCs and AGNs
# =============================================================================
if comp_models_plot:
    # Figure 13
    fig_name = 'Figure_13_'
    NGC253HR_compfigs.plot_LIR_comp(fig_path, results_path, source, D_Mpc=D_Mpc, fig_name = fig_name, fig_format = figure_format)

# =============================================================================
# Ring spectra figures
# =============================================================================
if ringspectra_plot:
    # Appendix Ring Figures
    NGC253HR_ringfigs.ring_create_and_plot(source, NGC253_path, f'{fig_path}{source}/rings/', size = 1.5, step = 0.1)