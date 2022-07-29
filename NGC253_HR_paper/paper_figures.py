import os

from astrothesispy.scripts import NGC253HR_contfigs

D_Mpc = 3.5
cont219_plot = False
zoom_cont219_plot = False
moments_plot = True

# =============================================================================
# Global paths
# =============================================================================
source = 'SHC_13'
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