import os

from astrothesispy.scripts import NGC253HR_contfigs

D_Mpc = 3.5
cont219_plot = False
zoom_cont219_plot = True

# =============================================================================
# Global paths
# =============================================================================
NGC253_path = '/mnt/c/Users/Usuario/Documents/CAB/NGC253_HR/'
results_path = NGC253_path+'Results_v2/'
cont_path = 'Continuums/'
location_path = ''
fig_path = NGC253_path+'new_figs/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# =============================================================================
# Continuum figures
# =============================================================================
if cont219_plot:
    # Figure 1
    NGC253HR_contfigs.plot_cont219(NGC253_path, cont_path, location_path, results_path, fig_path)
    
if zoom_cont219_plot:
    # Figure 2
    NGC253HR_contfigs.plot_cont219_zoom(NGC253_path, cont_path, location_path, results_path, fig_path)