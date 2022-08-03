import os
from astrothesispy.HC3N import HC3N_enerdiag

energydiag_plot = True

# HC3N CDMS data 
hc3n_info = 'HC3N/CDMS/'
fig_path = f'Figures/Thesis/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

if energydiag_plot:
    # Energy diagram figure for thesis HC3N intro
    HC3N_enerdiag.HC3N_energydiag(fig_path, hc3n_info, plot_rotational_levels = True, fig_format = '.pdf')