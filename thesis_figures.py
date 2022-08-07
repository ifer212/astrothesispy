import os
import numpy as np
import random

from astrothesispy.HC3N import HC3N_enerdiag
from astrothesispy.utiles import utiles_molecules

# Figures
energydiag_plot = True
rotational_diag_plot = True

# Setting seeds for reproduction
random.seed(1001)
np.random.seed(1001)

# HC3N CDMS data 
hc3n_info = 'HC3N/CDMS/'
fig_path = f'Figures/Thesis/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
if energydiag_plot:
    # Energy diagram figure for thesis HC3N intro
    HC3N_enerdiag.HC3N_energydiag(fig_path, hc3n_info, plot_rotational_levels = True, show_fig=True, fig_format = '.pdf')
    
if rotational_diag_plot:
    source = 'SHC_13'
    results_path = 'data/NGC253_HR/Results/'
    # Observed beam sizes
    Bmin_arcsec = 0.020
    Bmaj_arcsec = 0.022
    # Molecule info
    info_path = f'{results_path}Tables/SLIM_HC3N_info.xlsx'
    # Formatting lines info
    lines_dict = utiles_molecules.HC3N_line_dict_builder(source, results_path, info_path, Bmin_arcsec, Bmaj_arcsec)
    # Fitting to get Column density and Temperatures
    bootstrap = True # Use bootstrap method to improve fit and error estimation.
    vib_temp_df, fit_dict = utiles_molecules.rotational_diagram(lines_dict, Jselect=24, bootstrap=bootstrap)
    # Plotting
    utiles_molecules.Rotational_Diagram_plot(source, vib_temp_df, fit_dict, fig_path, plot_noboots = False, plot_boots = True)
    
   