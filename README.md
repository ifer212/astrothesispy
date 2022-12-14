# Astrothesispy
---

> Astrothesispy is just a recompilation of equations, methods and plots created for my PhD in Atrophysics and for some scientific publications.

To install just use pip:
&emsp;  ```
            pip install astrothesispy
        ```

- The ```utiles``` module includes most of the methods defined during the thesis:
    - ```utiles.utiles```: Contains simple method to work with arrays and some simple statistics function (e.g. bootrstrapping).
    - ```utiles.utiles_plot```: Contains methods to use recursively in plots (e.g. ```add_cbar()``` to add a color bar to a plot, ```map_figure_starter()``` to create the axis with World Coordinate System (WCS) and proper RA and Dec labelling).
    - ```utiles.utiles_physical```: Contains equations and funcitons for physics.
    - ```utiles.utiles_nLTEmodel```: Contains methods for handling the non Local Thermodynamical Equilibrium (non LTE) radiative transfer modeling files. The modelling is carried in fortran and is not incluided. The module modifies de input files, call the fortran compiler, and collects the outputs from the different files generated.
    - ```utiles.utiles_molecules```: Contains some methods for calculations with moelcules (e.g. ```transition_citical_density()```, ```Columndensity_thin()```, ```Columndensity_thin()```) and also obtain the excitation temperature and column density through a rotational digram (```rotational_diagram()```) and plot it.
    - ```utiles.utiles_cubes```: Contains the ```Cube_Handler()``` class to handle the .fits format data cubes and some functions to calculate the cube moments, sigma masking, etc...
    - ```utiles.utiles_alma```: Contains some functions to estimate observational parameters from the ALMA interferometer observations.
    - ```utiles.u_conversion```: Contains methods to transform between units for complex units systems.
    - ```utiles.Madcuba_plotter```: Contains methods to plot the spectra extracted from [Madcuba](https://cab.inta-csic.es/madcuba/) (a software developed in the Center of Astrobiology of Madrid to analyze astronomical datacubes and multiple spectra from the main astronomical facilities), labelling the molecular lines and auto-positioning the labels to avoid text overlapping.

 

- The ```data/``` directory includes some of the data used for my thesis (mostly for the _Rico-Villas et al. 2022_ publication) and required to create the plots inside ```Figures/```

- The ```HC3N``` module includes some data from the CDMS catalogue for HC3N and the ```HC3N_enerdiag``` to plot the energy diagram of HC3N vibrational states.

- The ```radiative_transfer``` module includes some functions to make calculations from the non LTE results.
- The ```scripts``` module contains examples on how to re-create the figures.


Figures created for  _Rico-Villas et al. 2022_ using this repository (see ```paper_figures.py```):
- Figure 1: Continuum map at 219 GHz of the NGC 253 nuclear region with a resolution of $0.020^{\prime\prime} \times 0.019^{\prime\prime}$ above $3\sigma$.

<img src="Figures/PaperNGC253HR/NGC253/Figure_1_219GHz.png" alt="Fig1" style="background-color: white;" />

- Figure 2: Zoom of the continuum map at 219 GHz above $5\sigma$ for the different regions containing the proto-SSCs studied in _Rico-Villas et al. 2020_. Overlaid in red is the continuum emission at 345 GHz.
<img src="Figures/PaperNGC253HR/NGC253/Figure_2_ALL_subcont_219GHz_and_350GHz.png" alt="Fig2" style="background-color: white;" />
- Appendix figure:  Proto-SSC 13 ring averaged spectra between 0.1 and 0.2 pc. Overlaid in red is the total fitted model, in green or blue the contribution from HC3N. 
<img src="Figures/PaperNGC253HR/SHC_13/rings/SHC_13_d0p15.png" alt="Fig3" style="background-color: white;" />
- Figure 4: LTE fitted values with SLIM. Top left panel shows the HC3N column density, top right panel the vibrational temperature, bottom left the VLSR and bottom right the FWHM.
<img src="Figures/PaperNGC253HR/SHC_13/Figure_4_SHC_13_SLIM_cubes_HC3Nvib_J24J26.png" alt="Fig4" style="background-color: white;" />
- Figure 5: HC3N* vibrational temperature (Tvib) and column density (log N (HC3N)) profiles derived from the LTE mode.
<img src="Figures/PaperNGC253HR/SHC_13/Figure_5_SHC_13_SLIM_Tex_and_logN_profiles.png" alt="Fig5" style="background-color: white;" />

- Figure 13: Comparison between the derived properties of proto-SSC 13a (red circle) to Milky Way HCs (blue circles) and BGNs from (U)LIRGs (green circles).
<img src="Figures/PaperNGC253HR/SHC_13/Figure_13_SHC_13_LIR_comp.png" alt="Fig13"  style="background-color: white;" />



- Example of the energy diagram figure:


```python
import os
from astrothesispy.HC3N import HC3N_enerdiag
# HC3N CDMS data 
hc3n_info = 'HC3N/CDMS/'
fig_path = f'Figures/Thesis/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
import numpy as np
np.seterr(divide = 'ignore') 
# Energy diagram figure for thesis HC3N intro
HC3N_enerdiag.HC3N_energydiag(fig_path, hc3n_info, plot_rotational_levels = True, fig_format = '.png')
```

    /home/fmirg/anaconda3/envs/astro/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.1
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"


<img src="Figures/Thesis/HC3N_Ediag_K_wrot_lvls.png" alt="EnergyDiag"  style="background-color: white;" />

- Example of a rotational diagram using bootstrap:


```python
from astrothesispy.utiles import utiles_molecules
import os
import numpy as np
import random
# Setting seeds for reproduction
random.seed(1001)
np.random.seed(1001)

fig_path = f'Figures/Thesis/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    
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
utiles_molecules.Rotational_Diagram_plot(source, vib_temp_df, fit_dict, fig_path, plot_noboots = False, plot_boots = True, fig_format = '.png')
```

    ????????????????????????????????????
    No bootstrap:
    N$_{\rm{Tot}}$=$2.3\times10^{16}\pm1.8\times10^{12}$cm$^{-1}$ 	 T$_{\rm{vib}}$=$462.0\pm38.9$ K
    Bootstrap:
    N$_{\rm{Tot}}$=$2.3\times10^{16}\pm6.2\times10^{12}$cm$^{-1}$ 	 T$_{\rm{vib}}$=$455.0\pm127.2$ K


<img src="Figures/Thesis/SHC_13_Rotational_Diagram.png" alt="RotDiag" style="background-color: white;" />

### Scientific publications using this repository

 * _Rico-Villas et al. 2022_    &emsp; &emsp; &emsp;   <img src="Figures/readme_logos/arxiv-logo.svg" alt="arxiv" style="width: 50px; height: 20px;"/> [2208.01941](https://ui.adsabs.harvard.edu/link_gateway/2022arXiv220801941R/arxiv:2208.01941)  &nbsp;  <img src="Figures/readme_logos/DOI_logo.svg" alt="DOI" style="width: 20px; height: 20px;"/> _submitted_
 * _Rico-Villas et al. 2021_    &emsp; &emsp; &emsp;    <img src="Figures/readme_logos/arxiv-logo.svg" alt="arxiv" style="width: 50px; height: 20px;"/> [2008.03693](https://ui.adsabs.harvard.edu/link_gateway/2021MNRAS.502.3021R/arxiv:2008.03693)  &nbsp; <img src="Figures/readme_logos/DOI_logo.svg" alt="DOI" style="width: 20px; height: 20px;"/> [10.1093/mnras/stab197](https://ui.adsabs.harvard.edu/link_gateway/2021MNRAS.502.3021R/doi:10.1093/mnras/stab197)
 * _Rico-Villas et al. 2020_   &emsp; &emsp; &emsp;     <img src="Figures/readme_logos/arxiv-logo.svg" alt="arxiv" style="width: 50px; height: 20px;"/>  [1909.11385](https://ui.adsabs.harvard.edu/link_gateway/2020MNRAS.491.4573R/arxiv:1909.11385) &nbsp;  <img src="Figures/readme_logos/DOI_logo.svg" alt="DOI" style="width: 20px; height: 20px;"/> [10.1093/mnras/stz3347](https://ui.adsabs.harvard.edu/link_gateway/2020MNRAS.491.4573R/doi:10.1093/mnras/stz3347)



