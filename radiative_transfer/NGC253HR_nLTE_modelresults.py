
import os




from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_cubes
from astrothesispy.utiles import utiles_nLTEmodel
from astrothesispy.utiles import SHC_compfig_helper
from astrothesispy.utiles import utiles_plot as plot_utiles
from astrothesispy.utiles import u_conversion
from astrothesispy.utiles import utiles_physical


import pandas as pd
import numpy as np
import astropy.units as u

import astropy.constants.si as _si
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from scipy import special
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
mpl.rc('xtick', color='k', direction='in', labelsize=6)
mpl.rc('ytick', color='k', direction='in', labelsize=6)
cluz = _si.c.to(u.km/u.s).value

load_observed_results = True
check_model = False
spec_index = False
print_fluxes_latex = False
models_calculations = True
fit_logn = False
plot_cont = False

# Parameters
D_Mpc = 3.5 
Rcrit = 0.85 # pc
source = 'SHC_13'
SHC = source
molecule = 'HC3Nvib_J24J26'
# Data cubes paths
NGC253_path = '/mnt/c/Users/Usuario/Documents/CAB/NGC253_HR/'
cont_cube_path = f'{NGC253_path}Continuums/{source}/'
line_cube_path = f'{NGC253_path}SHC/{source}/contsub/'
# Radiative transfer models paths
rad_transf_path = '/mnt/c/Users/Usuario/Documents/Ed/transf/program/'
ed_model_path = f'{rad_transf_path}models/'
my_model_path = f'{rad_transf_path}/models/mymods/'
fort_paths = f'{rad_transf_path}/{source}/'
dustmod_radpc = 17 # All dustmodels are ran with r=17pc
ed_fig_save_path = f'{NGC253_path}new_figs/'
my_model_figs = f'{ed_fig_save_path}/figures_radtransf/'
results_path = f'{NGC253_path}Results_v2/'
# Observed fluxes 
new_hb_path = f'{NGC253_path}/{source}/EdFlux/{source}_flujos_hc3n_python.xlsx'
new_hb_df = pd.read_excel(new_hb_path, header=0)
# Observed conts
cont_path = fort_paths+'SHC13_flujos_continuo_def.dat'
cont_df = pd.read_csv(cont_path, comment='!', header=None, delim_whitespace=True)
cont_df.columns = ['dist', 'F235GHz_mjy_beam', 'F235GHz_mjy_beam_err', 'F235GHz_mjy_beam345', 'F235GHz_mjy_beam345_err',
                 'F345GHz_mjy_beam', 'F345GHz_mjy_beam_err']
# Results paths
if not os.path.exists(results_path):
    os.makedirs(results_path)
fig_path = my_model_figs+'Extra_Figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
figrt_path = my_model_figs+'Figures_radtransf/'
if not os.path.exists(figrt_path):
    os.makedirs(figrt_path)
figinp_path = my_model_figs+'model_inputs/'
if not os.path.exists(figinp_path):
    os.makedirs(figinp_path)
figmod_path = my_model_figs+'lines_python/'
if not os.path.exists(figmod_path):
    os.makedirs(figmod_path)
finalfigmod_path = my_model_figs+'Models_final/'
if not os.path.exists(finalfigmod_path):
    os.makedirs(finalfigmod_path)
    
# SLIM observed Results
if load_observed_results:
    def load_observed_LTE(results_path, fig_path, plot_LTE_lum = False):
        """ 
            Loads the observed data from LTE
        """
        obs_df = pd.read_csv(f'{results_path}SHC_13_SLIM_Tex_and_logN_profiles.csv', sep=';')
        obs_df['Dist_mean_cm'] = obs_df['Dist_mean_pc']*(1*u.pc).to(u.cm).value
        tstring = 'Tex_SM_ave_ring'
        s_si = _si.sigma_sb.to(u.Lsun/(u.pc**2*u.K**4)) # Boltzmann constant in Lsun/(pc**2 K**4)
        for i,row in obs_df.iterrows():
            obs_df.loc[i, 'Lum'] = 4*np.pi*((row['dist_ring_pc']+0.05)**2-(row['dist_ring_pc']-0.05)**2)*s_si.value*row[tstring]**4
        # LTE luminosity profile
        if plot_LTE_lum:
            fig = plt.figure(figsize=[10,8])
            ax = fig.add_subplot(111)
            ax.plot(obs_df['dist_ring_pc'], obs_df['Lum']/1e8, linestyle='-', marker='o', color='k')
            ax.set_xlabel(r'r (pc)')
            ax.set_ylabel(r'L ($10^8$ L$_\odot$)')
            fig.savefig(fig_path+'LTE_Lum_profile.pdf', bbox_inches='tight', transparent=True, dpi=400)
            plt.close()
        return obs_df
    
# Models
if check_model:
    model_name = 'LTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5col1.0E+16vt5_c2'
    dust_model = 'dustsblum1.0E+11cd1.0E+25exp1.5nsh1003rad17'
    cubo219_path = '/Users/frico/Documents/data/NGC253_HR/Continuums/MAD_CUB_219GHz_spw25_continuum.I.image.pbcor.fits'
    cubo219 = utiles.Cube_Handler('219', cubo219_path)
    cubo_219_pixlen_pc = u_conversion.lin_size(D_Mpc, cubo219.pylen*3600).to(u.pc).value
    pxsize_pc2 = cubo_219_pixlen_pc**2
    arcsec2_beam=np.pi*cubo219.bmaj*3600*cubo219.bmin*3600/4.0/np.log(2.0)
    pc2_beam = arcsec2_beam*(D_Mpc*1e6*np.pi/(180.0*3600.0))**2
    flux = 2.2364E+12
    f_pix = flux * ((cubo_219_pixlen_pc/(D_Mpc*1e6))**2)
    f_beam = flux * ((cubo_219_pixlen_pc/(D_Mpc*1e6))**2)/(cubo_219_pixlen_pc**2)*pc2_beam
# Calculate spectral index
if spec_index:
    def calculate_spindex(NGC253_path, results_path):
        """
            Calculates the spectral index from 111GHz, 219GHz and 345GHz data
        """
        f36  = 36.0
        f111 = 111.55
        f219 = 219.1
        f350 = 345.15
        f111_rms = 4.303E-5#Jy/beam
        f219_rms = 1.307E-5#Jy/beam
        f350_rms = 2.671e-5#Jy/beam
        Te = 10000 # Temperatura electronica
        beta = 1.5
        cubo_111_path = f'{NGC253_path}/Continuums/MAD_CUB_MOD_111GHz_spw31_continuum_mfs.I.manual.image.pbcor.fits'
        cubo_111 = utiles_cubes.Cube_Handler('111', cubo_111_path)
        rms_111 = 4.303E-5
        cubo_111_pixlen_pc = u_conversion.lin_size(D_Mpc, cubo_111.pylen*3600).to(u.pc).value
        Leroy36df_new = pd.read_excel(results_path+'Leroy_36GHz_python_219_nospind_v2.xlsx', header=0, na_values='-')
        Leroy36df_new['F36GHz_app_Jy'] = Leroy36df_new['F36GHz_app_mJy']/1000
        Leroy36df_new['F36GHz_app_err_Jy'] = Leroy36df_new['F36GHz_app_err_mJy']/1000
        # Si salen tamaños muy grandes para el SHC13, algo debe estar mal
        Leroy36df_new['111GHz_deconv_x_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['111GHz_deconv_x_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['111GHz_deconv_y_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['111GHz_deconv_y_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM2_deconv_x_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM2_deconv_x_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM2_deconv_y_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM2_deconv_y_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['36GHz_111GHz_sp_ind_int_vlast']  = np.log10(Leroy36df_new['111GHz_SM_obsmad_int_Jy']/Leroy36df_new['F36GHz_app_Jy'])/np.log10(f111/f36)
        spec_ind_dict = {'f1': f219,
                        'f2': f350,
                        'F1': '219GHz_SM',
                        'F2': '350GHz'
                        }
        spec_indint_dict = {'f1': f219,
                        'f2': f350,
                        'F1': '219GHz_SM',
                        'F2': '345GHz'
                        }
        Leroy36df_new['219GHz_345GHz_sp_ind_peak_vlast'] = np.log10(Leroy36df_new['F'+spec_ind_dict['F2']+'_Jy_beam']/Leroy36df_new['F'+spec_ind_dict['F1']+'_Jy_beam'])/np.log10(spec_ind_dict['f2']/spec_ind_dict['f1'])
        Leroy36df_new['219GHz_345GHz_sp_ind_int_vlast']  = np.log10(Leroy36df_new[spec_indint_dict['F2']+'_obsmad_int_Jy']/Leroy36df_new[spec_indint_dict['F1']+'_obsmad_int_Jy'])/np.log10(spec_indint_dict['f2']/spec_indint_dict['f1'])
        des_sp_ind = 3.5
        # Expected non-dust emission at 219GHz obtained by substracting the extrapol dust emission from 345GHz 
        Leroy36df_new['S219_SM_nondust_peak_Jy_beam'] = Leroy36df_new['F219GHz_SM_Jy_beam']-Leroy36df_new['F350GHz_Jy_beam']*(f219/f350)**des_sp_ind
        Leroy36df_new['S219_SM_nondust_int_Jy'] = Leroy36df_new['219GHz_SM_obsmad_int_Jy']-Leroy36df_new['345GHz_obsmad_int_Jy']*(f219/f350)**des_sp_ind
        Leroy36df_new['S219_SM_nondust_peak_fraction'] = 100*Leroy36df_new['S219_SM_nondust_peak_Jy_beam']/Leroy36df_new['F219GHz_SM_Jy_beam']
        Leroy36df_new['S219_SM_nondust_int_fraction'] =  100*Leroy36df_new['S219_SM_nondust_int_Jy']/Leroy36df_new['219GHz_SM_obsmad_int_Jy']
        # Optically thick
        # alpha = 2+beta*tau*(e^tau-1)**-1
        # Expected non-dust emission at 219GHz obtained by substracting the extrapol dust emission from 345GHz 
        Leroy36df_new['S219_SM_nondust_peak_Jy_beam'] = Leroy36df_new['F219GHz_SM_Jy_beam']-Leroy36df_new['F350GHz_Jy_beam']*(f219/f350)**des_sp_ind
        Leroy36df_new['S219_SM_nondust_int_Jy'] = Leroy36df_new['219GHz_SM_obsmad_int_Jy']-Leroy36df_new['345GHz_obsmad_int_Jy']*(f219/f350)**des_sp_ind
        Leroy36df_new['S219_SM_nondust_peak_fraction'] = 100*Leroy36df_new['S219_SM_nondust_peak_Jy_beam']/Leroy36df_new['F219GHz_SM_Jy_beam']
        Leroy36df_new['S219_SM_nondust_int_fraction'] =  100*Leroy36df_new['S219_SM_nondust_int_Jy']/Leroy36df_new['219GHz_SM_obsmad_int_Jy']
        # Ionising photon rate (from Murray2011 eq.10) and Mass ZAMS from Leroy2018
        Leroy36df_new['S219_SM_nondust_int_W/m2'] = u_conversion.jy_to_wm2(Leroy36df_new['S219_SM_nondust_int_Jy'], f219)
        Leroy36df_new['S219_SM_nondust_int_Lsun'] = u_conversion.flux_to_lum(Leroy36df_new['S219_SM_nondust_int_W/m2'], D_Mpc, z=False)
        Leroy36df_new['S219_SM_nondust_int_L_erg/s'] =  u_conversion.lsun_to_ergs(Leroy36df_new['S219_SM_nondust_int_Lsun'])
        Leroy36df_new['S219_SM_nondust_int_Q0'] = utiles_physical.ion_prod_rate(f219, Leroy36df_new['S219_SM_nondust_int_L_erg/s'], Te)
        Leroy36df_new['S219_SM_nondust_int_MZAMS_Msun'] = Leroy36df_new['S219_SM_nondust_int_Q0']/4e46
        Leroy36df_new['S219_SM_nondust_int_Q0_J'] = utiles_physical.ion_prod_rate_j(f219, Leroy36df_new['S219_SM_nondust_int_Jy'], Te, D_Mpc*1000)
        Leroy36df_new['S219_SM_nondust_int_MZAMS_Msun_J'] = Leroy36df_new['S219_SM_nondust_int_Q0_J']/4e46
        # Remaining dust emission at 219GHz
        Leroy36df_new['S219_SM_dust_int_Jy'] = (100-Leroy36df_new['S219_SM_nondust_int_fraction'])*Leroy36df_new['219GHz_SM_obsmad_int_Jy']/100
        Leroy36df_new['S219_SM_dust_peak_Jy'] = (100-Leroy36df_new['S219_SM_nondust_peak_fraction'])*Leroy36df_new['F219GHz_SM_Jy_beam']/100
        Leroy36df_new['F219_SM_sp3.5_Jy_beam'] = 10**(np.log10(Leroy36df_new['F'+spec_ind_dict['F2']+'_Jy_beam'])-des_sp_ind*np.log10(spec_ind_dict['f2']/spec_ind_dict['f1']))
        Leroy36df_new['F219_SM_sp3.5_mJy_beam'] = Leroy36df_new['F219_SM_sp3.5_Jy_beam']*1000
        Leroy36df_new['F219_SM_sp3.5_contr'] = 100*Leroy36df_new['F219_SM_sp3.5_Jy_beam']/Leroy36df_new['F'+spec_ind_dict['F1']+'_Jy_beam']
        subdf = Leroy36df_new.dropna(subset = ['Source_altern_sub_final'])
        for i, row in subdf.iterrows():
            print(f'{row["Source_altern_sub_final"]}   \t\t {row["S219_SM_nondust_peak_Jy_beam"]*1000:1.2f} \t {row["S219_SM_nondust_peak_fraction"]:1.1f}  \t {row["S219_SM_nondust_int_Jy"]*1000:1.2f} \t {row["S219_SM_nondust_int_fraction"]:1.1f}')
        for i, row in subdf.iterrows():
            print(f'{row["Source_altern_sub_final"]}   \t\t {row["111GHz_Int_fit_Jy"]*1000:1.2f} \t {row["111GHz_Obs_Int_fit_Jy"]*1000:1.2f}')
        Leroy36df_new['S111_int_Q0_J'] = utiles_physical.ion_prod_rate_j(f111, Leroy36df_new['111GHz_Int_fit_Jy'], Te, D_Mpc*1000)
        Leroy36df_new['S111_int_MZAMS_Msun_J'] = Leroy36df_new['S111_int_Q0_J']/4e46
        for i, row in subdf.iterrows():
            print(f'{row["Source_altern_sub_final"]}   \t\t {row["S219_SM_nondust_peak_Jy_beam"]/row["S219_SM_nondust_int_Jy"]:1.2f}')
        # At resolution of 345GHz
        Leroy36df_new['219GHz_SM_deconv_x_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM_deconv_x_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM_deconv_y_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM_deconv_y_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit'] = np.sqrt(Leroy36df_new['219GHz_SM_deconv_x_FWHM_pc_fit']*Leroy36df_new['219GHz_SM_deconv_y_FWHM_pc_fit'])
        Leroy36df_new['350GHz_deconv_x_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['350GHz_deconv_x_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['350GHz_deconv_y_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['350GHz_deconv_y_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['350GHz_deconvFWHM_pc_fit'] = np.sqrt(Leroy36df_new['350GHz_deconv_x_FWHM_pc_fit']*Leroy36df_new['350GHz_deconv_y_FWHM_pc_fit'])
        # At resolution of 111GHz
        Leroy36df_new['219GHz_SM2_deconv_x_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM2_deconv_x_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM2_deconv_y_FWHM_pc_fit'] = u_conversion.lin_size(D_Mpc, Leroy36df_new['219GHz_SM2_deconv_y_FWHM_arcsec_fit'])*(1*u.m).to(u.pc).value
        Leroy36df_new['219GHz_SM2_deconvFWHM_pc_fit'] = np.sqrt(Leroy36df_new['219GHz_SM2_deconv_x_FWHM_pc_fit']*Leroy36df_new['219GHz_SM2_deconv_y_FWHM_pc_fit'])
        # Extrapolated dust emission from 36-111GHz spectral index
        #111GHz_obsmad_int_Jy # orig resolution
        #111GHz_SM_obsmad_int_Jy # resolution of 36GHz
        Leroy36df_new['219_111_sqr_size_ratio'] = (Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']/Leroy36df_new['FWHM_pc'])**2
        Leroy36df_new['Sprime219_SM_nondust_int_Jy'] = Leroy36df_new['111GHz_SM_obsmad_int_Jy']*Leroy36df_new['219_111_sqr_size_ratio']*((f219/f111)**Leroy36df_new['36GHz_111GHz_sp_ind_int_vlast'])
        Leroy36df_new['Sprime219_SM_nondust_int_fraction'] = 100*Leroy36df_new['Sprime219_SM_nondust_int_Jy']/Leroy36df_new['219GHz_SM_obsmad_int_Jy']
        # Assuming 111GHz same size as 219GHz
        Leroy36df_new['Sprime219_SM_nondust_samesize_int_Jy'] = Leroy36df_new['111GHz_SM_obsmad_int_Jy']*((f219/f111)**Leroy36df_new['36GHz_111GHz_sp_ind_int_vlast'])
        Leroy36df_new['Sprime219_SM_nondust_samesize_int_fraction'] = 100*Leroy36df_new['Sprime219_SM_nondust_samesize_int_Jy']/Leroy36df_new['219GHz_SM_obsmad_int_Jy']
        # Remaining dust emission at 219GH
        Leroy36df_new['Sprime219_SM_dust_int_Jy'] = Leroy36df_new['219GHz_SM_obsmad_int_Jy']-Leroy36df_new['Sprime219_SM_nondust_int_Jy']
        Leroy36df_new['111GHz_219GHz_sp_ind_int_vlast'] = np.log10(Leroy36df_new['S219_SM_nondust_int_Jy']/(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*Leroy36df_new['219_111_sqr_size_ratio']))/np.log10(f219/f111)
        Leroy36df_new['Size_limit'] = np.sqrt(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*(Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']**2)*((f219/f111)**(Leroy36df_new['36GHz_111GHz_sp_ind_int_vlast']))/Leroy36df_new['S219_SM_nondust_int_Jy'])
        Leroy36df_new['Size_limit2'] = np.sqrt(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*(Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']**2)*((f219/f111)**(Leroy36df_new['111GHz_219GHz_sp_ind_int_vlast']))/Leroy36df_new['Sprime219_SM_nondust_int_Jy'])
        Leroy36df_new['219_111_sqr_size_limit_ratio'] = (Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']/Leroy36df_new['Size_limit'])**2
        Leroy36df_new['111GHz_219GHz_sp_ind_int_vlast_size_limit'] = np.log10(Leroy36df_new['S219_SM_nondust_int_Jy']/(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*Leroy36df_new['219_111_sqr_size_limit_ratio']))/np.log10(f219/f111)
        subdf = Leroy36df_new.dropna(subset = ['Source_altern_sub_final'])
        for i, row in subdf.iterrows():
            print(f'{row["Source_altern_sub_final"]} \t\t {row["Sprime219_SM_nondust_samesize_int_fraction"]:1.2f} \t\t {row["Sprime219_SM_nondust_int_Jy"]*1000:1.2f}  \t\t {row["Sprime219_SM_nondust_samesize_int_Jy"]*1000:1.2f}')
        spind_219_111_thin = -0.1
        Leroy36df_new['Size_limit_opt_thin'] = np.sqrt(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*(Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']**2)*((f219/f111)**(spind_219_111_thin))/Leroy36df_new['S219_SM_nondust_int_Jy'])
        Leroy36df_new['219_111_sqr_size_limitthin_ratio'] = (Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit']/Leroy36df_new['Size_limit_opt_thin'])**2
        Leroy36df_new['111GHz_219GHz_sp_ind_int_vlast_size_limitthin'] = np.log10(Leroy36df_new['S219_SM_nondust_int_Jy']/(Leroy36df_new['111GHz_SM_obsmad_int_Jy']*Leroy36df_new['219_111_sqr_size_limitthin_ratio']))/np.log10(f219/f111)
        # Eq.11 Murphy
        Leroy36df_new['S219_SM_nondust_int_W/m2'] = u_conversion.jy_to_wm2(Leroy36df_new['S219_SM_nondust_int_Jy'], f219)
        Leroy36df_new['S219_SM_nondust_int_Lsun'] = u_conversion.flux_to_lum(Leroy36df_new['S219_SM_nondust_int_W/m2'], D_Mpc, z=False)
        Leroy36df_new['S219_SM_nondust_int_L_erg/s'] =  u_conversion.lsun_to_ergs(Leroy36df_new['S219_SM_nondust_int_Lsun'])
        Leroy36df_new['S219_SM_obsmad_int_W/m2'] = u_conversion.jy_to_wm2(Leroy36df_new['219GHz_SM_obsmad_int_Jy'], f219)
        Leroy36df_new['S219_SM_obsmad_int_Lsun'] = u_conversion.flux_to_lum(Leroy36df_new['S219_SM_obsmad_int_W/m2'], D_Mpc, z=False)
        Leroy36df_new['S219_SM_obsmad_int_L_erg/s'] =  u_conversion.lsun_to_ergs(Leroy36df_new['S219_SM_obsmad_int_Lsun'])
        Leroy36df_new['AllSFR_thermal_Msun_yr'] = 4.6e-28*((Te/1e4)**(-0.45))*f219*Leroy36df_new['S219_SM_obsmad_int_L_erg/s']/219.1e9
        Leroy36df_new['nodustSFR_thermal_Msun_yr'] = 4.6e-28*((Te/1e4)**(-0.45))*f219*Leroy36df_new['S219_SM_nondust_int_L_erg/s']/219.1e9
        Leroy36df_new.to_excel(results_path+'Leroy_36GHz_python_219_wspind_vfinal.xlsx')
        subdf = Leroy36df_new.dropna(subset = ['Source_altern_sub_final'])
        for i, row in subdf.iterrows():
            print(f'{row["Source_altern_sub_final"]}  \t\t {row["111GHz_219GHz_sp_ind_int_vlast_size_limit"]:1.2f} \t {row["111GHz_219GHz_sp_ind_int_vlast_size_limitthin"]:1.2f}')   
        return Leroy36df_new
    
# Print latex table with fluxes
if print_fluxes_latex:
    def print_fluxes_latex():
        """
            Print latex table with fluxes
        """
        line_table = { 
                    'v=0_24_23_SM':                 [r'$v=0$',  '24 - 23',           218.324723,   83.7551, 0.8261E-3],
                    'v=0_26_25_SM':                 [r'$v=0$',  '26 - 25',           236.5127888, 98.6235, 0.1052E-2],
                    'v7=1_24_1_23_-1_SM':           [r'$v_{7}=1$',  '24,1 - 23,-1',  219.1737567, 307.0811, 0.8303E-3],
                    'v7=1_26_1_25_-1_SM':           [r'$v_{7}=1$',  '26,1 - 25,-1',  237.432261,  322.0074, 0.1058E-2],
                    'v7=2_24_0_23_0_SM':            [r'$v_{7}=2$',  '24,0 - 23,0',   219.6751141, 530.2792, 0.83373E-3],
                    'v7=2_26_0_25_0_SM':            [r'$v_{7}=2$',  '26,0 - 25,0',   237.9688357, 545.2395, 0.1061E-2],
                    'v6=1_24_-1_23_1_SM':           [r'$v_{6}=1$',  '24,-1 - 23,1',  218.6825609, 582.7042, 0.8265E-3],
                    'v6=1_26_-1_25_1_SM':           [r'$v_{6}=1$',  '26,-1 - 25,1',  236.900358,  597.597, 0.1053E-2],
                    'v5=1_v7=3_26_1_0_25_-1_0_SM':  [r'$v_{5}=1/v_7=3$',  '26,1,0 - 25,-1,0',  236.661413, 761.7543, 0.1038E-3],
                    'v6=v7=1_26_2_25_-2_SM':        [r'$v_{6}=v_{7}=1$',  '26,2,2 - 25,-2,2',  237.81488, 820.3528, 0.1055E-2],
                    'v4=1_26_25_SM':                [r'$v_{4}=1$',  '26 - 25',      236.18406, 964.2095, 0.1010E-2],
                    'v6=2_24_0_23_0_SM':            [r'$v_{6}=2$',  '24,0 - 23,0',  219.0235155, 1097.1952, 0.8012E-3],
            }
        for l, line in enumerate(line_table):
            u_conversion.Energy_cmtoK(line_table[line][3])
            print(line_table[line][4])
        # Latex printing!
        for i,row in new_hb_df.iterrows():
            line_string = '$'
            for l, line in enumerate(line_table):
                val     = row[line+"_mJy_kms_beam_orig"]
                val_err = row[line+"_mJy_kms_beam_orig_err"]
                if 3*val_err > val:
                    prval = 3*val_err
                    prvalerr = np.nan
                else:
                    prval = val
                    prvalerr =  val_err
                line_string = line_string+' & $'+f'{prval:1.1f}\pm$ & ${prvalerr:1.1f}$ &'
            line_string = line_string  +'\\\\'
            print(f'${row["dist"]}'+line_string)
        



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
violet      = sns.xkcd_palette(['violet'])[0]
bred        = sns.xkcd_palette(['blood red'])[0]
yellow      = sns.xkcd_palette(['orange yellow'])[0]
dviolet      = sns.xkcd_palette(['light magenta'])[0]
dazure      = sns.xkcd_palette(['bright blue'])[0]
dgreen      = sns.xkcd_palette(['darkish green'])[0]

    
if models_calculations:
    def models_calculations(results_path, source, D_Mpc):
        distance_pc = D_Mpc*1e6
        # Reading saved model summary
        TcenAGN = 1300 # Temperatural central del AGN
        k100 = 44.5             # Opacidad a 100 um cm^2 /g 
        mh = 1.67352*10**-24    # M hidrogeno g
        atomic_hydrogen_mass_kg = 1.6737236E-27*u.kg
        light_to_mass = 1000
        gas_to_dust = 100
        SHC13_MZAMS_Msun = 0.63E5 # From Leroy2018
        vel_disp = 25#31#utiles.fwhm_to_stdev(31,0)[0] # km/s FWHM o sigma
        sigma_disp = utiles.fwhm_to_stdev(vel_disp,0)[0]
        kappa_mean = 10 # cm2 g-1
        vel_diff = 21 # km/s Difference between peaks 
        modsum_df = pd.read_excel(f'{results_path}{source}_finalmods_modsummary_Rcrit'+str(Rcrit).replace('.','p')+'.xlsx', header=0)
        lum_type = ['Ltot_name_Lsun', 'Ltot_Lsun']
        mass_type = ['Mgas_Msun', 'Mgas_Msun_corr']
        for lum in lum_type:
            if 'name' in lum:
                sublum = '_name'
            else:
                sublum = ''
            for mass in mass_type:
                if 'corr' in mass:
                    submass = '_corr'
                else:
                    submass = ''
                modsum_df['LIR'+sublum+'/Mgas'+submass] = modsum_df[lum]/modsum_df[mass]
            modsum_df['SigmaIR'+sublum+'_Lsun_pc2'] = modsum_df[lum]/(np.pi*modsum_df['R_pc']**2)
        for i,row in modsum_df.iterrows():
            print(f'{i}  & $ {row["q"]:1.1f} $ & $  {row["Ltot_Lsun"]:1.1e} $ & $  {row["SigmaIR_Lsun_pc2"]:1.1e} $ & $ {row["Mgas_Msun_corr"]:1.1e} $ & $ {row["LIR/Mgas_corr"]:1.0f} $ & $ {row["half_rad_pcv2"]:1.1f} $ & $ {row["NH2_corr"]:1.1e} $ & $ {row["NHC3N"]:1.1e} $')
        # AGN luminosity size
        modsum_df['RAGN_m'] = np.sqrt(modsum_df['Ltot_Lsun']*((1*u.Lsun).to(u.W)/(4*np.pi*_si.sigma_sb*(TcenAGN*u.K)**4)).value)
        modsum_df['RAGN_pc'] = modsum_df['RAGN_m']*(1*u.m).to(u.pc).value
        # All densities are from the corrected mass
        # All times are using the corrected mass
        # Free-fall time
        modsum_df['tauff_yr'] = np.sqrt(3*np.pi/(32*_si.G*modsum_df['nH2_kg_m3']))*(1*u.s).to(u.yr).value
        # Dynamical time
        modsum_df['tdyn_yr'] = (1/np.sqrt(_si.G*modsum_df['nH2_kg_m3']))*(1*u.s).to(u.yr).value
        # Edd luminosity
        G_cm3_Msun_s2 = _si.G.to(u.cm**3/u.Msun/u.s**2)
        modsum_df['Lum_edd_Lsun'] = (4*np.pi*G_cm3_Msun_s2*_si.c.to(u.cm/u.s).value*modsum_df['Mgas_Msun']/(kappa_mean))*(1*u.cm**2*u.g/u.s**3).to(u.Lsun).value
        modsum_df['Lum_edd_Lsun_corr'] = (4*np.pi*G_cm3_Msun_s2*_si.c.to(u.cm/u.s).value*modsum_df['Mgas_Msun_corr']/(kappa_mean))*(1*u.cm**2*u.g/u.s**3).to(u.Lsun).value
        # Mprotostars assuming lum_to_mass
        modsum_df['Mprotostars_name_Msun'] = modsum_df['Ltot_name_Lsun']/light_to_mass
        modsum_df['Mprotostars_Msun'] = modsum_df['Ltot_Lsun']/light_to_mass
        # Total mass (with proto-stars) Faltaria sumar las ZAMS?
        modsum_df['Mtotal_corr'] = modsum_df['Mprotostars_Msun']+modsum_df['Mgas_Msun_corr']
        # Total dens (with proto-stars)
        modsum_df['dens_withstars_kg_m3'] = modsum_df['Mtotal_corr']*(1*u.Msun).to(u.kg).value/(4/3*np.pi*(modsum_df['R_pc']*(1*u.pc).to(u.m).value)**3)
        # Free-fall time with proto-stars
        modsum_df['tauff_stars_yr'] = np.sqrt(3*np.pi/(32*_si.G*modsum_df['dens_withstars_kg_m3']))*(1*u.s).to(u.yr).value
        # Virial mass
        modsum_df['Mvir_Msun'] = 2*((vel_disp*1000)**2*modsum_df['R_pc']*(1*u.pc).to(u.m).value/_si.G.value)*(1*u.kg).to(u.Msun).value
        modsum_df['Mvir_sigma_Msun'] = 2*((sigma_disp*1000)**2*modsum_df['R_pc']*(1*u.pc).to(u.m).value/_si.G.value)*(1*u.kg).to(u.Msun).value
        # Viral Parameter
        # alpha < 1.0 Bound
        # alpha > 2.0 Unbound 
        modsum_df['alpha_vir_fromsigma'] = 5*(sigma_disp*1000)**2*modsum_df['R_pc']*(1*u.pc).to(u.m).value/(_si.G.value*modsum_df['Mgas_Msun_corr']*(1*u.Msun).to(u.kg).value)
        modsum_df['alpha_vir_fromFWHM']  = 5*(vel_disp*1000)**2*modsum_df['R_pc']*(1*u.pc).to(u.m).value/(_si.G.value*modsum_df['Mgas_Msun_corr']*(1*u.Msun).to(u.kg).value)
        # SFE from mass
        modsum_df['SFE_mass'] = modsum_df['Mprotostars_Msun']/modsum_df['Mtotal_corr']
        modsum_df['SFE_total_mass'] = (SHC13_MZAMS_Msun+modsum_df['Mprotostars_Msun'])/(modsum_df['Mprotostars_Msun']+modsum_df['Mgas_Msun_corr']+SHC13_MZAMS_Msun)
        # SFE per free-fall time in Myr
        modsum_df['SFE_total_mass_ff'] = modsum_df['SFE_total_mass']/(modsum_df['tauff_yr']/1e6)

        modsum_df['SFE_mass_ff'] = modsum_df['SFE_mass']/(modsum_df['tauff_yr']/1e6)
        # SF per free_fall
        modsum_df['SFR_total_mass_ff_Msun_yr'] = (SHC13_MZAMS_Msun+modsum_df['Mprotostars_Msun'])/modsum_df['tauff_yr']
        modsum_df['SFR_mass_ff_Msun_yr'] = modsum_df['Mprotostars_Msun']/(modsum_df['tauff_yr'])
        # Depletion time
        modsum_df['t_dep_total_mass_ff_yr'] = modsum_df['Mtotal_corr']/modsum_df['SFR_total_mass_ff_Msun_yr']
        modsum_df['t_dep_mass_ff_yr'] = modsum_df['Mtotal_corr']/modsum_df['SFR_mass_ff_Msun_yr']
        # SFR from LIR 
        # Kennicutt1998a Salpeter IMF or Piero Madau 2014
        modsum_df['SFR_salpeter_Msun_yr'] = 4.5e-44 * modsum_df['Ltot_Lsun']*(1.*u.Lsun).to(u.erg/u.s).value
        # Hayward2015 Kroupa IMF
        modsum_df['SFR_kroupa_Msun_yr'] = 3.7e-37 * modsum_df['Ltot_Lsun']*(1.*u.Lsun).to(u.watt).value
        # Superficial SFR Krumholz & McKee 2005
        modsum_df['SFR_Msun_yr_krumholz'] = modsum_df['SFE_mass']*modsum_df['Mgas_Msun_corr']/modsum_df['tauff_yr']
        # Superficial SFR i.e. Kennicutt-Schmidt law
        modsum_df['SigmaSFR_Msun_yr_pc2'] = modsum_df['SigmaH2_Msun_pc2']**1.4 
        # External pressure due colliding clouds
        # Kisetsu2021
        PI = 0.5
        modsum_df['Pe_kisetsu_K_cm3'] = PI*modsum_df['nH2_cm3']*(vel_diff**2)
        # Johnson2015
        kboltz = _si.k_B.to(u.m**2*u.kg*u.s**-2*u.K**-1)
        modsum_df['Pe_johnson_K_cm3'] = kboltz**-1*3*PI*modsum_df['Mgas_Msun_corr']*(1*u.Msun).to(u.kg).value*((sigma_disp*1000)**2)/(4*np.pi*(modsum_df['R_pc']*(1*u.pc).to(u.cm).value)**3)
        # Johnson2015
        modsum_df['Pe_sizeLinewidth_km2_s2_pc'] = vel_disp**2/modsum_df['R_pc']
        # Finn2019 Ram pressure
        modsum_df['Pram_Finn_K_cm3'] = modsum_df['nH2_kg_m3']*((u.kg/u.m**3)*(vel_diff*1000*u.m/u.s)**2/kboltz).to(u.K/u.cm**3).value
        # Levy2021
        Vmaxoutflow = 250 + 2*vel_diff/(2. * np.sqrt(2* np.log(2)))
        modsum_df['Mgas_Msun_corr']*(1*u.Msun).to(u.kg)/(4/3*np.pi*((1.5*u.pc).to(u.m))**3)
        # Selecting model 2 (ie bestfit model)
        modsum_df = modsum_df[modsum_df['model']=='m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9']
        # Model error in luminosity
        Lmod_err = 0.5 # 50%
        bgn_df = pd.read_excel(f'{results_path}BGN_properties.xlsx', header=0)
        hc_df  = pd.read_excel(f'{results_path}HCs_properties.xlsx', header=0)
        pfalzner_df = pd.read_excel(f'{results_path}SSC_tables/pfalzner2009.xlsx', header=1)
        pfalzner_df['M_Msun'] = 10**pfalzner_df['log_Mc_Msun']
        pfalzner_df['Lum_Lsun'] = pfalzner_df['M_Msun']*light_to_mass
        pfalzner_df['SigmaIR_Lsun_pc2'] = pfalzner_df['Lum_Lsun']/(np.pi*(pfalzner_df['size_pc']/2)**2)
        MK_sun = 3.28
        lada_df = pd.read_excel('/Users/frico/Documents/data/NGC253_HR/Results_v2/SSC_tables/Lada2003.xlsx', header=1)
        lada_df['Lum_Lsun'] = lada_df['Mass_Msun']*light_to_mass
        lada_df['SigmaIR_Lsun_pc2'] = lada_df['Lum_Lsun']/(np.pi*(lada_df['size_pc']/2)**2)
        Mv_sun = 4.8
        portout_df = pd.read_excel('/Users/frico/Documents/data/NGC253_HR/Results_v2/SSC_tables/Portegies2010_outLG.xlsx', header=1, na_values='-')
        portout_df['Mv_mag'] = portout_df['Mv_mag'].astype(float)
        portout_df['Lum_Lsun'] = 10**(0.4*(Mv_sun-portout_df['Mv_mag']))
        portout_df['SigmaIR_Lsun_pc2'] = portout_df['Lum_Lsun']/(np.pi*(portout_df['r_eff_pc'])**2)
        portin_df = pd.read_excel('/Users/frico/Documents/data/NGC253_HR/Results_v2/SSC_tables/Portegies2010_inLG.xlsx', header=1, na_values='-')
        portin_df['Mv_mag'] = portin_df['Mv_mag'].astype(float)
        portin_df['Lum_Lsun'] = 10**(0.4*(Mv_sun-portin_df['Mv_mag']))#light_to_mass*10**portin_df['log_Mphot']#
        portin_df['SigmaIR_Lsun_pc2'] = portin_df['Lum_Lsun']/(np.pi*(portin_df['r_eff_pc'])**2)
        hc_df['SigmaIR_Lsun_pc2'] = hc_df['LIR_Lsun']/(np.pi*hc_df['r_pc']**2)
        hc_df['SigmaH2_Msun_pc2'] = hc_df['Mass_Msun']/(np.pi*hc_df['r_pc']**2)
        hc_df['LIR/Mgas'] = hc_df['LIR_Lsun']/hc_df['Mass_Msun']
        # We have Mass for total both Arp together.
        # We assume the have a similar percentage in mass than in luminosity
        apr_df = bgn_df[bgn_df['Source'].str.contains('Arp')]
        total_lum1 = np.nansum(apr_df['LIR_Lsun'])
        total_lum2 = np.nansum(apr_df['LIR2_Lsun'])
        for i, row in bgn_df.iterrows():
            if 'Arp' in row['Source']:
                lum_percen1 = row['LIR_Lsun']/total_lum1
                lum_percen2 = row['LIR2_Lsun']/total_lum2
                l_perc = np.nanmean([lum_percen1, lum_percen2])
                bgn_df.loc[i, 'MH2_Msun'] = l_perc*row['MH2_Msun']
        bgn_df['SigmaH2_Msun1_pc2'] = bgn_df['MH2_Msun']/(np.pi*bgn_df['R_out_pc']**2)
        bgn_df['SigmaH2_Msun2_pc2'] = bgn_df['MH2_Msun']/(np.pi*bgn_df['R_out2_pc']**2)
        bgn_df['LIR/Mgas'] = bgn_df['LIR_Lsun']/bgn_df['MH2_Msun']
        bgn_df['LIR2/Mgas'] = bgn_df['LIR2_Lsun']/bgn_df['MH2_Msun']
        # Removing some Hot Cores
        ### Rolffs2011
        rolffs_df = hc_df[hc_df['References']=='Rolffs2011a']
        rolffs_df['r_pc'] = u_conversion.lin_size(rolffs_df['D_kpc']/1000, rolffs_df['FWHM_arcsec'])*(1*u.m).to(u.pc).value
        rolffs_df['rmod_pc'] = rolffs_df['R_ph_au']*(1*u.au).to(u.pc).value
        rolffs_df['rmod_out_pc'] = rolffs_df['R_out_au']*(1*u.au).to(u.pc).value
        #rolffs_df['r_pc'] = rolffs_df['rmod_out_pc'] 
        rolffs_df['LIR/Mgas'] = rolffs_df['LIR_Lsun']/rolffs_df['Mass_Msun']
        rolffs_df['SigmaIR_Lsun_pc2'] = rolffs_df['LIR_Lsun']/(np.pi*rolffs_df['r_pc']**2)
        rolffs_df['SigmaH2_Msun_pc2'] = rolffs_df['Mass_Msun']/(np.pi*rolffs_df['r_pc']**2)
        rolffs_df['LIRmod_Lsun'] = rolffs_df['LIR_mod']
        rolffs_df['LIRmod/Mgas'] = rolffs_df['LIR_mod']/rolffs_df['Mass_Msun']
        rolffs_df['rmod_pc'] = rolffs_df['R_ph_au']*(1*u.au).to(u.pc).value
        rolffs_df['rmod_out_pc'] = rolffs_df['R_out_au']*(1*u.au).to(u.pc).value
        rolffs_df['SigmaIRmod_Lsun_pc2'] = rolffs_df['LIRmod_Lsun']/(np.pi*rolffs_df['rmod_out_pc']**2)
        rolffs_df['SigmaH2mod_Msun_pc2'] = rolffs_df['Mass_Msun']/(np.pi*rolffs_df['rmod_out_pc']**2)
        # Modelled values
        modelled_values_rolffs = False
        if modelled_values_rolffs:
            rolffs_df['r_pc'] = rolffs_df['rmod_out_pc'] 
            rolffs_df['LIR_Lsun'] = rolffs_df['LIR_mod']
            rolffs_df['SigmaIR_Lsun_pc2'] = rolffs_df['SigmaIRmod_Lsun_pc2']
            rolffs_df['SigmaH2_Msun_pc2'] = rolffs_df['SigmaH2mod_Msun_pc2']
        ### deVicente
        deVicente_df = hc_df[hc_df['References']=='deVicente2000']
        # Correcting devicente luminosities by a factor 10
        deVicente_df['LIR_cor_Lsun'] = deVicente_df['LIR_Lsun']/10
        deVicente_df['SigmaIR_Lsun_pc2'] = deVicente_df['LIR_cor_Lsun']/(np.pi*deVicente_df['r_pc']**2)
        deVicente_df['SigmaH2_Msun_pc2'] = deVicente_df['Mass_Msun']/(np.pi*deVicente_df['r_pc']**2)
        deVicente_df['LIR/Mgas'] = deVicente_df['LIR_cor_Lsun']/deVicente_df['Mass_Msun']
        ### Etxaluze2013, nice lums and mass
        hc_df = hc_df[hc_df['References']=='Etxaluze2013']
        for i, row in bgn_df.iterrows():
            print(f'{np.log10(row["LIR/Mgas"]):1.1f}\t {np.log10(row["LIR2/Mgas"]):1.1f}')
            
        return modsum_df, hc_df, rolffs_df, bgn_df, pfalzner_df, lada_df, portout_df, portin_df

        SHC_compfig_helper.plot_LIR_comp_ALL(modsum_df, hc_df, rolffs_df, bgn_df, results_path
                                            , pfalzner_df, lada_df, portout_df,portin_df, Lmod_err=0.5, only_HC = True)
        
        SHC_compfig_helper.plot_LIR_comp_ALL_big(modsum_df, hc_df, rolffs_df, bgn_df, results_path
                                            , pfalzner_df, lada_df, portout_df,portin_df, Lmod_err=0.5, only_HC = True)
        
        
    ind_plots = False
    if ind_plots:
        
        only_HC = True
        ms = 12
        figsize = 8
        naxis = 2
        maxis = 1
        labelsize = 20
        ticksize = 18
        fontsize = 14
        fig = plt.figure(figsize=(figsize*2.15, figsize*0.85))
        gs = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs.update(wspace = 0.18, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        
        axis = []
        axis.append(fig.add_subplot(gs[0]))
        axis.append(fig.add_subplot(gs[1]))
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='k', label='HCs', markerfacecolor=azure, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='SHC13a', markerfacecolor=redpink, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='BGNs', markerfacecolor=green, markersize=11, linestyle='')]
    
        # Create the figure
        axis[0].legend(handles=legend_elements, loc='upper left', frameon=False, bbox_to_anchor=(0.05,0.95))
        
        # LIR vs R
        #axis[0].plot(np.log10(modsum_df['Ltot_Lsun']), np.log10(modsum_df['R_pc']), marker='o', color=redpink, linestyle ='', markeredgecolor='k')
        
        axis[0].plot(np.log10(modsum_df['R_pc']), np.log10(modsum_df['Ltot_name_Lsun']), marker='o', color=redpink, linestyle ='', markeredgecolor='k', ms = ms)
        modsum_df['log_Ltot_name_err'] = (10**np.log10(Lmod_err*modsum_df['Ltot_name_Lsun']))*(1/np.log(10))/(10**np.log10(modsum_df['Ltot_name_Lsun']))
        axis[0].errorbar(np.log10(modsum_df['R_pc']), np.log10(modsum_df['Ltot_name_Lsun']), 
                                                     yerr=modsum_df['log_Ltot_name_err'],
                                                     marker='o', markersize=ms,
                                                     markerfacecolor=redpink,
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = redpink,
                                                     elinewidth= 0.8,
                                                     barsabove= True,
                                                     zorder=1)
        
        axis[0].plot(np.log10(hc_df['r_pc']), np.log10(hc_df['LIR_Lsun']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[0].plot(np.log10(rolffs_df['r_pc']), np.log10(rolffs_df['LIR_Lsun']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        for i, row in rolffs_df.iterrows():
            name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
            if row['Source'] in hc_df['Source'].tolist():
                indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
                x_vals = [np.log10(row['r_pc']), np.log10(hc_df.loc[indx, 'r_pc'])]
                y_vals = [np.log10(row['LIR_Lsun']), np.log10(hc_df.loc[indx, 'LIR_Lsun'])]
                axis[0].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
            axis[0].annotate(name, xy=(np.log10(row['r_pc']),np.log10(row['LIR_Lsun'])), xytext=(np.log10(row['r_pc'])+row['LIR_xpos'], np.log10(row['LIR_Lsun'])+row['LIR_ypos']),
                     va='center', color = 'k')
    
        for i, row in bgn_df.iterrows():
            if row['Source'] != 'Arp220E':
                if row['Source'] == 'Arp220W':
                    plname = 'Arp220'
                else:
                    plname = row['Source']
                y_vals = [np.log10(row['LIR_Lsun']), np.log10(row['LIR2_Lsun'])]
                x_vals = [np.log10(row['R_out_pc']), np.log10(row['R_out2_pc'])]
                axis[0].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = green, markeredgecolor='k', ms = ms)
                axis[0].annotate(plname, xy=(x_vals[0],y_vals[0]), xytext=(x_vals[0]+row['LIR_xpos']*x_vals[0],y_vals[0]+row['LIR_ypos']*y_vals[0]),
                         va='center', color = 'k')
            
    
        axis[0].set_xlabel(r'$\log{R}$ (pc)', fontsize=labelsize)
        axis[0].set_ylabel(r'$\log{L_{\text{IR}}}$ (L$_{\odot}$)', fontsize=labelsize)
        
        # LIR /Mgas vs LIR
        #axis[1].plot(modsum_df['LIR/Mgas'], modsum_df['Ltot_Lsun'], marker='o', color=redpink, linestyle ='', markeredgecolor='k')
        axis[1].plot(np.log10(modsum_df['SigmaIR_name_Lsun_pc2']), np.log10(modsum_df['LIR_name/Mgas']), marker='o', color=redpink, linestyle ='', markeredgecolor='k', ms = ms)
        modsum_df['LtotMgas_name_err'] = np.sqrt((Lmod_err*modsum_df['Ltot_name_Lsun']/modsum_df['Mgas_Msun'])**2+(modsum_df['Ltot_name_Lsun']*(0*Lmod_err/2)*modsum_df['Mgas_Msun']/(modsum_df['Mgas_Msun']**2))**2)
        modsum_df['log_LtotMgas_name_err'] = (10**np.log10(modsum_df['LtotMgas_name_err']))*(1/np.log(10))/(10**np.log10(modsum_df['LIR_name/Mgas']))
        axis[1].errorbar(np.log10(modsum_df['SigmaIR_name_Lsun_pc2']), np.log10(modsum_df['LIR_name/Mgas']), 
                                                     yerr=modsum_df['log_LtotMgas_name_err'],
                                                     xerr=modsum_df['log_SigmaIR_name_err'],
                                                     marker='o', markersize=ms,
                                                     markerfacecolor=redpink,
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = redpink,
                                                     elinewidth= 0.8,
                                                     barsabove= True,
                                                     zorder=1)
        axis[1].plot(np.log10(hc_df['SigmaIR_Lsun_pc2']), np.log10(hc_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[1].plot(np.log10(rolffs_df['SigmaIR_Lsun_pc2']), np.log10(rolffs_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        for i, row in rolffs_df.iterrows():
            name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
            if row['Source'] in hc_df['Source'].tolist():
                
                indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
                x_vals = [np.log10(row['SigmaIR_Lsun_pc2']), np.log10(hc_df.loc[indx, 'SigmaIR_Lsun_pc2'])]
                y_vals = [np.log10(row['LIR/Mgas']), np.log10(hc_df.loc[indx, 'LIR/Mgas'])]
                axis[1].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
    
            x_val = np.log10(row['SigmaIR_Lsun_pc2'])
            y_val = np.log10(row['LIR/Mgas'])
            #axis[0].annotate(name, xy=(x_val,y_val), xytext=(x_val+row['LIRMH2sig_xpos']*x_val, y_val+row['LIRMH2sig_ypos']*y_val),
            #        va='center', color = 'k')
            axis[1].annotate(name, xy=(x_val,y_val), xytext=(row['LIRMH2sig_xpos'], row['LIRMH2sig_ypos']),
                    va='center', color = 'k')
            #axis[0].annotate(name, xy=(x_val,y_val), xytext=(x_val, y_val),
            #        va='center', color = 'k')
            
        for i, row in bgn_df.iterrows():
            if row['Source'] != 'Arp220E':
                if row['Source'] == 'Arp220W':
                    plname = 'Arp220'
                else:
                    plname = row['Source']
                x_vals = [np.log10(row['SigmaIR_Lsun_pc2']), np.log10(row['SigmaIR2_Lsun_pc2'])]
                y_vals = [np.log10(row['LIR/Mgas']), np.log10(row['LIR2/Mgas'])]
                axis[1].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = green, markeredgecolor='k', ms = ms)
                axis[1].annotate(plname, xy=(x_vals[0],y_vals[0]), xytext=(x_vals[0]+row['LIRMH2sig_xpos']*x_vals[0],y_vals[0]+row['LIRMH2sig_ypos']*y_vals[0]),
                          va='center', color = 'k')
        for v,ax in enumerate(axis):
            axis[v].tick_params(direction='in')
            axis[v].tick_params(axis="both", which='major', length=8)
            axis[v].tick_params(axis="both", which='minor', length=4)
            axis[v].xaxis.set_tick_params(which='both', top ='on')
            axis[v].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axis[v].tick_params(axis='both', which='major', labelsize=ticksize)
            #axis[v].xaxis.set_minor_locator(minor_locator)
            axis[v].tick_params(labelleft=True,
                           labelright=False)
        
        axis[1].set_xlabel(r'$\log{\Sigma_{\text{IR}}}$ (L$_{\odot}$/pc$^{2}$)', fontsize=labelsize)
        axis[1].set_ylabel(r'$\log{L_{\text{IR}}/M_{\text{H2}}}$ (L$_{\odot}$/M$_{\odot}$)', fontsize=labelsize)
            
        fig.savefig(results_path+'LIR_R_LIRMgasSigma.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
        
        only_HC = True
        ms = 12
        figsize = 8
        naxis = 1
        maxis = 1
        labelsize = 20
        ticksize = 18
        fontsize = 14
        fig = plt.figure(figsize=(figsize*1.15, figsize*0.85))
        gs = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs.update(wspace = 0.18, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        
        axis = []
        axis.append(fig.add_subplot(gs[0]))
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='k', label='HCs', markerfacecolor=azure, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='SHC13a', markerfacecolor=redpink, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='BGNs', markerfacecolor=green, markersize=11, linestyle='')]
    
        # Create the figure
        axis[0].legend(handles=legend_elements, loc='upper left', frameon=False)
        
        # LIR /Mgas vs LIR
        #axis[1].plot(modsum_df['LIR/Mgas'], modsum_df['Ltot_Lsun'], marker='o', color=redpink, linestyle ='', markeredgecolor='k')
        axis[0].plot(np.log10(modsum_df['SigmaIR_name_Lsun_pc2']), np.log10(modsum_df['LIR_name/Mgas']), marker='o', color=redpink, linestyle ='', markeredgecolor='k', ms = ms)
        modsum_df['LtotMgas_name_err'] = np.sqrt((Lmod_err*modsum_df['Ltot_name_Lsun']/modsum_df['Mgas_Msun'])**2+(modsum_df['Ltot_name_Lsun']*(0*Lmod_err/2)*modsum_df['Mgas_Msun']/(modsum_df['Mgas_Msun']**2))**2)
        modsum_df['log_LtotMgas_name_err'] = (10**np.log10(modsum_df['LtotMgas_name_err']))*(1/np.log(10))/(10**np.log10(modsum_df['LIR_name/Mgas']))
        axis[0].errorbar(np.log10(modsum_df['SigmaIR_name_Lsun_pc2']), np.log10(modsum_df['LIR_name/Mgas']), 
                                                     yerr=modsum_df['log_LtotMgas_name_err'],
                                                     xerr=modsum_df['log_SigmaIR_name_err'],
                                                     marker='o', markersize=ms,
                                                     markerfacecolor=redpink,
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = redpink,
                                                     elinewidth= 0.8,
                                                     barsabove= True,
                                                     zorder=1)
        axis[0].plot(np.log10(hc_df['SigmaIR_Lsun_pc2']), np.log10(hc_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[0].plot(np.log10(rolffs_df['SigmaIR_Lsun_pc2']), np.log10(rolffs_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        for i, row in rolffs_df.iterrows():
            name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
            if row['Source'] in hc_df['Source'].tolist():
                
                indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
                x_vals = [np.log10(row['SigmaIR_Lsun_pc2']), np.log10(hc_df.loc[indx, 'SigmaIR_Lsun_pc2'])]
                y_vals = [np.log10(row['LIR/Mgas']), np.log10(hc_df.loc[indx, 'LIR/Mgas'])]
                axis[0].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
    
            x_val = np.log10(row['SigmaIR_Lsun_pc2'])
            y_val = np.log10(row['LIR/Mgas'])
            #axis[0].annotate(name, xy=(x_val,y_val), xytext=(x_val+row['LIRMH2sig_xpos']*x_val, y_val+row['LIRMH2sig_ypos']*y_val),
            #        va='center', color = 'k')
            axis[0].annotate(name, xy=(x_val,y_val), xytext=(row['LIRMH2sig_xpos'], row['LIRMH2sig_ypos']),
                    va='center', color = 'k')
            #axis[0].annotate(name, xy=(x_val,y_val), xytext=(x_val, y_val),
            #        va='center', color = 'k')
            
        for i, row in bgn_df.iterrows():
            row['SigmaIR2_Lsun_pc2']
            x_vals = [np.log10(row['SigmaIR_Lsun_pc2']), np.log10(row['SigmaIR2_Lsun_pc2'])]
            y_vals = [np.log10(row['LIR/Mgas']), np.log10(row['LIR2/Mgas'])]
            axis[0].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = green, markeredgecolor='k', ms = ms)
            axis[0].annotate(row['Source'], xy=(x_vals[0],y_vals[0]), xytext=(x_vals[0]+row['LIRMH2sig_xpos']*x_vals[0],y_vals[0]+row['LIRMH2sig_ypos']*y_vals[0]),
                      va='center', color = 'k')
        for v,ax in enumerate(axis):
            axis[v].tick_params(direction='in')
            axis[v].tick_params(axis="both", which='major', length=8)
            axis[v].tick_params(axis="both", which='minor', length=4)
            axis[v].xaxis.set_tick_params(which='both', top ='on')
            axis[v].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axis[v].tick_params(axis='both', which='major', labelsize=ticksize)
            #axis[v].xaxis.set_minor_locator(minor_locator)
            axis[v].tick_params(labelleft=True,
                           labelright=False)
        
        axis[0].set_xlabel(r'$\log{\Sigma_{\text{IR}}}$ (L$_{\odot}$/pc$^{2}$)', fontsize=labelsize)
        axis[0].set_ylabel(r'$\log{L_{\text{IR}}/M_{\text{H2}}}$ (L$_{\odot}$/M$_{\odot}$)', fontsize=labelsize)
            
        fig.savefig(results_path+'LIRMgas_Sigma.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
        
        
        only_HC = True
        ms = 12
        figsize = 8
        naxis = 1
        maxis = 1
        labelsize = 20
        ticksize = 18
        fontsize = 14
        fig = plt.figure(figsize=(figsize*1.15, figsize*0.85))
        gs = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs.update(wspace = 0.18, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        
        axis = []
        axis.append(fig.add_subplot(gs[0]))
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='k', label='HCs', markerfacecolor=azure, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='SHC13a', markerfacecolor=redpink, markersize=11, linestyle=''),
                           Line2D([0], [0], marker='o', markeredgecolor='k', label='BGNs', markerfacecolor=green, markersize=11, linestyle='')]
    
        # Create the figure
        axis[0].legend(handles=legend_elements, loc='upper left', frameon=False)
        
        # LIR /Mgas vs LIR
        #axis[1].plot(modsum_df['LIR/Mgas'], modsum_df['Ltot_Lsun'], marker='o', color=redpink, linestyle ='', markeredgecolor='k')
        axis[0].plot(np.log10(modsum_df['Ltot_name_Lsun']), np.log10(modsum_df['LIR_name/Mgas']), marker='o', color=redpink, linestyle ='', markeredgecolor='k', ms = ms)
        modsum_df['LtotMgas_name_err'] = np.sqrt((Lmod_err*modsum_df['Ltot_name_Lsun']/modsum_df['Mgas_Msun'])**2+(modsum_df['Ltot_name_Lsun']*(0*Lmod_err/2)*modsum_df['Mgas_Msun']/(modsum_df['Mgas_Msun']**2))**2)
        modsum_df['log_LtotMgas_name_err'] = (10**np.log10(modsum_df['LtotMgas_name_err']))*(1/np.log(10))/(10**np.log10(modsum_df['LIR_name/Mgas']))
        axis[0].errorbar(np.log10(modsum_df['Ltot_name_Lsun']), np.log10(modsum_df['LIR_name/Mgas']), 
                                                     yerr=modsum_df['log_LtotMgas_name_err'],
                                                     xerr=modsum_df['log_Ltot_name_err'],
                                                     marker='o', markersize=ms,
                                                     markerfacecolor=redpink,
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = redpink,
                                                     elinewidth= 0.8,
                                                     barsabove= True,
                                                     zorder=1)
        axis[0].plot(np.log10(hc_df['LIR_Lsun']), np.log10(hc_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[0].plot(np.log10(rolffs_df['LIR_Lsun']), np.log10(rolffs_df['LIR/Mgas']),  marker='o', color=azure, linestyle ='', markeredgecolor='k', ms = ms)
        for i, row in rolffs_df.iterrows():
            name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
            if row['Source'] in hc_df['Source'].tolist():
                
                indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
                x_vals = [np.log10(row['LIR_Lsun']), np.log10(hc_df.loc[indx, 'LIR_Lsun'])]
                y_vals = [np.log10(row['LIR/Mgas']), np.log10(hc_df.loc[indx, 'LIR/Mgas'])]
                axis[0].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
    
            x_val = np.log10(row['LIR_Lsun'])
            y_val = np.log10(row['LIR/Mgas'])
            axis[0].annotate(name, xy=(x_val,y_val), xytext=(row['LIRMH2_xpos'], row['LIRMH2_ypos']),
                     va='center', color = 'k')
            
        for i, row in bgn_df.iterrows():
            if row['Source'] != 'Arp220E':
                if row['Source'] == 'Arp220W':
                    plname = 'Arp220'
                else:
                    plname = row['Source']
                x_vals = [np.log10(row['LIR_Lsun']), np.log10(row['LIR2_Lsun'])]
                y_vals = [np.log10(row['LIR/Mgas']), np.log10(row['LIR2/Mgas'])]
                axis[0].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = green, markeredgecolor='k', ms = ms)
                axis[0].annotate(plname, xy=(x_vals[0],y_vals[0]), xytext=(x_vals[0]+row['LIRMH2_xpos']*x_vals[0],y_vals[0]+row['LIRMH2_ypos']*y_vals[0]),
                        va='center', color = 'k')
        
        axis[0].set_xlabel(r'$\log{L_{\text{IR}}}$ (L$_{\odot}$)', fontsize=labelsize)
        axis[0].set_ylabel(r'$\log{L_{\text{IR}}/M_{\text{H2}}}$ (L$_{\odot}$/M$_{\odot}$)', fontsize=labelsize)
        
        for v,ax in enumerate(axis):
            axis[v].tick_params(direction='in')
            axis[v].tick_params(axis="both", which='major', length=8)
            axis[v].tick_params(axis="both", which='minor', length=4)
            axis[v].xaxis.set_tick_params(which='both', top ='on')
            axis[v].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axis[v].tick_params(axis='both', which='major', labelsize=ticksize)
            #axis[v].xaxis.set_minor_locator(minor_locator)
            axis[v].tick_params(labelleft=True,
                           labelright=False)
            
        fig.savefig(results_path+'LIRMgas_Lir.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
        
if fit_logn:
    mname = 'm24_LTHC3Nsbsig5.5E+07cd1.0E+25q1.5nsh30rad1.5vt5_a4'
    fit_XHC3N, x_profile, shell_dists, model_dists = utiles_nLTEmodel.plot_model_input([mname], my_model_path, figinp_path)
    poly = np.polyfit(model_dists, x_profile, deg=16)
    y_int = np.polyval(poly, model_dists)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(model_dists, y_int)
    ax.plot(model_dists, x_profile)
    ax.set_yscale('log')
    for xhc3n in x_profile:
        print(f'{xhc3n:1.1e}')

if plot_cont:
    NGC253_path = '/Users/frico/Documents/data/NGC253_HR/'
    cont_path = 'Continuums/'
    
    results_path = NGC253_path+'Results_v2/'
    lines_column = {'v7=1_24_1_23_-1_SM':                [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 130], 6, 219.1737567],
                    'v7=1_26_1_25_-1_SM':                [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 130], 2, 237.432261],
                    'v6=1_24_-1_23_1_SM':                [15, r'$v_{6}=1$  24,-1 - 23,1', [-10, 95], 10, 218.6825609],
                    'v6=1_26_-1_25_1_SM':                [15, r'$v_{6}=1$  26,-1 - 25,1', [-10, 95],  3, 236.900358],
                    'v7=2_24_0_23_0_SM':                 [15, r'$v_{7}=2$  24,0 - 23,0',  [-5, 70],  11, 219.6751141],
                    'v7=2_26_0_25_0_SM':                 [15, r'$v_{7}=2$  26,0 - 25,0',  [-5, 70],   4, 237.9688357],
                    'v5=1_v7=3_26_1_0_25_-1_0_SM':       [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-3, 40], 7, 236.661413],
                    'v6=v7=1_v4=1_v7=2_v5=2_26_25_SM':   [10, r'$v_{6}=v_{7}=1$  26,2 - 25,-2', [-3, 30],  8, 237.81488],
                    'v=0_26_25_SM':                      [15, r'$v=0$  26 - 25', [-10, 150], 1, 236.5127888],
                    'v4=1_26_25_SM':                     [10, r'$v_{4}=1$  26 - 25', [-3, 30], 5, 236.18406],
                    'v6=2_24_0_23_0_SM':                 [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 30], 9, 219.0235155],
               }
    
    d05_base0 = '/Users/frico/Documents/data/NGC253_HR/SHC/SHC_13/Madextracted_220_nocontsub_v2/distpc_0p05/origbeam_SMaver/origbeam_SMaver/01_NGC_253_237_BASE0'
    d05_base0_df = pd.read_csv(d05_base0, sep='\t', header=None)
    d05_base0_df.columns=['freq', 'jy_beam']
    d05_base0_df['mjy_beam'] = d05_base0_df['jy_beam']*1000
    d55_base0 = '/Users/frico/Documents/data/NGC253_HR/SHC/SHC_13/Madextracted_220_nocontsub_v2/distpc_0p55/origbeam_SMaver/origbeam_SMaver/01_NGC_253_237_BASE0'
    d55_base0_df = pd.read_csv(d55_base0, sep='\t', header=None)
    d55_base0_df.columns=['freq', 'jy_beam']
    d55_base0_df['mjy_beam'] = d55_base0_df['jy_beam']*1000
    fontsize = 12
    maxis = 2
    naxis = 1
    ds='steps-mid'
    fig = plt.figure()
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    axes=[]
    axes.append(fig.add_subplot(gs1[0]))
    axes.append(fig.add_subplot(gs1[1]))
    axes[0].plot(d05_base0_df['freq'], d05_base0_df['mjy_beam'], drawstyle=ds, color = 'k', lw= 0.7,zorder=1)
    axes[1].plot(d55_base0_df['freq'], d55_base0_df['mjy_beam'], drawstyle=ds, color = 'k', lw= 0.7,zorder=1)
    axes[0].set_ylim([-0.5, 2])
    axes[0].set_xlim([np.nanmin(d05_base0_df['freq']), np.nanmax(d05_base0_df['freq'])])
    axes[1].set_xlim([np.nanmin(d05_base0_df['freq']), np.nanmax(d05_base0_df['freq'])])
    axes[0].hlines(0, xmin=234, xmax=238.1, color='lime', linestyle='--', lw=0.9, zorder=3)
    axes[1].hlines(0, xmin=234, xmax=238.1, color='lime', linestyle='--', lw=0.9, zorder=3)
    for line in lines_column:
        if lines_column[line][-1] > 230:
            axes[0].vlines(lines_column[line][-1], ymin=0, ymax=2, color='r', linestyle='--', lw=1.2,zorder=2)
#            axes[0].text(lines_column[line][-1], 0.95, lines_column[line][1],
#                            horizontalalignment='left',
#                            verticalalignment='top',
#                            fontsize=fontsize,
#                            transform=axes[0].transAxes)
            axes[1].vlines(lines_column[line][-1], ymin=0, ymax=2, color='r', linestyle='--', lw=1.2, zorder=2)
#            axes[1].text(lines_column[line][-1], 0.95, lines_column[line][1],
#                            horizontalalignment='left',
#                            verticalalignment='top',
#                            fontsize=fontsize,
#                            transform=axes[1].transAxes)
    axes[0].text(0.1, 0.95,  'r=0.05 pc',
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[0].transAxes)
    axes[1].text(0.1, 0.95,  'r=0.55 pc',
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[1].transAxes)
    fig.savefig(results_path+'NGC253_contsd0p55_contsd0p05_fluxes.pdf', bbox_inches='tight', transparent=True, dpi=400)
