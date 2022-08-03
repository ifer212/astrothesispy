from astrothesispy.radiative_transfer import NGC253HR_nLTE_modelresults
from astrothesispy.utiles import utiles_nLTEmodel
from astrothesispy.utiles import utiles_plot as plot_utiles
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob

get_chi2_and_ind_plot = False
redo_lum_and_sum = False
plot_models_absorption = False
comp_LTEvsNLTE = False
model_difs = False

 
def nLTE_model_plot(NGC253_path, source, results_path, fig_path, rad_transf_path,
                    D_Mpc = 3.5, Rcrit = 0.85, plot_type = 'SBmods', paper_figs = True,
                    presen_figs = False, fortcomp=False, fig_name = ['', '', '',], fig_format = '.pdf'):
    """
        Plotting model results
        plot_type = "SBmods" plots the SB models (distributed star formation)
        plot_type = "AGNmods" plots the AGN models (centered star formation)
        presen_figs = True plots the figures for presentations (simpler figs)
        paper_figs = True plots the figures for publication
        fortcomp = False avoids fortran compilation to convolve by beam the modelled data (need the compiled .31 files)
    """
    # Radiative transfer models paths
    models_path = f'{rad_transf_path}models/'
    fort_paths = f'{rad_transf_path}{source}/'
    # Loads obs fluxes
    obs_df, new_hb_df, cont_df = NGC253HR_nLTE_modelresults.load_observed_LTE(NGC253_path, source, results_path, fig_path, fort_paths, plot_LTE_lum = False) 
    # Lines for presen images
    if presen_figs:
        line_column = {'plot_conts': [],
                    'v=0_26_25_SM': [15, r'$v=0$  26 - 25', [-10, 111], 1], 
                    'v7=1_24_1_23_-1_SM': [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 111], 6],
                    'v7=1_26_1_25_-1_SM': [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 111], 2],
                    'v6=1_24_-1_23_1_SM': [15, r'$v_{6}=1$  24,-1 - 23,1', [-4, 54], 10],
                    'v6=1_26_-1_25_1_SM': [15, r'$v_{6}=1$  26,-1 - 25,1', [-4, 54], 3],
                    'v7=2_24_0_23_0_SM':  [15, r'$v_{7}=2$  24,0 - 23,0', [-4, 54], 11],
                    'v7=2_26_0_25_0_SM':  [15, r'$v_{7}=2$  26,0 - 25,0', [-4, 54], 4],
                    'v5=1_v7=3_26_1_0_25_-1_0_SM':  [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-4, 34], 7],
                    'v6=v7=1_26_2_25_-2_SM': [10, r'$v_{6}=v_{7}=1$  26,2,2 - 25,-2,2', [-4, 34],  8],
                    'v4=1_26_25_SM': [10, r'$v_{4}=1$  26 - 25', [-3, 23], 5],
                    'v6=2_24_0_23_0_SM':  [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 23], 9],
                    }
    # Lines for paper
    if paper_figs:
        line_column = {'plot_conts': [],
                    'v=0_26_25_SM': [15, r'$v=0$  26 - 25', [-10, 150], 1], 
                    'v7=1_24_1_23_-1_SM': [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 150], 6],
                    'v7=1_26_1_25_-1_SM': [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 150], 2],
                    'v6=1_24_-1_23_1_SM': [15, r'$v_{6}=1$  24,-1 - 23,1', [-10, 74], 10],
                    'v6=1_26_-1_25_1_SM': [15, r'$v_{6}=1$  26,-1 - 25,1', [-10, 74], 3],
                    'v7=2_24_0_23_0_SM':  [15, r'$v_{7}=2$  24,0 - 23,0', [-5, 74], 11],
                    'v7=2_26_0_25_0_SM':  [15, r'$v_{7}=2$  26,0 - 25,0', [-5, 74], 4],
                    'v5=1_v7=3_26_1_0_25_-1_0_SM':  [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-3, 39], 7],
                    'v6=v7=1_26_2_25_-2_SM': [10, r'$v_{6}=v_{7}=1$  26,2,2 - 25,-2,2', [-3, 39],  8],
                    'v4=1_26_25_SM': [10, r'$v_{4}=1$  26 - 25', [-3, 29], 5],
                    'v6=2_24_0_23_0_SM':  [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 29], 9],
                    }
    # Naming flux errors properly
    for l,line in enumerate(line_column):
        if line != 'plot_conts':
            # Cont error already included in line error!  
            new_hb_df[line+'_mJy_kms_beam_orig_errcont'] = new_hb_df[line+'_mJy_kms_beam_orig_err']
            new_hb_df[line+'_mJy_kms_beam_345_errcont'] = new_hb_df[line+'_mJy_kms_beam_345_err']
    # Best SB nLTE models
    if plot_type == 'SBmods':
        cont_modelplot = 'model2'
        modelos = {
                    'model1': ['m13_LTHC3Nsbsig1.3E+07cd1.0E+25q1.0nsh30rad1.5vt5_b3','dustsblum1.2E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.azure, [1, 1, 2.0]],
                    'model2': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.redpink , [1, 2.3, 1.9]],
                    'model3': ['m23_LTHC3Nsbsig5.5E+07cd1.0E+25q1.0nsh30rad1.5vt5_b7','dustsblum5.0E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.green, [1, 1, 1]],
                    'model4': ['m24_LTHC3Nsbsig5.5E+07cd1.0E+25q1.5nsh30rad1.5vt5_a7','dustsblum5.0E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.violet, [1, 1, 1]] 
                    }
    # Best AGN nLTE models (including SB model 2)
    if plot_type == 'AGNmods':
        cont_modelplot = 'model7'
        # Model 2 is SB model (the bestfit model)
        modelos= { 
                        'model2': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.redpink , [1, 1.0, 1.9]],
                        'model5': ['agn4_LTHC3Nagnsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_a6','dustagnlum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, plot_utiles.dviolet, [1, 1, 1.0]],
                        'model6': ['agn5_LTHC3Nagnsig1.3E+07cd1.0E+25q1.0nsh30rad1.5vt5_a2','dustagnlum1.2E+10cd1.0E+25exp1.0nsh1002rad17',
                                    1.5, plot_utiles.dazure, [1, 1, 1.0]],
                        'model7': ['agn9_LTHC3Nagnsig1.3E+07cd5.6E+24q1.0nsh30rad1.5vt5_x4','dustagnlum1.2E+10cd5.6E+24exp1.0nsh564rad17',
                                    1.5, plot_utiles.dgreen, [1, 1, 1.0]],
                    }
        
    # Final figures for present
    if presen_figs:
        utiles_nLTEmodel.plot_models_and_inp_finalfig_diap(Rcrit, line_column, modelos, new_hb_df, cont_df, models_path, fig_path, fig_path,
                                                           fort_paths, results_path, D_Mpc = D_Mpc, cont_modelplot = cont_modelplot, fortcomp=fortcomp)

    # Final paper figures for publication
    if paper_figs:
        convolve = True # Using or not convolved fluxes (for adjacent rings)
        
        utiles_nLTEmodel.plot_models_and_inp_finalpaperfig(convolve, Rcrit, line_column, modelos, new_hb_df, cont_df, models_path, fig_path, fig_path,
                                                           fort_paths, results_path, D_Mpc = D_Mpc, cont_modelplot = cont_modelplot, fortcomp=fortcomp,
                                                           fig_name=fig_name, fig_format = fig_format)

if get_chi2_and_ind_plot:
    for i,modelo in enumerate(modelos):
        modsing = {modelo: modelos[modelo]}
        utiles_nLTEmodel.line_profiles_chi2(new_hb_df, line_column, modsing, fort_paths, models_path, Rcrit, results_path)

if redo_lum_and_sum:
    slines = ['ring', 'dist',
              'v=0_24_23_SM_mJy_kms_beam_orig',
              'v=0_26_25_SM_mJy_kms_beam_orig',
              'v7=1_24_1_23_-1_SM_mJy_kms_beam_orig',
              'v7=1_26_1_25_-1_SM_mJy_kms_beam_orig',
              'v6=1_24_-1_23_1_SM_mJy_kms_beam_orig',
              'v6=1_26_-1_25_1_SM_mJy_kms_beam_orig']
    save_obs_df = new_hb_df[slines]
    save_obs_df.to_excel(finalfigmod_path+'Observed_fluxes.xlsx')    
    Rcrit = 0.85 # pc Size of the free-free region (where 235/345 starts to rise)
    redo_lum = False
    if redo_lum:
        distance_pc = D_Mpc*(1*u.Mpc).to(u.pc).value
        modsum_df = pd.DataFrame()
        for mod in modelos:
            modelo = modelos[mod][0]
            dustmod = modelos[mod][1]
            mod_dict = utiles_nLTEmodel.model_summary(modelo, dustmod, models_path, distance_pc, Rcrit)
            modsum_df = modsum_df.append(mod_dict, ignore_index=True)
        modsum_df.to_excel(finalfigmod_path+SHC+'_finalmod_modsummary_Rcrit'+str(Rcrit).replace('.','p')+'.xlsx')  
        
if plot_models_absorption:
    Rcrit = 0.85
    bestmod = {'model2': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, redpink , [1, 2.3, 1.9]]}
    modelo = 'm28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9'
    model2_list = glob(models_path+'m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9_*.inp')
    dustmod = 'dustsblum1.2E+10cd1.0E+25exp1.0nsh1002rad17'
    rad = 1.5
    modabs_dict = {}
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(model2_list)))
    colors = cm.gist_rainbow(np.linspace(0, 1, len(model2_list)))
    cmap_name = 'my_list'
    colorlist = [(0.847, 0.057, 0.057),
                 (0.527, 0.527, 0),
                 (0, 0.592, 0),
                 (0, 0.559, 0.559),
                 (0.316, 0.316, 0.991),
                 (0.718, 0, 0.718)]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colorlist, N= len(model2_list)) 
    sort_dict = {}
    for m,mod in enumerate(model2_list):
        modelo = mod.split('/')[-1].split('.inp')[0]
        mname = modelo.split('_')[-1]
        sort_dict[mname] = modelo
    colors = cm.gist_rainbow(np.linspace(0, 1, len(sort_dict)))
    colors_r = colors#[::-1]
    for m,mod in enumerate(modelos):
        modelo = sort_dict[str(m)]
        modabs_dict[str(m)] = [modelo, dustmod, rad, colors_r[m], [1, 2.3, 1.9]]
    modelos = modabs_dict
    utiles_nLTEmodel.plot_models_and_inp_abscompfig(Rcrit, line_column, modelos, new_hb_df, cont_df, models_path, finalfigmod_path, finalfigmod_path, fort_paths)
    
if comp_LTEvsNLTE:
    new_hb_df['v=0_24_23_SM_mJy_kms_beam_345_errcont'] = 1*new_hb_df['v=0_26_25_SM_mJy_kms_beam_345_errcont']
    new_hb_df['v=0_24_23_SM_mJy_kms_beam_orig_errcont'] = 1*new_hb_df['v=0_26_25_SM_mJy_kms_beam_orig_errcont']
    Rcrit = 0.85
    compmods1= { 'LTE 3E17': ['m28_cLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, redpink , [1, 1.0, 1.9], '-', True],
                'LTE 3E16': ['m28_cLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9_low','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, dviolet, [1, 1.0, 1.9], '-', True],
                'LTE2 3E16': ['mc_cLTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5vt5_b9_low','dustsblum1.0E+11cd1.0E+25exp1.5nsh1003rad17',
                            1.5, green, [1, 1.0, 1.9], '-', True],
                'LTE3 1E16': ['mc_cLTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5vt5_b9_vlow','dustsblum1.0E+11cd1.0E+25exp1.5nsh1003rad17',
                            1.5, dgreen, [1, 1.0, 1.9], '-', True],
                'NLTE 3E16': ['m28_NLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustagnlum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, violet, [1, 1.0, 1.9], '--', False],
                'NLTE2 3E16': ['mc_NLTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.0E+11cd1.0E+25exp1.5nsh1003rad17',
                            1.5, dazure, [1, 1.0, 1.9], '--', False],
                'NLTE3 1E16': ['mc_NLTHC3Nsbsig1.1E+08cd1.0E+25q1.5nsh30rad1.5vt5_b9_vlow','dustsblum1.0E+11cd1.0E+25exp1.5nsh1003rad17',
                            1.5, 'cyan', [1, 1.0, 1.9], '--', False],
               }
    compmods= { 'LTE 3E17': ['m28_cLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, redpink , [1, 1.0, 1.9], '-', True],
                'NLTE 3E17': ['m28_NLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9_up','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, azure, [1, 1.0, 1.9], '--', False],
                'NLTE 1E16': ['m28_NLTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9_low','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, green, [1, 1.0, 1.9], '--', False],
               }
    modelos = compmods
    utiles_nLTEmodel.plot_models_and_inp_comp(Rcrit, line_column, modelos, new_hb_df, cont_df, models_path, finalfigmod_path, finalfigmod_path, fort_paths)
    
    slines = ['ring', 'dist',
              'v=0_24_23_SM_mJy_kms_beam_orig',
              'v=0_26_25_SM_mJy_kms_beam_orig',
              'v7=1_24_1_23_-1_SM_mJy_kms_beam_orig',
              'v7=1_26_1_25_-1_SM_mJy_kms_beam_orig',
              'v6=1_24_-1_23_1_SM_mJy_kms_beam_orig',
              'v6=1_26_-1_25_1_SM_mJy_kms_beam_orig']
    
    save_obs_df = new_hb_df[slines]
    save_obs_df.to_excel(finalfigmod_path+'Observed_fluxes.xlsx')    
    modelos = compmods
    Rcrit = 0.85 # pc Size of the free-free region (where 235/345 starts to rise)
    redo_lum = False
    if redo_lum:
        distance_pc = D_Mpc*(1*u.Mpc).to(u.pc).value
        modsum_df = pd.DataFrame()
        for mod in modelos:
            modelo = modelos[mod][0]
            dustmod = modelos[mod][1]
            mod_dict = utiles_nLTEmodel.model_summary(modelo, dustmod, models_path, distance_pc, Rcrit)
            modsum_df = modsum_df.append(mod_dict, ignore_index=True)
        modsum_df.to_excel(finalfigmod_path+SHC+'_finalmodscomp_modsummary_Rcrit'+str(Rcrit).replace('.','p')+'.xlsx')  
        
if model_difs:
    for m, mod in enumerate(bestmod):
        SM_chanwidth_kms_230GHz = 2.49
        SM_chanwidth_kms_218GHz = 2.67
        SM_chanwidth_kms_220GHz = 2.65 
        # Errors with cont
        for l,line in enumerate(line_column):
            if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_x']:
                hb_df[line+'_mJy_kms_beam_orig_errcont'] = np.sqrt(hb_df[line+'_mJy_kms_beam_orig_err']**2 + hb_df['cont_'+line+'_beam_orig_err']**2)
                hb_df[line+'_mJy_kms_beam_345_errcont'] = np.sqrt(hb_df[line+'_mJy_kms_beam_345_err']**2 + hb_df['cont_'+line+'_beam_345_err']**2)
                hb_df[line+'_width_kms'] = np.abs(hb_df[line+'_vmin']-hb_df[line+'_vmax'])
                hb_df[line+'_width_channs'] = hb_df[line+'_width_kms']/SM_chanwidth_kms_230GHz
        bestmod = {'mod28b1': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b1','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                                    1.5, redpink , [1, 1, 1]]}
        modelo = bestmod[mod]
        mdust, m_molec, m_molec345 = utiles_nLTEmodel.model_reader(modelo, fort_paths, 1, 1, 1, line_column, models_path, LTE=True, read_only=True)
        lineas = {
                   'v7=1_24_1_23_-1_SM': [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 130], 6],
                   'v7=1_26_1_25_-1_SM': [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 130], 2],
                   'v6=1_24_-1_23_1_SM': [15, r'$v_{6}=1$  24,-1 - 23,1', [-10, 95], 10],
                   'v6=1_26_-1_25_1_SM': [15, r'$v_{6}=1$  26,-1 - 25,1', [-10, 95], 3],
                   'v7=2_24_0_23_0_SM':  [15, r'$v_{7}=2$  24,0 - 23,0', [-5, 70], 11],
                   'v7=2_26_0_25_0_SM':  [15, r'$v_{7}=2$  26,0 - 25,0', [-5, 70], 4],
                   'v5=1_v7=3_26_1_0_25_-1_0_SM':  [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-3, 40], 7],
                   'v6=v7=1_26_2_25_-2_SM': [10, r'$v_{6}=v_{7}=1$  26,2,2 - 25,-2,2', [-3, 30],  8],
                   'v=0_26_25_SM': [15, r'$v=0$  26 - 25', [-10, 150], 1],
                   'v4=1_26_25_SM': [10, r'$v_{4}=1$  26 - 25', [-3, 30], 5],
                   'v6=2_24_0_23_0_SM':  [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 30], 9],
                   }
        
        maxis = 4
        naxis = 3
        figsize = 20
        fontsize = 12
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs1.update(wspace = 0.1, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        axes=[]

        for l,line in enumerate(lineas):
            poly = np.polyfit(hb_df['dist'], hb_df[line+'_mJy_kms_beam_orig'], deg=10)
            fit_profile= np.polyval(poly, m_molec[0])
            axes.append(fig.add_subplot(gs1[l]))
            
            axes[l].plot(new_hb_df['dist'], new_hb_df[line+'_mJy_kms_beam_orig'], marker='s', linestyle='', color='b',zorder=2)
            axes[l].plot(new_hb_df['dist'], new_hb_df[line+'_mJy_kms_beam_345'], marker='s', linestyle='', color='g',zorder=2)
            
            axes[l].plot(hb_df['dist'], hb_df[line+'_mJy_kms_beam_orig'], marker='o', linestyle='', color='k',zorder=2)
            axes[l].plot(hb_df['dist'], hb_df[line+'_mJy_kms_beam_345'], marker='o', linestyle='', color='r',zorder=2)
            axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], linestyle='-', markersize=5, color='b')
            if mod_difs:
                diff = []
                axes[l].plot(m_molec[0], fit_profile, linestyle='-', markersize=5, color='r')
                for r, ring in hb_df.iterrows():
                    
                    for d, dist in enumerate(m_molec[0]):
                        if dist == ring['dist']:
                            difer = np.abs(m_molec[line+'_beam_orig'][d]-ring[line+'_mJy_kms_beam_orig'])
                            diff.append(difer)
                            hb_df.loc[r, line+'_mJy_kms_beam_orig_difmod'] = difer
                            print(f' Found dist {dist}')
                hb_df[line+'_intcont'] = hb_df['cont_'+line+'_beam_orig']*hb_df[line+'_width_kms']
                hb_df[line+'_contdifmean'] = hb_df[line+'_mJy_kms_beam_orig_difmod']/hb_df[line+'_width_kms']
            
            #minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.5])
            axes[l].set_ylim(lineas[line][2])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major')
            
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            axes[l].text(0.9, 0.95, lineas[line][1],
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
            
            if l <= 7:
                axes[l].tick_params(
                           labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)')
        fig_spec = NGC253_path+'SHC/'+SHC+'/EdFlux/Figures_python/'
        fig.savefig(fig_spec+SHC+'_fluxes_comp.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
