import os

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from astrothesispy.utiles import u_conversion
from astrothesispy.utiles import utiles_molecules

# Loading amsmath LaTeX
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
np.seterr(divide = 'ignore') 
def HC3N_energydiag(save_fig_path, HC3N_dir, plot_rotational_levels = True, show_fig = False, fig_format = '.pdf'):
    """
        Plots the energy diagram for HC3N
        plot_rotational_levels = True plots the rotational levels energy diagram
    """
    labelsize = 11
    ticklabsize = 9
    
    # Out dir Figure
    if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
    # Vibrational leveles 
    extra = ['v5=1v7=2', 'v5=v6=1', 'v6=2v7=1', 'v6=1v7=3' ,'v7=5']
    vib_states = np.array(['v=0', 'v1=1', 'v2=1', 'v3=1', 'v4=1', 'v5=1', 'v6=1', 'v7=1', 'v7=2', 'v7=3', 'v7=4', 'v6=2', 'v6=v7=1', 'v6=1v7=2', 'v4=v7=1', 'v5=v7=1', 'v5=2', 'v4=1v7=2', 'v5=1v7=2', 'v5=v6=1', 'v6=2v7=1', 'v6=1v7=3' ,'v7=5'])
    ene_cm = np.array([0, 3327.37141, 2273.99485, 2079.30610, 878.312, 663.368484, 498.733806, 221.838739, 442.899036, 663.2205, 882.85147, 997.913, 720.293173, 941.070371, 1101.0654, 885.37215, 1320.8343, 1320.8343, 1115, 1175, 1220, 1170, 1115])
    ene_K = u_conversion.Energy_cmtoK(ene_cm)
    freq_hz = utiles_molecules.trans_freq(ene_K)
    wave_microns = (freq_hz*u.Hz).to(u.micron, equivalencies=u.spectral())
    
    head_cols_onlyj = ['freq_MHz', 'freq_MHz_err', 'log10_int300K', 'dof', 'Elo_cm1', 'gup', 'tag', 'qn_code', 'jup', 'jlow']
    v0_df = pd.read_csv(HC3N_dir+'sub/'+'HC3Nv0.cat', delim_whitespace=True, header=None)
    v0_df.columns = head_cols_onlyj
    v21_df = pd.read_csv(HC3N_dir+'sub/'+'HC3Nv21.cat', delim_whitespace=True, header=None)
    v21_df.columns = head_cols_onlyj
    v31_df = pd.read_csv(HC3N_dir+'sub/'+'HC3Nv31.cat', delim_whitespace=True, header=None)
    v31_df.columns = head_cols_onlyj
    v41_df = pd.read_csv(HC3N_dir+'sub/'+'HC3Nv41.cat', delim_whitespace=True, header=None)
    v41_df.columns = head_cols_onlyj
    
    
    head_cols_onlyjl = ['freq_MHz', 'freq_MHz_err', 'log10_int300K', 'dof', 'Elo_cm1', 'gup', 'tag', 'qn_code', 'jup', 'lup', 'jlow', 'llow']
    v71_df = pd.read_csv(HC3N_dir+'sub/'+'HC3Nv71.cat', delim_whitespace=True, header=None)
    v71_df.columns = head_cols_onlyjl
    v71_df= v71_df[v71_df['jup']!=v71_df['jlow']]
    v71_df.reset_index(inplace=True)
    v61_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv61.cat', delim_whitespace=True, header=None)
    v61_df.columns = head_cols_onlyjl
    v72_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv72.cat', delim_whitespace=True, header=None)
    v72_df.columns = head_cols_onlyjl
    v62_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv62.cat', delim_whitespace=True, header=None)
    v62_df.columns = head_cols_onlyjl
    v41v71_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv41v71.cat', delim_whitespace=True, header=None)
    v41v71_df.columns = head_cols_onlyjl
    
    head_cols_onlyjlk = ['freq_MHz', 'freq_MHz_err', 'log10_int300K', 'dof', 'Elo_cm1', 'gup', 'tag', 'qn_code', 'jup', 'lup', 'kup', 'jlow', 'llow', 'klow']
    v74v51v71_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv74v51v71.cat', delim_whitespace=True, header=None)
    v74v51v71_df.columns = head_cols_onlyjlk
    v61v71_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv61v71.cat', delim_whitespace=True, header=None)
    v61v71_df.columns = head_cols_onlyjlk
    v51v73_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv51v73.cat', delim_whitespace=True, header=None)
    v51v73_df.columns = head_cols_onlyjlk
    v41v72v52_df = pd.read_csv(HC3N_dir+'/sub/'+'HC3Nv41v72v52.cat', delim_whitespace=True, header=None)
    v41v72v52_df.columns = head_cols_onlyjlk
    
    # Plotting levels
    posv5 = [2,3]
    posv5v6 = [2.5,3.5]
    posv0 = [3.5,4.5]
    posv0 = [2, 9]
    posv6 = [3.5,4.5]
    posv6v7 = [5,6]
    posv7 = [6.5,7.5]
    posv4 = [8,9]
    posv3 = [8,9]
    posv2 = [8,9]
    posv1 = [8, 9]
    
    df_dict = {'v=0':{'coup': '', 'label': r'$v=0$','ene_K': ene_K[0], 'ene_cm': ene_cm[0], 'freq_hz': freq_hz[0], 'wave_microns':0, 'df': v0_df, 'pos':posv0},
               'v1=1':{'coup': '','label': r'$v_1=1$','ene_K': ene_K[1], 'ene_cm': ene_cm[1], 'freq_hz': freq_hz[1], 'wave_microns':wave_microns[1].value, 'df': None, 'pos':posv1},
               'v2=1':{'coup': '','label': r'$v_2=1$','ene_K': ene_K[2], 'ene_cm': ene_cm[2], 'freq_hz': freq_hz[2], 'wave_microns':wave_microns[2].value, 'df': v21_df, 'pos':posv2},
               'v3=1':{'coup': '','label': r'$v_3=1$','ene_K': ene_K[3], 'ene_cm': ene_cm[3], 'freq_hz': freq_hz[3], 'wave_microns':wave_microns[3].value, 'df': v31_df, 'pos':posv3},
               'v4=1':{'coup': '**','label': r'$v_4=1$','ene_K': ene_K[4], 'ene_cm': ene_cm[4], 'freq_hz': freq_hz[4], 'wave_microns':wave_microns[4].value, 'df': v41_df, 'pos':posv4},
               'v5=1':{'coup': '*','label': r'$v_5=1$','ene_K': ene_K[5], 'ene_cm': ene_cm[5], 'freq_hz': freq_hz[5], 'wave_microns':wave_microns[5].value, 'df': v51v73_df, 'pos':posv5},
               'v6=1':{'coup': '','label': r'$v_6=1$','ene_K': ene_K[6], 'ene_cm': ene_cm[6], 'freq_hz': freq_hz[6], 'wave_microns':wave_microns[6].value, 'df': v61_df, 'pos':posv6},
               'v7=1':{'coup': '','label': r'$v_7=1$','ene_K': ene_K[7], 'ene_cm': ene_cm[7], 'freq_hz': freq_hz[7], 'wave_microns':wave_microns[7].value, 'df': v71_df, 'pos':posv7},
               'v7=2':{'coup': '','label': r'$v_7=2$','ene_K': ene_K[8], 'ene_cm': ene_cm[8], 'freq_hz': freq_hz[8], 'wave_microns':wave_microns[8].value, 'df': v72_df, 'pos':posv7},
               'v7=3':{'coup': '*','label': r'$v_7=3$','ene_K': ene_K[9], 'ene_cm': ene_cm[9], 'freq_hz': freq_hz[9], 'wave_microns':wave_microns[9].value, 'df': v51v73_df, 'pos':posv7},
               'v7=4':{'coup': '**','label': r'$v_7=4$','ene_K': ene_K[10], 'ene_cm': ene_cm[10], 'freq_hz': freq_hz[10], 'wave_microns':wave_microns[10].value, 'df': v74v51v71_df, 'pos':posv7},
               'v6=2':{'coup': '**','label': r'$v_6=2$','ene_K': ene_K[11], 'ene_cm': ene_cm[11], 'freq_hz': freq_hz[11], 'wave_microns':wave_microns[11].value, 'df': v62_df, 'pos':posv6},
               'v6=v7=1':{'coup': '','label': r'$v_6=v_7=1$','ene_K': ene_K[12], 'ene_cm': ene_cm[12], 'freq_hz': freq_hz[12], 'wave_microns':wave_microns[12].value, 'df': v61v71_df, 'pos':posv6v7},
               'v6=1v7=2':{'coup': '','label': r'$v_6=1,v_7=2$','ene_K': ene_K[13], 'ene_cm': ene_cm[13], 'freq_hz': freq_hz[13], 'wave_microns':wave_microns[13].value, 'df': None, 'pos':posv6v7},
               'v4=v7=1':{'coup': '','label': r'$v_4=v_7=1$','ene_K': ene_K[14], 'ene_cm': ene_cm[14], 'freq_hz': freq_hz[14], 'wave_microns':wave_microns[14].value, 'df': v41v71_df, 'pos':posv4},
               'v5=v7=1':{'coup': '**','label': r'$v_5=v_7=1$','ene_K': ene_K[15], 'ene_cm': ene_cm[15], 'freq_hz': freq_hz[15], 'wave_microns':wave_microns[15].value, 'df': v74v51v71_df, 'pos':posv5},
               'v5=2':{'coup': '***','label': r'$v_5=2$','ene_K': ene_K[16], 'ene_cm': ene_cm[16], 'freq_hz': freq_hz[16], 'wave_microns':wave_microns[16].value, 'df': v41v72v52_df, 'pos':posv5},
               'v4=1v72':{'coup': '***','label': r'$v_4=1,v_7=2$','ene_K': ene_K[17], 'ene_cm': ene_cm[17], 'freq_hz': freq_hz[17], 'wave_microns':wave_microns[17].value, 'df': v41v72v52_df, 'pos':posv4},
               'v5=1v7=2':{'coup': '','label': r'$v_5=1,v_7=2$','ene_K': ene_K[18], 'ene_cm': ene_cm[18], 'freq_hz': freq_hz[18], 'wave_microns':wave_microns[18].value, 'df': None, 'pos':posv5},
               'v5=v6=1':{'coup': '','label': r'$v_5=v_6=1$','ene_K': ene_K[19], 'ene_cm': ene_cm[19], 'freq_hz': freq_hz[19], 'wave_microns':wave_microns[19].value, 'df': None, 'pos':posv5},
               'v6=2v7=1':{'coup': '','label': r'$v_6=2,v_7=1$','ene_K': ene_K[20], 'ene_cm': ene_cm[20], 'freq_hz': freq_hz[20], 'wave_microns':wave_microns[20].value, 'df': None, 'pos':posv6},
               'v6=1v7=3':{'coup': '','label':r'$v_6=1,v_7=3$' ,'ene_K': ene_K[21], 'ene_cm': ene_cm[21], 'freq_hz': freq_hz[21], 'wave_microns':wave_microns[21].value, 'df': None, 'pos':posv6v7},
               'v7=5':{'coup': '','label': r'$v_7=5$','ene_K': ene_K[22], 'ene_cm': ene_cm[22], 'freq_hz': freq_hz[22], 'wave_microns':wave_microns[22].value, 'df': None, 'pos':posv7},
               }
    
    if not plot_rotational_levels:
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', which='major', labelsize=ticklabsize)
        # Hide the right bottom and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # Only show ticks on the left spines
        ax.yaxis.set_ticks_position('left')
        # Hide Xaxis ticks
        ax.get_xaxis().set_visible(False)
        # Hide Yaxis ticks    
        label_sep = 50
        maxy = 2100
        ax.set_ylim([-5, maxy])
        # Vib states
        for key in df_dict:
            if key in extra: 
                lins = '--'
            else:
                lins = '-'
            if df_dict[key]['ene_cm'] < 1500:
                ax.plot([df_dict[key]['pos'][0], df_dict[key]['pos'][1]], [df_dict[key]['ene_K'],df_dict[key]['ene_K']],color='k', linewidth=1, zorder=1, linestyle=lins)
                ax.text(np.mean([df_dict[key]['pos'][0], df_dict[key]['pos'][1]]), df_dict[key]['ene_K']-0.025*maxy, df_dict[key]['label'], ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
                ax.text(np.mean([df_dict[key]['pos'][0], df_dict[key]['pos'][1]]), df_dict[key]['ene_K']+0.025*maxy, df_dict[key]['coup'], ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold', zorder=0)
        ax.set_ylabel('Energy (K)', fontsize=labelsize)
        ax.annotate("", xy=(np.mean(df_dict['v7=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v7=1']['pos'])-0.4, df_dict['v7=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v7=1']['pos'])-0.4+0.07, 125, r''+str(np.round(df_dict['v7=1']['wave_microns'],1))+'\,$\mu$m', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        ax.annotate("", xy=(np.mean(df_dict['v6=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v6=1']['pos'])-0.4, df_dict['v6=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v6=1']['pos']), 415, r''+str(np.round(df_dict['v6=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        ax.annotate("", xy=(np.mean(df_dict['v6=v7=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v6=v7=1']['pos'])-0.4, df_dict['v6=v7=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v6=v7=1']['pos']), 705, r''+str(np.round(df_dict['v6=v7=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        ax.annotate("", xy=(np.mean(df_dict['v7=2']['pos'])+0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v7=2']['pos'])+0.4, df_dict['v7=2']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v7=2']['pos'])+0.4+0.4, 405, r''+str(np.round(df_dict['v7=2']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        ax.annotate("", xy=(np.mean(df_dict['v4=1']['pos'])-0.3, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v4=1']['pos'])-0.3, df_dict['v4=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v4=1']['pos'])+0.1, 805, r''+str(np.round(df_dict['v4=1']['wave_microns'],1))+'\,$\mathrm{\mu}$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        ax.annotate("", xy=(np.mean(df_dict['v5=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v5=1']['pos'])-0.4, df_dict['v5=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v5=1']['pos']), 655, r''+str(np.round(df_dict['v5=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        fig.savefig(f'{save_fig_path}/HC3N_Ediag_K_{fig_format}', bbox_inches='tight', transparent=False, dpi=400)
        plt.close()
    
    
    if plot_rotational_levels:
        fig, (ax, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
        plt.subplots_adjust(wspace=0.05, hspace=0)
        ax.tick_params(axis='both', which='major', labelsize=ticklabsize)
        # Hide the right bottom and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Only show ticks on the left spines
        ax.yaxis.set_ticks_position('left')
        # Hide Xaxis ticks
        ax.get_xaxis().set_visible(False)
        # Hide Yaxis ticks
        #ax.get_yaxis().set_visible(False)

        label_sep = 50
        maxy = 2100
        ax.set_ylim([-5, maxy])
        ax2.set_ylim([300, 700])
        ax2.yaxis.tick_right()
        ax2.spines['right'].set_visible(True)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.set_xticks([])
        
        rot_v7poslp = [2,2.5]
        rot_v7poslm = [3,3.5]
        df_dict['v7=1']['df']['Elo_K'] = u_conversion.Energy_cmtoK(df_dict['v7=1']['df']['Elo_cm1'])
        df_dict['v7=1']['df']['wave_mm'] = (1*u.MHz).to(u.mm, equivalencies=u.spectral()).value/df_dict['v7=1']['df']['freq_MHz']
        wave_microns = (freq_hz*u.Hz).to(u.micron, equivalencies=u.spectral())
        for i, row in df_dict['v7=1']['df'].iterrows():
            if row['llow'] == 1:
                ax2.plot(rot_v7poslp, [row['Elo_K'],row['Elo_K']], color='k', linewidth=0.5, zorder=1)
            else: 
                ax2.plot(rot_v7poslm, [row['Elo_K'],row['Elo_K']], color='k', linewidth=0.5, zorder=1)
            
            if row['jlow'] in ['16', '24', '26', '39']:
                freq_E = utiles_molecules.trans_energy(row['freq_MHz']/1000)
                if (row['llow'] == 1.0):
                    strplt = row["jlow"]
                    ax2.text(1.85, row['Elo_K'], r'$'+strplt+'$', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
                    
        df39_p1_38_m1 = df_dict['v7=1']['df'][(df_dict['v7=1']['df']['jup']=='39') & (df_dict['v7=1']['df']['lup']==1)]
        freq_E = utiles_molecules.trans_energy(df39_p1_38_m1['freq_MHz']/1000)
        ax2.annotate("", xy=(3, df39_p1_38_m1['Elo_K']),
                            xytext=(2.5, df39_p1_38_m1['Elo_K']+freq_E),
                            arrowprops=dict(arrowstyle="->, head_length=0.2, head_width=0.1",
                                        fc='g', ec='g', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2)
        ax2.text(2.75, df39_p1_38_m1['Elo_K']+freq_E+4, str(np.round(df39_p1_38_m1['wave_mm'].tolist()[0],2))+'mm',ha='center', va='center', backgroundcolor='none', fontsize=6 , color='g', fontweight='bold')
        
        df24_p1_23_m1 = df_dict['v7=1']['df'][(df_dict['v7=1']['df']['jup']=='24') & (df_dict['v7=1']['df']['lup']==1)]
        freq_E = utiles_molecules.trans_energy(df24_p1_23_m1['freq_MHz']/1000)
        ax2.annotate("", xy=(3, df24_p1_23_m1['Elo_K']),
                            xytext=(2.5, df24_p1_23_m1['Elo_K']+freq_E),
                            arrowprops=dict(arrowstyle="->, head_length=0.2, head_width=0.1",
                                        fc='g', ec='g', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2)
        ax2.text(2.75, df24_p1_23_m1['Elo_K']+freq_E+3, str(np.round(df24_p1_23_m1['wave_mm'].tolist()[0],2))+'mm',ha='center', va='center', backgroundcolor='none', fontsize=6 , color='g', fontweight='bold')
        
        
        df26_p1_25_m1 = df_dict['v7=1']['df'][(df_dict['v7=1']['df']['jup']=='26') & (df_dict['v7=1']['df']['lup']==1)]
        freq_E = utiles_molecules.trans_energy(df26_p1_25_m1['freq_MHz']/1000)
        ax2.annotate("", xy=(3, df26_p1_25_m1['Elo_K']),
                            xytext=(2.5, df26_p1_25_m1['Elo_K']+freq_E),
                            arrowprops=dict(arrowstyle="->, head_length=0.2, head_width=0.1",
                                        fc='g', ec='g', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2)
        ax2.text(2.75, df26_p1_25_m1['Elo_K']+freq_E+3, str(np.round(df26_p1_25_m1['wave_mm'].tolist()[0],2))+'mm',ha='center', va='center', backgroundcolor='none', fontsize=6 , color='g', fontweight='bold')
        
        
        df16_p1_15_m1 = df_dict['v7=1']['df'][(df_dict['v7=1']['df']['jup']=='16') & (df_dict['v7=1']['df']['lup']==1)]
        freq_E = utiles_molecules.trans_energy(df16_p1_15_m1['freq_MHz']/1000)
        ax2.annotate("", xy=(3, df16_p1_15_m1['Elo_K']),
                            xytext=(2.5, df16_p1_15_m1['Elo_K']+freq_E),
                            arrowprops=dict(arrowstyle="->, head_length=0.2, head_width=0.1",
                                        fc='g', ec='g', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2)
        ax2.text(2.75, df16_p1_15_m1['Elo_K']+freq_E+3, str(np.round(df16_p1_15_m1['wave_mm'].tolist()[0],2))+'mm',ha='center', va='center', backgroundcolor='none', fontsize=6 , color='g', fontweight='bold')
        
        ax2.text(np.mean(rot_v7poslp), 310, r'$l=+1$',ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
        ax2.text(np.mean(rot_v7poslm), 310, r'$l=-1$',ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
        ax2.text(1.85, 330, r'$J$',ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
        
        # Vib states
        for key in df_dict:
            if key in extra: 
                lins = '--'
            else:
                lins = '-'
            if df_dict[key]['ene_cm'] < 1500:
                ax.plot([df_dict[key]['pos'][0], df_dict[key]['pos'][1]], [df_dict[key]['ene_K'],df_dict[key]['ene_K']],color='k', linewidth=1, zorder=1, linestyle=lins)
                ax.text(np.mean([df_dict[key]['pos'][0], df_dict[key]['pos'][1]]), df_dict[key]['ene_K']-0.025*maxy, df_dict[key]['label'], ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold')
                ax.text(np.mean([df_dict[key]['pos'][0], df_dict[key]['pos'][1]]), df_dict[key]['ene_K']+0.025*maxy, df_dict[key]['coup'], ha='center', va='center', backgroundcolor='none', fontsize=8 , color='k', fontweight='bold', zorder=0)
                
        ax.set_ylabel('Energy (K)', fontsize=labelsize)
        
        ax.annotate("", xy=(np.mean(df_dict['v7=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v7=1']['pos'])-0.4, df_dict['v7=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v7=1']['pos'])-1.5+0.07, 125, r''+str(np.round(df_dict['v7=1']['wave_microns'],1))+'\,$\mu$m', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        ax.annotate("", xy=(np.mean(df_dict['v6=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v6=1']['pos'])-0.4, df_dict['v6=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v6=1']['pos'])+0.1, 415, r''+str(np.round(df_dict['v6=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        ax.annotate("", xy=(np.mean(df_dict['v6=v7=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v6=v7=1']['pos'])-0.4, df_dict['v6=v7=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v6=v7=1']['pos'])+0.1, 705, r''+str(np.round(df_dict['v6=v7=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        ax.annotate("", xy=(np.mean(df_dict['v7=2']['pos'])+0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v7=2']['pos'])+0.4, df_dict['v7=2']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v7=2']['pos'])-1.0+0.45+0.4, 440, r''+str(np.round(df_dict['v7=2']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        ax.annotate("", xy=(np.mean(df_dict['v4=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v4=1']['pos'])-0.4, df_dict['v4=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v4=1']['pos'])+0.1, 805, r''+str(np.round(df_dict['v4=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        ax.annotate("", xy=(np.mean(df_dict['v5=1']['pos'])-0.4, df_dict['v=0']['ene_K']),
                            xytext=(np.mean(df_dict['v5=1']['pos'])-0.4, df_dict['v5=1']['ene_K']),
                            arrowprops=dict(arrowstyle="<->, head_length=0.2, head_width=0.1",
                                        fc='r', ec='r', shrinkA = 0, shrinkB = 0, linewidth=0.7)
                                        , zorder=2) 
        ax.text(np.mean(df_dict['v5=1']['pos'])+0.1, 655, r''+str(np.round(df_dict['v5=1']['wave_microns'],1))+'\,$\mu$m', ha='center', va='center', backgroundcolor='none', fontsize=8 , color='r', fontweight='bold')
        
        fig.savefig(f'{save_fig_path}/HC3N_Ediag_K_wrot_lvls{fig_format}', bbox_inches='tight', transparent=False, dpi=400)
        if show_fig:
            plt.show()
        plt.close()


