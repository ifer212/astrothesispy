
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from astrothesispy.radiative_transfer import NGC253HR_nLTE_modelresults
from astrothesispy.utiles import utiles_plot as plot_utiles

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
mpl.rc('xtick', color='k', direction='in', labelsize=6)
mpl.rc('ytick', color='k', direction='in', labelsize=6)

def plot_LIR_comp_ALL_big(fig_path, results_path, source, D_Mpc=3.5, Lmod_err=0.5, only_HC = True, fig_name = ''):
    """
        Comparison btw HC, SHC and AGNs
    """
    modsum_df, hc_df, rolffs_df, bgn_df, pfalzner_df, lada_df, portout_df, portin_df = NGC253HR_nLTE_modelresults.models_calculations(results_path, source, D_Mpc)
    ms = 12
    figsize = 8
    naxis = 2
    maxis = 2
    labelsize = 30
    ticksize = 25
    fontsize = 18
    anotsize = 14
    fig = plt.figure(figsize=(figsize*2.15, figsize*1.85))
    gs = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs.update(wspace = 0.20, hspace=0.14, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    
    panel_2_MH2_R2 = False
    
    axis = []
    axis.append(fig.add_subplot(gs[0]))
    axis.append(fig.add_subplot(gs[1]))
    axis.append(fig.add_subplot(gs[2]))
    axis.append(fig.add_subplot(gs[3]))
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', markeredgecolor='k', label='HCs', markerfacecolor=plot_utiles.azure, markersize=13, linestyle=''),
                       Line2D([0], [0], marker='o', markeredgecolor='k', label='Proto-SSC13a', markerfacecolor=plot_utiles.redpink, markersize=13, linestyle=''),
                       Line2D([0], [0], marker='o', markeredgecolor='k', label='BGNs', markerfacecolor=plot_utiles.green, markersize=13, linestyle='')]

    # Create the figure
    axis[0].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=fontsize)
    
    # SigmaIR vs R^2
    modxvar = np.log10(modsum_df['R_pc']**2)
    modyvar = np.log10(modsum_df['SigmaIR_Lsun_pc2'])
    modsum_df['SigmaIR_err'] = Lmod_err*modsum_df['Ltot_Lsun']/(np.pi*modsum_df['R_pc']**2)
    modsum_df['log_SigmaIR_err'] = (10**np.log10(modsum_df['SigmaIR_err']))*(1/np.log(10))/(10**np.log10(modsum_df['SigmaIR_Lsun_pc2']))
    modyvar_err = modsum_df['log_SigmaIR_err'] 
    axis[0].plot(modxvar, modyvar, marker='o', color=plot_utiles.redpink, linestyle ='', markeredgecolor='k', ms = ms)
    modsum_df['log_Ltot_err'] = 0.5#np.log10(modsum_df['Ltot_name_Lsun'])-np.log10(Lmod_err*modsum_df['Ltot_name_Lsun']) #(10**np.log10(Lmod_err*modsum_df['Ltot_name_Lsun']))*(1/np.log(10))/(10**np.log10(modsum_df['Ltot_name_Lsun']))
    axis[0].errorbar(modxvar, modyvar, 
                     yerr=modyvar_err,
                     marker='o', markersize=ms,
                     markerfacecolor=plot_utiles.redpink,
                     markeredgecolor='k', markeredgewidth=0.8,
                     ecolor='k',
                     color = plot_utiles.redpink,
                     elinewidth= 0.8,
                     barsabove= True,
                     zorder=1)
    # HCs
    xvarlab = 'r_pc'
    yvarlab = 'SigmaIR_Lsun_pc2'
    for column in rolffs_df.columns.to_list():
        try:
            rolffs_df[column] = [i.value for i in rolffs_df[column]]
        except:
            ski_p = True
    axis[0].plot(np.log10(hc_df[xvarlab]**2),  np.log10(hc_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    axis[0].plot(np.log10(rolffs_df[xvarlab]**2), np.log10(rolffs_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    for i, row in rolffs_df.iterrows():
        name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
        if row['Source'] in hc_df['Source'].tolist():
            indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
            x_vals = [np.log10(row[xvarlab]**2), np.log10(hc_df.loc[indx, xvarlab]**2)]
            y_vals = [np.log10(row[yvarlab]), np.log10(hc_df.loc[indx, yvarlab])]
            axis[0].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=plot_utiles.azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
            ind_text = 0
            if row['Source'] == 'SgrB2(M)':
                ind_text = 1
                halign = 'center'
                valign = 'bottom'
                xm = 0
                ym = 0.05
            elif row['Source'] == 'SgrB2(N)':
                ind_text = 0
                halign = 'center'
                valign = 'top'
                xm = 0.0
                ym = -0.055
            axis[0].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                     ha=halign, va=valign, color = 'k', fontsize=anotsize)
    
    # BGNs 
    xvarlab= 'R_out_pc'
    xvarlab2= 'R_out2_pc'
    yvarlab = 'SigmaIR_Lsun_pc2'
    yvarlab2 = 'SigmaIR2_Lsun_pc2'
    for i, row in bgn_df.iterrows():
        x_vals = [np.log10(row[xvarlab]**2), np.log10(row[xvarlab2]**2)]
        y_vals = [np.log10(row[yvarlab]), np.log10(row[yvarlab2])]
        axis[0].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = plot_utiles.green, markeredgecolor='k', ms = ms)
        ind_text = 0
        if row['Source'] == 'Arp220E':
            halign = 'right'
            valign = 'top'
            xm = -0.2
            ym = 0
        elif row['Source'] == 'NGC4418':
            halign = 'right'
            valign = 'top'
            xm = -0.2
            ym = 0
        elif row['Source'] == 'Arp220W':
            halign = 'left'
            valign = 'top'
            xm = +0.03#-0.13
            ym = -0.13
        elif row['Source'] == 'Zw049.057':
            halign = 'right'
            valign = 'top'
            xm = 0.4
            ym = -0.2
        elif row['Source'] == 'IC860':
            halign = 'right'
            valign = 'bottom'
            xm = -0.15
            ym = 0.13
        elif row['Source'] == 'Mrk231':
            ind_text = 1
            halign = 'right'
            valign = 'bottom'
            xm = -0.12
            ym = 0.05
        axis[0].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                ha='right', va='center', color = 'k', fontsize=anotsize)
    axis[0].set_xlabel(r'$\log{R^2}$ (pc)', fontsize=labelsize)
    axis[0].set_ylabel(r'$\log{\Sigma_{\text{IR}}}$ (L$_{\odot}$/pc$^{2}$)', fontsize=labelsize)
    axis[0].set_xlim([-2.1, 4.15])
    axis[0].set_ylim([4.7, 8.6])
     
    
    # MH2 vs R^2
    if panel_2_MH2_R2:
        modxvar = np.log10(modsum_df['R_pc']**2)
        modyvar = np.log10(modsum_df['Mgas_Msun_corr'])
        xvarlab = 'r_pc'
        yvarlab = 'Mass_Msun'
        bgnxvarlab= 'R_out_pc'
        bgnxvarlab2= 'R_out2_pc'
        bgnyvarlab = 'MH2_Msun'
    else:
        modxvar = np.log10(modsum_df['SigmaH2_Msun_pc2'])
        modyvar = np.log10(modsum_df['SigmaIR_Lsun_pc2'])
        modyvar_err = modsum_df['log_SigmaIR_err'] 
        xvarlab = 'SigmaH2_Msun_pc2'
        yvarlab = 'SigmaIR_Lsun_pc2'
        bgnyvarlab = 'SigmaIR_Lsun_pc2'
        bgnyvarlab2 = 'SigmaIR2_Lsun_pc2'
        bgnxvarlab  = 'SigmaH2_Msun1_pc2'
        bgnxvarlab2 = 'SigmaH2_Msun2_pc2'
        
    modyvar_err = 0.5
    axis[1].plot(modxvar, modyvar, marker='o', color=plot_utiles.redpink, linestyle ='', markeredgecolor='k', ms = ms)
    if not panel_2_MH2_R2:
        axis[1].errorbar(modxvar, modyvar, 
                     yerr=modyvar_err,
                     marker='o', markersize=ms,
                     markerfacecolor=plot_utiles.redpink,
                     markeredgecolor='k', markeredgewidth=0.8,
                     ecolor='k',
                     color = plot_utiles.redpink,
                     elinewidth= 0.8,
                     barsabove= True,
                     zorder=1)
    # HCs
    if panel_2_MH2_R2:
        axis[1].plot(np.log10(hc_df[xvarlab]**2),  np.log10(hc_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[1].plot(np.log10(rolffs_df[xvarlab]**2), np.log10(rolffs_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    else:
        x_lin = np.linspace(10**3.5, 10**6.4,200)
        y_lin = 100*x_lin
        axis[1].plot(np.log10(x_lin), np.log10(y_lin), color='k', marker='', linestyle='--', zorder=0)
        axis[1].plot(np.log10(hc_df[xvarlab]),  np.log10(hc_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
        axis[1].plot(np.log10(rolffs_df[xvarlab]), np.log10(rolffs_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    for i, row in rolffs_df.iterrows():
        name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
        if row['Source'] in hc_df['Source'].tolist():
            indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
            if panel_2_MH2_R2:
                x_vals = [np.log10(row[xvarlab]**2), np.log10(hc_df.loc[indx, xvarlab]**2)]
            else:
                x_vals = [np.log10(row[xvarlab]), np.log10(hc_df.loc[indx, xvarlab])]
            y_vals = [np.log10(row[yvarlab]), np.log10(hc_df.loc[indx, yvarlab])]
            axis[1].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=plot_utiles.azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
            ind_text = 0
            if row['Source'] == 'SgrB2(M)':
                ind_text = 1
                halign = 'center'
                valign = 'bottom'
                xm = 0
                ym = 0.05
            elif row['Source'] == 'SgrB2(N)':
                ind_text = 0
                halign = 'center'
                valign = 'top'
                xm = 0.0
                ym = -0.055
            axis[1].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                    ha=halign, va=valign, color = 'k', fontsize=anotsize)

    
    # BGNs 
    for i, row in bgn_df.iterrows():
        if panel_2_MH2_R2:
            x_vals = [np.log10(row[bgnxvarlab]**2), np.log10(row[bgnxvarlab2]**2)]
        else:
            x_vals = [np.log10(row[bgnxvarlab]), np.log10(row[bgnxvarlab2])]
        if panel_2_MH2_R2:
            y_vals = [np.log10(row[bgnyvarlab]), np.log10(row[bgnyvarlab])]
        else:
            y_vals = [np.log10(row[bgnyvarlab]), np.log10(row[bgnyvarlab2])]
        axis[1].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = plot_utiles.green, markeredgecolor='k', ms = ms)
        ind_text = 0
        if row['Source'] == 'Arp220E':
            ind_text = 1
            halign = 'right'
            valign = 'top'
            xm = -0.1
            ym = 0
        elif row['Source'] == 'NGC4418':
            halign = 'left'
            valign = 'bottom'
            xm = 0.1
            ym = 0
        elif row['Source'] == 'Arp220W':
            halign = 'left'
            valign = 'top'
            xm = 0.1
            ym = 0.06
        elif row['Source'] == 'Zw049.057':
            halign = 'center'
            valign = 'top'
            xm = -0.06
            ym = -0.11
        elif row['Source'] == 'IC860':
            halign = 'right'
            valign = 'top'
            xm = -0.06
            ym = -0.08
        elif row['Source'] == 'Mrk231':
            ind_text = 1
            halign = 'right'
            valign = 'bottom'
            xm = -0.08
            ym = 0.03
        axis[1].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                 ha=halign, va=valign, color = 'k', fontsize=anotsize) #arrowprops={'arrowstyle': '-', 'color': 'k'},
    if panel_2_MH2_R2:
        axis[1].set_xlabel(r'$\log{R^2}$ (pc)', fontsize=labelsize)
        axis[1].set_ylabel(r'$\log{M_{\text{H2}}}$ (M$_{\odot}$)', fontsize=labelsize)
    else:
        axis[1].set_xlabel(r'$\log{\Sigma_{\text{H2}}}$ (M$_{\odot}$/pc$^{2}$)', fontsize=labelsize)
        axis[1].set_ylabel(r'$\log{\Sigma_{\text{IR}}}$ (L$_{\odot}$/pc$^{2}$)', fontsize=labelsize)
    
    axis[1].set_xlim([3.4, 6.5])
    axis[1].set_ylim([4.7, 8.6])
    
    
    # LIR/MH2 vs MH2
    modyvar = np.log10(modsum_df['LIR/Mgas_corr'])
    modxvar = np.log10(modsum_df['Mgas_Msun_corr'])
    modsum_df['LtotMgas_err'] = np.sqrt((Lmod_err*modsum_df['Ltot_Lsun']/modsum_df['Mgas_Msun_corr'])**2+(modsum_df['Ltot_Lsun']*(0*Lmod_err/2)*modsum_df['Mgas_Msun_corr']/(modsum_df['Mgas_Msun_corr']**2))**2)
    modsum_df['log_LtotMgas_err'] = (10**np.log10(modsum_df['LtotMgas_err']))*(1/np.log(10))/(10**np.log10(modsum_df['LIR/Mgas_corr']))
    modyvar_err = modsum_df['log_LtotMgas_err']
    axis[2].plot(modxvar, modyvar, marker='o', color=plot_utiles.redpink, linestyle ='', markeredgecolor='k', ms = ms)
    axis[2].errorbar(modxvar, modyvar, 
                     yerr=modyvar_err,
                     marker='o', markersize=ms,
                     markerfacecolor=plot_utiles.redpink,
                     markeredgecolor='k', markeredgewidth=0.8,
                     ecolor='k',
                     color = plot_utiles.redpink,
                     elinewidth= 0.8,
                     barsabove= True,
                     zorder=1)
    # HCs
    yvarlab = 'LIR/Mgas'
    xvarlab = 'Mass_Msun'
    axis[2].hlines(2, xmin=np.log10(np.nanmin(rolffs_df[xvarlab])), xmax=np.log10(np.nanmax(bgn_df['MH2_Msun'])), color='k', linestyles='--',zorder=0)
    axis[2].plot(np.log10(hc_df[xvarlab]),  np.log10(hc_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    axis[2].plot(np.log10(rolffs_df[xvarlab]), np.log10(rolffs_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    for i, row in rolffs_df.iterrows():
        name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
        if row['Source'] in hc_df['Source'].tolist():
            indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
            x_vals = [np.log10(row[xvarlab]), np.log10(hc_df.loc[indx, xvarlab])]
            y_vals = [np.log10(row[yvarlab]), np.log10(hc_df.loc[indx, yvarlab])]
            axis[2].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=plot_utiles.azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
            ind_text = 0
            if row['Source'] == 'SgrB2(M)':
                ind_text = 1
                halign = 'center'
                valign = 'bottom'
                xm = 0
                ym = 0.04
            elif row['Source'] == 'SgrB2(N)':
                ind_text = 0
                halign = 'center'
                valign = 'bottom'
                xm = 0.0
                ym = 0.04
            axis[2].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                   ha=halign, va=valign, color = 'k', fontsize=anotsize)

    # BGNs 
    yvarlab = 'LIR/Mgas'
    yvarlab2 = 'LIR2/Mgas'
    xvarlab = 'MH2_Msun'
    for i, row in bgn_df.iterrows():
        x_vals = [np.log10(row[xvarlab]), np.log10(row[xvarlab])]
        y_vals = [np.log10(row[yvarlab]), np.log10(row[yvarlab2])]
        axis[2].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = plot_utiles.green, markeredgecolor='k', ms = ms)
        ind_text = 0
        if row['Source'] == 'Arp220E':
            ind_text = 1
            halign = 'center'
            valign = 'bottom'
            xm = 0.08
            ym = 0.043
        elif row['Source'] == 'NGC4418':
            ind_text = 1
            halign = 'left'
            valign = 'bottom'
            xm = 0.06
            ym = 0.01
        elif row['Source'] == 'Arp220W':
            halign = 'center'
            valign = 'top'
            xm = -0.07
            ym = -0.05
        elif row['Source'] == 'Zw049.057':
            halign = 'center'
            valign = 'top'
            xm = 0.0
            ym = -0.06
        elif row['Source'] == 'IC860':
            halign = 'center'
            valign = 'top'
            xm = 0
            ym = -0.06
        elif row['Source'] == 'Mrk231':
            ind_text = 1
            halign = 'center'
            valign = 'bottom'
            xm = -0.08
            ym = 0.04
        axis[2].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                ha=halign, va=valign, color = 'k', fontsize=anotsize)

    axis[2].set_ylabel(r'$\log{L_{\text{IR}}/M_{\text{H2}}}$ (L$_{\odot}$/M$_{\odot}$)', fontsize=labelsize)
    axis[2].set_xlabel(r'$\log{M_{\text{H2}}}$ (M$_{\odot}$)', fontsize=labelsize)
    axis[2].set_ylim([0.75, 2.7])
    axis[2].set_xlim([2.3, 10.25])
    
    # LIR/MH2 vs Sigma
    modyvar = np.log10(modsum_df['LIR_name/Mgas_corr'])
    modxvar = np.log10(modsum_df['SigmaIR_Lsun_pc2'])
    modsum_df['SigmaIR_err'] = Lmod_err*modsum_df['Ltot_Lsun']/(np.pi*modsum_df['R_pc']**2)
    modsum_df['log_SigmaIR_err'] = (10**np.log10(modsum_df['SigmaIR_err']))*(1/np.log(10))/(10**np.log10(modsum_df['SigmaIR_Lsun_pc2']))
    modyvar_err = modsum_df['log_LtotMgas_err']
    modxvar_err = modsum_df['log_SigmaIR_err'] 
    axis[3].plot(modxvar, modyvar, marker='o', color=plot_utiles.redpink, linestyle ='', markeredgecolor='k', ms = ms)
    axis[3].errorbar(modxvar, modyvar, 
                     xerr = modxvar_err,
                     yerr=modyvar_err,
                     marker='o', markersize=ms,
                     markerfacecolor=plot_utiles.redpink,
                     markeredgecolor='k', markeredgewidth=0.8,
                     ecolor='k',
                     color = plot_utiles.redpink,
                     elinewidth= 0.8,
                     barsabove= True,
                     zorder=1)
    # HCs
    yvarlab = 'LIR/Mgas'
    xvarlab = 'SigmaIR_Lsun_pc2'
    axis[3].hlines(2, xmin=np.log10(np.nanmin(rolffs_df[xvarlab])), xmax=np.log10(np.nanmax(bgn_df['SigmaIR2_Lsun_pc2'])), color='k', linestyles='--')
    axis[3].plot(np.log10(hc_df[xvarlab]),  np.log10(hc_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    axis[3].plot(np.log10(rolffs_df[xvarlab]), np.log10(rolffs_df[yvarlab]),  marker='o', color=plot_utiles.azure, linestyle ='', markeredgecolor='k', ms = ms)
    for i, row in rolffs_df.iterrows():
        name = row['Source'].split('-')[0].split('+')[0].split('.')[0]
        if row['Source'] in hc_df['Source'].tolist():
            indx = hc_df.index[hc_df['Source']==row['Source']].tolist()[0]
            x_vals = [np.log10(row[xvarlab]), np.log10(hc_df.loc[indx, xvarlab])]
            y_vals = [np.log10(row[yvarlab]), np.log10(hc_df.loc[indx, yvarlab])]
            axis[3].plot(x_vals, y_vals,  marker='o', color='k', markerfacecolor=plot_utiles.azure, linestyle ='-', markeredgecolor='k', ms = ms, zorder=2)
            ind_text = 0
            if row['Source'] == 'SgrB2(M)':
                ind_text = 1
                halign = 'center'
                valign = 'bottom'
                xm = 0
                ym = 0.04
            elif row['Source'] == 'SgrB2(N)':
                ind_text = 0
                halign = 'center'
                valign = 'bottom'
                xm = 0.0
                ym = 0.04
            axis[3].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                     ha=halign, va=valign, color = 'k', fontsize=anotsize)

    # BGNs 
    yvarlab = 'LIR/Mgas'
    yvarlab2 = 'LIR2/Mgas'
    xvarlab = 'SigmaIR_Lsun_pc2'
    xvarlab2 = 'SigmaIR2_Lsun_pc2'
    for i, row in bgn_df.iterrows():
        if np.isnan(row[xvarlab2]):
            xval2 = row[xvarlab]
        else:
            xval2 = row[xvarlab2]
        x_vals = [np.log10(row[xvarlab]), np.log10(xval2)]
        
        y_vals = [np.log10(row[yvarlab]), np.log10(row[yvarlab2])]
        axis[3].plot(x_vals, y_vals, marker='o', color='k', linestyle ='-', markerfacecolor = plot_utiles.green, markeredgecolor='k', ms = ms)
        ind_text = 0
        if row['Source'] == 'Arp220E':
            ind_text = 1
            halign = 'right'
            valign = 'bottom'
            xm = -0.05
            ym = 0.04
        elif row['Source'] == 'NGC4418':
            ind_text = 1
            halign = 'center'
            valign = 'bottom'
            xm = -0.08
            ym = 0.02
        elif row['Source'] == 'Arp220W':
            halign = 'right'
            valign = 'top'
            xm = -0.05
            ym = -0.03
        elif row['Source'] == 'Zw049.057':
            halign = 'center'
            valign = 'top'
            xm = 0.0
            ym = -0.06
        elif row['Source'] == 'IC860':
            ind_text = 1
            halign = 'center'
            valign = 'bottom'
            xm = 0
            ym = 0.04
        elif row['Source'] == 'Mrk231':
            ind_text = 0
            halign = 'center'
            valign = 'bottom'
            xm = 0.0
            ym = 0.04
        axis[3].annotate(row['Source'], xy=(x_vals[ind_text],y_vals[ind_text]), xytext=(x_vals[ind_text]+xm,y_vals[ind_text]+ym),
                 ha=halign, va=valign, color = 'k', fontsize=anotsize)

    axis[3].set_ylabel(r'$\log{L_{\text{IR}}/M_{\text{H2}}}$ (L$_{\odot}$/M$_{\odot}$)', fontsize=labelsize)
    axis[3].set_xlabel(r'$\log{\Sigma_{\text{IR}}}$ (L$_{\odot}$/pc$^{2}$)', fontsize=labelsize)    
    axis[3].set_ylim([0.75, 2.7])
    axis[3].set_xlim([4.7, 8.55])
    
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
        
    fig.savefig(f'{fig_path}{fig_name}LIR_comp_ALL_big_v3.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()