from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants.si as _si
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from astrothesispy.utiles import utiles
from astrothesispy.utiles import u_conversion

# Loading amsmath LaTeX
import matplotlib as mpl
mpl.rc('text', usetex=True)

#==============================================================================
#                    
#                   Utils: Recurrent formulas and functions for Molecules stuff
#
#==============================================================================

def trans_temperature(wavelength):
    """
    To get temperature of the transition in K
    Wavelength in micros
    T = h*f / kB
    """
    w = u.Quantity(wavelength, u.um)
    l = w.to(u.m)
    c = _si.c.to(u.m / u.s)
    h = _si.h.to(u.eV * u.s)
    kb = _si.k_B.to(u.eV / u.K)
    f = c/l
    t = h*f/kb
    return t

def trans_energy(freq):
    """
    # Energy of a transition
    return Energy in Kelvins of a transition given in frecuency
    E=h*nu
    freq in GHz
    """
    nu = u.Quantity(freq, u.GHz)
    h_si = _si.h # Planck constant
    k_si = _si.k_B # boltzmann constant
    E = h_si*(nu.to(u.Hz))/u.Hz/u.s # Joules
    E_K = E/k_si
    return E_K.value

def trans_freq(E):
    """
    # Frequency of a transition
    return freq in Hz of a transition given its energy in K
    E=h*nu
    nu=E/h
    freq in GHz
    """
    #freq = 354.1
    h_si = _si.h # Planck constant
    k_si = _si.k_B # boltzmann constant
    nu = E*k_si.value/h_si.value
    return nu

def vibexc_HC3N(freq_Hz):
    """
    Einstein coefficient given transition frequency
    Deguchi 1979
    """
    h_si = _si.h
    c_si = _si.c
    A = 64*(np.pi**4)/(3*h_si)*((freq_Hz/c_si)**3)
    return A

def tau_transition(N_up, B_ul, freq, FWHM, T):
    """
    Opcity of a transition
    Goldsmith & Langer 1999
    """
    h_si = _si.h
    k_si = _si.k_B
    tau = (h_si/FWHM)* N_up * B_ul * (np.exp(h_si*freq/(k_si*T))-1.)
    return tau

def partition_function(T,B0):
    """
    Goldsmith & Lager 1999
    Linear Molecules in LTE -> Z ~ a^-1
    HC3N B0 = 4.55 GHz Werni et al. 2007 
    """
    h_si = _si.h
    k_si = _si.k_B
    Z = k_si*T/(h_si * (B0.to(u.Hz))) # Sale 1/(Hz * s) que se supone que es adim, pero astropy no lo coge
     
    return Z.value

def tau_HC3N(T, FWHM, B0, logN, J):
    """
    Optical depth of a linear molecule J -> J-1 transition
    Costagliola & Aalto 2010 (eq. A.2)
    """
    # For tau << 1 the N observed are the real col. densities but for tau > 1
    # the observed line intensities lead to an underestimate of the derived column density
    # if no correction is applied.
    
    h_si = _si.h
    mu = 3.6 # dipole moment of HC3N Wood et al. 2009 185, 273-288
    
    # Partition function for linear molecs in LTE
    Z = partition_function(T,B0)
    alpha = 1./Z
    
    # Column density
    N = 10.**logN
    tau = 8 * (np.pi**2.) * mu**2 * N * J * np.exp(-alpha*J*(J+1.))*(np.exp(2.*alpha*J)-1.) / (3.*h_si*Z*FWHM)
    return tau

def Tex_goldsmith(Tkin, Tcore, trans_freq, Crate, Arate, f):
    """
    Excitation Temperature (radiation excitation and collisonal excitation)
    Two level approach by Goldsmith1982
    Crate   -> Downward collison rate
    Arate   -> Downward stimulated transitions rate
    Tkin    -> Gas kinetic temperature
    Tcore   -> Source temperature
    f       -> Filling factor
    """
    # Equivalent temperature of the transition
    Tstar = trans_energy(trans_freq)
    coefs = Crate/Arate
    g = f*((np.exp(Tstar/Tcore)-1.)**-1)
    Tex = Tstar*(((Tstar/Tkin)+np.log((1.+g+coefs)/(g*np.exp(Tstar/Tkin)+coefs)))**-1)
    return Tex
    
def Tex_goldsmith_rad(Tkin, Tcore, trans_freq, f):
    """
    Excitation Temperature (purely radiation excitation C->0)
    Two level approach by Goldsmith1982
    Crate   -> Downward collison rate
    Arate   -> Downward stimulated transitions rate
    Tkin    -> Gas kinetic temperature
    Tcore   -> Source temperature
    f       -> Filling factor
    """
    # Equivalent temperature of the transition
    Tstar = trans_energy(trans_freq)
    Tex = Tstar * (1./(np.log(np.exp(Tstar/Tcore)-1.+f)-np.log(f)))
    return Tex

def Tex_goldsmith_col(Tkin, trans_freq, Crate, Arate):
    """ 
    Excitation Temperature (purely collisional excitation g->0)
    Two level approach by Goldsmith1982
    Crate   -> Downward collison rate
    Arate   -> Downward stimulated transitions rate
    Tkin    -> Gas kinetic temperature
    Tcore   -> Source temperature
    f       -> Filling factor
    """
    # Equivalent temperature of the transition
    Tstar = trans_energy(trans_freq)
    coefs = Arate/Crate
    print('ncrit=%1.2E' % coefs)
    Tex = Tstar /((Tstar/Tkin)+np.log(1.+coefs))
    return Tex

def transition_citical_density(freq,J,mu,q_Jj):
    """
    J -> J-1 Transition critical density
    Costagliola & Aalto 2010
    freq in Hz
    J = J
    j = J-1
    q == collisional coefficients
    mu == electric dipole moment of the molecule
    ncrit = A_Jj/q_Jj
    """
    h_si = _si.h
    c_si = _si.c.to(u.cm/u.s)
    ncrit = 64*(np.pi**4)*(freq**3)*(mu**2)*(J+1.)/(3*h_si*(c_si**3)*(2.*J+3.)*q_Jj)
    return ncrit

def dens_from_Tex_col(Tkin, Tex, trans_freq, Crate, Arate):
    Tstar = trans_energy(trans_freq)
    coefs = Arate/Crate
    dens = coefs/(np.exp((Tstar/Tex) - (Tstar/Tkin))-1)
    return dens


def vib_col(tkin):
    """
    Deesxcitation collisional cross section between vib. states (Goldsmith1982)
    """
    csec = (3E-12)*np.exp(-4.8*(tkin**-1./3))
    return csec

def linfit(x,m,b):
    """
    m = -1/(T*np.log(10))
    b = np.log10(Ntot/Z)
    x = Eu/k
    y = np.log10(Nu/gu)
    """
    return m*x+b


        
def line_name_formatter(line_name):
    """
        Reads quantum numbers from string name

    Args:
        line_name (str): line name

    Returns:
        line_dict (dict): line dictionary with quantum numbers
    """
    qn1, qn2, qn3 = '', '', ''
    qnn1, qnn2, qnn3 = '', '', ''
    if '_v' in line_name:
                sub_line_name = line_name.split('_v')
                line_name = f'{sub_line_name[0]}/v{sub_line_name[1]}'
    line_qnms = line_name.split('_')
    vib_state = line_qnms[0]
    qn1 = int(line_qnms[1])
    if len(line_qnms) < 4:
        qnn1 = int(line_qnms[2])
    else:
        qn2 = int(line_qnms[2])
        if len(line_qnms) < 6:
            qnn1 = int(line_qnms[3])
            qnn2 = int(line_qnms[4])
            if vib_state == 'v6=v7=1':
                # No 3rd quantun number in name
                qn3 = np.abs(qn2)
                qnn3 = np.abs(qn2)
        else:
            qn3 = int(line_qnms[3])
            qnn1 = int(line_qnms[4])
            qnn2 = int(line_qnms[5])
            qnn3 = int(line_qnms[6])
    formula = f'HC3N,{vib_state}'
    line_dict = {'Formula': formula,
                 'vib_state': vib_state,
                 'qn1': qn1,
                 'qn2': qn2,
                 'qn3': qn3,
                 'qnn1': qnn1,
                 'qnn2': qnn2,
                 'qnn3': qnn3,
                 }
    return line_dict


def HC3N_line_dict_builder(source, results_path, info_path, Bmin_arcsec, Bmaj_arcsec):
    electric_dipole = 3.742 # Electric dipole in Debyes for HC3N
    trans_info = pd.read_excel(info_path, header=0)
    trans_info.fillna('', inplace=True)
    trans_info['Frequency_GHz']  = trans_info['Frequency']/1000
    trans_info['ELO_K'] = u_conversion.Energy_cmtoK(trans_info['ELO'])
    trans_info['EUP_K'] =  trans_info['ELO_K'] + trans_energy(trans_info['Frequency_GHz'])
    formulas = trans_info['Formula'].unique()
    lines_dict = {}
    for formula in formulas:
        lines_dict[formula] = []
    # Observed fluxes 
    fluxes_path = f'{results_path}Tables/{source}_flujos_hc3n_python.xlsx'
    trans_fluxes = pd.read_excel(fluxes_path, header=0)
    trans_fluxes_ring4 = trans_fluxes[(trans_fluxes['ring']==4)]
    drop_cols = ['ring', 'dist', 'cont_', 'beam_345', 'Unnamed: 0']
    only_219beam_cols = [s for s in trans_fluxes_ring4.columns if not any(x in s for x in drop_cols)]
    trans_fluxes_ring4_219beam = trans_fluxes_ring4[only_219beam_cols].copy()
    for col in trans_fluxes_ring4_219beam.columns:
        if 'err' not in col and 'mJy' in col:
            line_name = col.split('_SM')[0]
            line_dict = line_name_formatter(line_name)
            subset = (trans_info['Formula']==line_dict['Formula']) & (trans_info['qn1']==line_dict['qn1']) & (trans_info['qn2']==line_dict['qn2']) & (trans_info['qn3']==line_dict['qn3'])
            sub_info = trans_info[subset].copy()
            # Getting upper column densities
            Nugu, Nugu_err = Columndensity_thin(freq_GHz=sub_info['Frequency_GHz'],
                                                                 Snu=trans_fluxes_ring4_219beam[col]/1000, Snu_err=trans_fluxes_ring4_219beam[col+'_err']/1000,
                                                                 Jup=sub_info['qn1'], elec_dipole_D=electric_dipole,
                                                                 Bmin_arcsec=Bmin_arcsec, Bmaj_arcsec=Bmaj_arcsec)
            line_dict['Frequency_GHz'] = sub_info['Frequency_GHz'].to_list()[0]
            line_dict['Nu/gu'] = Nugu.to_list()[0]
            line_dict['logNu/gu'] = np.log10(Nugu).to_list()[0]
            line_dict['Nu/gu_err'] = Nugu_err.to_list()[0]
            line_dict['logNu/gu_err'] = np.log10(Nugu_err).to_list()[0]
            line_dict['EUP_K'] = sub_info['EUP_K'].to_list()[0]
            lines_dict[line_dict['Formula']].append(line_dict)
    return lines_dict

def rotational_diagram(lines_dict, Jselect, bootstrap=True):
    """
        Obtains the vibrational temperature
    Args:
        lines_dict (dict): Dictionary with the lines information.
        Jselect (int): Jup quantum rotational number to make the fit.
        bootstrap (bool, optional): Use bootstrap. Defaults to True.

    Returns:
        vib_temp_df (df): DataFrame with the lines used info.
        fit_dict (dict): Dictionary with the fitted values.
    """
    Jselect = 24
    vib_temp = [] # Similar Jup (qn1) but different vib state
    for line in lines_dict:
        for trans in lines_dict[line]:
            if Jselect -2 <= trans['qn1'] <= Jselect +2:
                vib_temp.append(trans)
    vib_temp_df = pd.DataFrame(vib_temp)
    vib_temp_df['logNu/gu_relerr'] = vib_temp_df['logNu/gu_err']/np.log(10)/vib_temp_df['logNu/gu']
    print('———————————–')
    # Fitting without bootstap (needed to get first guess for bootstrap anyway)
    popt, pcov = curve_fit(linfit, vib_temp_df['EUP_K'], vib_temp_df['logNu/gu'], sigma=vib_temp_df['logNu/gu_relerr'])
    perr = np.sqrt(np.diag(pcov)) 
    Ntot, Ntot_err, Tex, Tex_err = HC3N_rotational_diagram_from_fit(popt[0], perr[0], popt[1], perr[1])      
    print('No bootstrap:')
    Ntext = r'N$_{\rm{Tot}}$='+f'${utiles.latex_float(Ntot)}\pm{utiles.latex_float(Ntot_err)}'+r'$cm$^{-1}$'    
    Ttext = r'T$_{\rm{vib}}$='+f'${Tex:1.1f}\pm{Tex_err:1.1f}$ K'
    print(f'{Ntext} \t {Ttext}')
    fit_dict = {'NoBootstrap':  {'popt': popt,
                                 'pcov': perr,
                                 'Ntot': Ntot,
                                 'Ntot_err': Ntot_err,
                                 'Tex': Tex,
                                 'Tex_err': Tex_err,
                                 'Ntext': Ntext,
                                 'Ttext': Ttext}}
    if bootstrap:
        fun = lambda p, x : p[0]*x+p[1]
        bpfit, bperr = utiles.fit_bootstrap(p0 = popt, datax = vib_temp_df['EUP_K'], datay = vib_temp_df['logNu/gu'],
                                            function=fun, yerr_systematic=vib_temp_df['logNu/gu_relerr'], niter=1000, montecarlo=True)
        bNtot, bNtot_err, bTex, bTex_err = HC3N_rotational_diagram_from_fit(bpfit[0], bperr[0], bpfit[1], bperr[1])
        print('Bootstrap:')
        BNtext = r'N$_{\rm{Tot}}$='+f'${utiles.latex_float(bNtot)}\pm{utiles.latex_float(bNtot_err)}'+r'$cm$^{-1}$' 
        BTtext = r'T$_{\rm{vib}}$='+f'${bTex:1.1f}\pm{bTex_err:1.1f}$ K'
        print(f'{BNtext} \t {BTtext}')
        fit_dict['Bootstrap'] = {'popt': bpfit,
                                 'pcov': bperr,
                                 'Ntot': bNtot,
                                 'Ntot_err': bNtot_err,
                                 'Tex': bTex,
                                 'Tex_err': bTex_err,
                                 'Ntext': BNtext,
                                 'Ttext': BTtext}
    return vib_temp_df, fit_dict

def HC3N_rotational_diagram_from_fit(slope, slope_err, intercept, intercept_err):
   """
   Getting Ntot and Tex from the slope and intecept and partition function Z of a rotational diagram
   Nu/gu = (Ntot/Z) * e^(-Eu/kTex)
   ln(Nu/gu) = Eu*(-1/kTex)+ln(Ntot/Z)
   y = x*m + b
   m = (-1/Tex*np.log(10))
   b = ln(Ntot/Z)
   Z(HC3N) = kT/hB0
   B0 = 4.55GHz
   """
   B0 = (4.55*u.GHz).to(u.Hz).value
   h_si = _si.h
   k_si = _si.k_B
   Tex = -1./(np.log(10.)*slope)
   Tex_err = -slope_err/(np.log(10.)*(slope**2))
   Z = k_si.value*Tex/(h_si.value*B0)
   Z_err = k_si.value*Tex_err/(h_si.value*B0)
   Ntot = Z*10**(intercept)
   Ntot_err = np.sqrt(((Z_err)**2)+((intercept_err*np.log(10.)*10**(intercept))**2))
   return Ntot, np.abs(Ntot_err), Tex, np.abs(Tex_err)

def Rotational_Diagram_plot(source, vib_temp_df, fit_dict, fig_path, plot_noboots = True, plot_boots = True,
                            noboots_col = '0.75', boots_col = 'r', colormarker = 'k', edgemarker = 'k',
                            markersize = 6, fontsize = 9, tick_fontsize = 8, label_fontsize = 10, fig_format = '.pdf'):
    markers = np.random.choice(Line2D.filled_markers, len(vib_temp_df['Formula'].unique()))
    fig = plt.figure()
    ax = fig.add_subplot((111))
    for v, vib in enumerate(vib_temp_df['Formula'].unique()):
        vibs = vib_temp_df[vib_temp_df['Formula'] == vib].copy()
        ax.errorbar(vibs['EUP_K'], vibs['logNu/gu'], yerr=(vibs['logNu/gu_relerr']),
                                    marker= markers[v], markersize=markersize,
                                    markerfacecolor=colormarker,
                                    markeredgecolor='k', markeredgewidth=0.6,
                                    ecolor='k',
                                    color = colormarker,
                                    elinewidth= 0.6,
                                    barsabove= True,
                                    zorder=1,
                                    linestyle = '')
        ax.plot(vibs['EUP_K'], vibs['logNu/gu'],
                        marker=markers[v], markersize=markersize,
                        markerfacecolor=colormarker,
                        markeredgecolor=edgemarker, markeredgewidth=0.6,
                        linestyle = '', zorder=3, label = vib )
    xx = np.arange( vib_temp_df['EUP_K'].min(skipna=True) - 50, vib_temp_df['EUP_K'].max(skipna=True) + 50)
    if plot_noboots:
        ax.plot(xx, linfit(xx, fit_dict['NoBootstrap']['popt'][0], fit_dict['NoBootstrap']['popt'][1]),
                linestyle = '-', marker=None, color=noboots_col,linewidth=0.75, zorder=2)
        legend_elements = [Line2D([0], [0], color=noboots_col, lw=0.75, label=fit_dict["NoBootstrap"]["Ntext"]+ ' ' +fit_dict["NoBootstrap"]["Ttext"])]
    if plot_boots:
        if ~plot_noboots:
            legend_elements = []
        ax.plot(xx, linfit(xx, fit_dict['Bootstrap']['popt'][0], fit_dict['Bootstrap']['popt'][1]),
                linestyle = '-', marker=None, color=boots_col,linewidth=0.75, zorder=2)
        legend_elements.append(Line2D([0], [0], color=boots_col, lw=0.75, label=fit_dict["Bootstrap"]["Ntext"]+ ' ' +fit_dict["Bootstrap"]["Ttext"]))
    ax.xaxis.set_tick_params(top =True, labeltop=False)
    ax.yaxis.set_tick_params(right=True, labelright=False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(labelsize=tick_fontsize)
    plt.ylabel(r'log (N$_{\rm u}$/g$_{\rm u}$)', fontsize = label_fontsize)
    plt.xlabel(r'E$_{\rm u}$/k (K)', fontsize = label_fontsize)
    ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=fontsize)
    plt.savefig(f'{fig_path}{source}_Rotational_Diagram{fig_format}', bbox_inches='tight', transparent=True, dpi=300)
    plt.close(fig)

def Columndensity_thin(freq_GHz, Snu, Snu_err, Jup, elec_dipole_D, Bmin_arcsec, Bmaj_arcsec):
    """
    Column density of the upper level in the optical thin limit
    given the transitiion integrated line intensiy in Jy km/s (converted inside to K kms)
    for a linear molecule
    Goldsmith 1999
    Returns Nu/gu in cm-2
    """
    if (isinstance(Snu, pd.Series)):
        Snu = Snu.to_list()[0]
        Snu_err = Snu_err.to_list()[0]
    # integrated line intensities in K cm/s
    W =  u_conversion.Jybeamkms_to_Tkms(Snu, freq_GHz, Bmin_arcsec, Bmaj_arcsec)*(1e5) # K cm / s
    W_err = u_conversion.Jybeamkms_to_Tkms(Snu_err, freq_GHz, Bmin_arcsec, Bmaj_arcsec)*(1e5) # K cm / s
    # Constants
    elec_dipole = elec_dipole_D * 3.1623E-25 *((u.J*1e6*u.cm**3)**(1/2.))
    h_si = _si.h
    k_si = _si.k_B
    c_si = _si.c.to(u.cm/u.s)
    if (isinstance(freq_GHz, pd.Series)):
        freq_GHz = freq_GHz.to_list()[0]
    freq_Hz = freq_GHz*1e9*(u.s**-1)
    if (isinstance(Jup, pd.Series)):
        Jup = Jup.to_list()[0]
    gu = 2.*Jup+1
    # Einstein coefficient
    Aul = 64.*(np.pi**4)*(freq_Hz.value**3)*(elec_dipole.value**2)*Jup/(3.*h_si.value*(c_si.value**3)*(2*Jup+1))
    gamma_u = 8.*np.pi*k_si.value*(freq_Hz.value**2)/(h_si.value*(c_si.value**3)*Aul)
    # Nu/gu
    Nuthin_gu = gamma_u*W/gu
    Nuthin_gu_err = gamma_u*W_err/gu
    return Nuthin_gu, Nuthin_gu_err

def Columndensity_thick(freq_GHz, Snu, Snu_err, Jup, elec_dipole_D, Bmin_arcsec, Bmaj_arcsec, tau):
    """
    Column density of the upper level
    given the transitiion integrated line intensiy in K kms
    for a linear molecule
    Goldsmith 1999
    Returns Nu/gu in cm-2
    """
    Nuthin_gu, Nuthin_gu_err = Columndensity_thin(freq_GHz, Snu, Snu_err, Jup, elec_dipole_D, Bmin_arcsec, Bmaj_arcsec)
    Ctau = tau/(1.-np.exp(-tau))
    Nu_gu = Nuthin_gu*Ctau
    Nu_gu_err = Nuthin_gu_err*Ctau
    return Nu_gu, Nu_gu_err

def HC3N_levelpopulation(Snu, Snu_err, freq_GHz, Bmin_arcsec, Bmaj_arcsec, Jup):
    """
    Level population for HC3N
    Rotational constant B0 = 4.55GHz
    Dipole moment  mu = 3.724 Debye
    Linear Molecule line strength Sij = J+1
    Partition function approx Z=kT/hB0
    Erot = B0*J(J+1)
    mu = 2B(J+1)
    De la tesis de rivilla:
    We assume the emission is optically thin,
    and that the medium is in Local Thermodynamic Equilibrium (LTE).
    This implies that the level populations are described by the Maxwell-Boltzmann
    distribution with temperature T
    """
    Sij = Jup+1.
    freq_MHz = freq_GHz*1000.
    h_si = _si.h
    k_si = _si.k_B
    freq_Hz = freq_GHz*1e9*(u.s**-1)
    teta_vib = h_si*freq_Hz/k_si
    mu = 3.724
    TBdv = u_conversion.Jybeamkms_to_Tkms(Snu, freq_GHz, Bmin_arcsec, Bmaj_arcsec)
    # From Zhang et al 1998
    Nugu = np.log10(1.669E17*TBdv/(Sij*freq_MHz*(mu**2)))
    TBdv_err = u_conversion.Jybeamkms_to_Tkms(Snu_err, freq_GHz, Bmin_arcsec, Bmaj_arcsec)
    Nugu_err = 1.669E17*TBdv_err/(Sij*freq_MHz*(mu**2))
    return Nugu, Nugu_err

def Trotational(logNu, logNl, Eu, El, Ju, Jl):
    """
    Rotational temperature from the boltzmann equation, between two different
    transitions
    nu/nl = gu/gl * exp(-AE/kT) -> nu and nl we can change it for col dens
    """
    h_si = _si.h
    c_si = _si.c
    # Degeneracies of the rotational states
    gu = 2.*Ju +1.
    gl = 2.*Jl +1.
    # Column densities
    Nu = 10.**logNu
    Nl = 10.**logNl
    # Boltzmann constant
    k_si = _si.k_B
    cm = 1. * h_si *c_si.to(u.cm / u.s) # cm-1 to J
    k_cm = k_si/(cm).value  # 0.69503476 # kB in cm-1 / K
    # Rotational temperature
    Trot = (Eu - El)/(k_cm * (np.log(gu/gl) - np.log(Nu/Nl)))
    # 153./(np.log(10**0.3) + np.log(71/49.)) = 144.1173
    return Trot

def tau_from_T(Tobs, Tkin):
    """
    Line optical depth from observed temperature and excitation temperature in Kelvin
    """
    tau = -np.log(1.-(Tobs/Tkin))
    return tau

def sourcesize_from_fit(Tmain_beam, Tex, Size_arcsec):
    """
    To retrieve source size (diameter) from observed temperature, extication temperature and beam size
    TMB * Size = Tex * Source_size
    """
    Source_size_arcsec = np.sqrt((Size_arcsec)*(Tmain_beam/Tex))
    return Source_size_arcsec


    