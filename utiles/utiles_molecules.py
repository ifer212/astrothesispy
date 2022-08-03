from scipy.optimize import curve_fit

import numpy as np
import astropy.units as u
import astropy.constants.si as _si

from astrothesispy.utiles import utiles
from astrothesispy.utiles import u_conversion

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
   return Ntot, Ntot_err, Tex, Tex_err

def Columndensity_thin(freq_GHz, Snu, Snu_err, Jup, elec_dipole_D, Bmin_arcsec, Bmaj_arcsec):
    """
    Column density of the upper level in the optical thin limit
    given the transitiion integrated line intensiy in K kms
    for a linear molecule
    Goldsmith 1999
    Returns Nu/gu in cm-2
    """
    # integrated line intensities in K cm/s
    W =  u_conversion.Jybeamkms_to_Tkms(Snu, freq_GHz, Bmin_arcsec, Bmaj_arcsec)*(1e5) # K cm / s
    W_err = u_conversion.Jybeamkms_to_Tkms(Snu_err, freq_GHz, Bmin_arcsec, Bmaj_arcsec)*(1e5) # K cm / s
    # Constants
    elec_dipole = elec_dipole_D * 3.1623E-25 *((u.J*1e6*u.cm**3)**(1/2.))
    h_si = _si.h
    k_si = _si.k_B
    c_si = _si.c.to(u.cm/u.s)
    freq_Hz = freq_GHz*1e9*(u.s**-1)
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


def Rdiag_rot_temp(vib_states, slims_df, ax1, line_width=1, xmax=1000, bootstrap=False, niter=1000, plot = False, weights=True):
    # Separating by vibrational state (Trot)
    # different J from same vib. state
    vib_fits = {}
    print('\tRotational temperatures')
    for v, vib in enumerate(vib_states):
        slims_df['vib_selection'] = (slims_df['vib'] == vib)
        vib_selection = slims_df[slims_df['vib_selection'] == True]
        if weights== True:
            weights_s = 1/(vib_selection['log10_Nugu_relerr']**2)
        else:
            weights_s = [1]*len(vib_selection['log10_Nugu_relerr'])
        # checking if enough values with no upper limits 
        #if (vib_selection['uplim']).values.sum() < len(vib_selection['uplim']):
        if (vib_selection['uplim']).values.sum() == 0: # All J are detected within specified vib
            print(f'\tRotational temperature for: {vib}')
            if bootstrap:
                fun = lambda p, x : p[0]*x+p[1]
                popt = [-0.001, 15.2]
                bpfit, bperr = utiles.fit_bootstrap(p0=popt, datax=vib_selection['Eu/k'], datay=vib_selection['log10_Nugu'], function=fun, yerr_systematic=vib_selection['log10_Nugu_relerr'], niter=niter)
                bNtot, bNtot_err, bTex, bTex_err = HC3N_rotational_diagram_from_fit(bpfit[0], bperr[0], bpfit[1], bperr[1])
                print('\tWith Bootstrap:')
                print('\t\tNtot=\t'+'%1.2E' % bNtot+' +- '+'%1.2E' % bNtot_err)
                print('\t\tTex=\t'+'%1.2f' % bTex+' +- '+'%1.6f' % bTex_err)
            else:
                popt, pcov = curve_fit(utiles.linfit, vib_selection['Eu/k'], vib_selection['log10_Nugu'], sigma=weights_s)#sigma=1./(vib_selection['log10_Nugu_relerr']**2))
                perr = np.sqrt(np.diag(pcov)) 
                m = popt[0]
                m_err = perr[0]
                b = popt[1]
                b_err = perr[1]
                print('\tWithout Bootstrap:')
                Ntot, Ntot_err, Tex, Tex_err = HC3N_rotational_diagram_from_fit(m, m_err, b, b_err)
                print('\t\tNtot=\t'+'%1.2E' % Ntot+' +- '+'%1.2E' % Ntot_err)
                print('\t\tTex=\t'+'%1.2f' % Tex+' +- '+'%1.2f' % Tex_err)
                vib_fits[vib] = {'vib': vib,
                                 'Ntot': Ntot, 'Ntot_err': Ntot_err, 
                                'Tex': Tex, 'Tex_err': Tex_err,    
                                'm': m, 'm_err': m_err,
                                'b': b, 'b_err':b
                                }
            if plot:
                xx = np.arange(np.nanmin(vib_selection['Eu/k'])-0.2*np.nanmin(vib_selection['Eu/k']), np.nanmax(vib_selection['Eu/k'])+0.2*np.nanmax(vib_selection['Eu/k']))
                if bootstrap:
                    ax1.plot(xx, utiles.linfit(xx, bpfit[0], bpfit[1]), '--', marker=None, color='0.4',linewidth=line_width, zorder=2)
                else:
                    ax1.plot(xx, utiles.linfit(xx, popt[0], popt[1]), '--', marker=None, color='0.4',linewidth=line_width, zorder=2)
            if bootstrap:
                bTex_err = np.abs(bTex_err)
                if bTex < 0:
                    bTex = 0
                if bTex_err > bTex:
                    bTex = 0
                if bTex > 2000: # establishing an upper limit on temperature
                    bTex = 0
                if bTex == 0:
                    bTex_err = 0
                vib_fits[vib] = {'vib': vib,
                                 'bNtot':  bNtot,
                                 'bNtot_err': np.abs(bNtot_err),
                                 'bTex':  bTex,
                                 'bTex_err': np.abs(bTex_err)}
        else:
            # All are uplims
            print('All values are uplims, skipping')
            vib_fits[vib] = {'vib': vib,
                             'Ntot': np.nan, 'Ntot_err': np.nan, 
                             'Tex': np.nan, 'Tex_err': np.nan,    
                             'm': np.nan, 'm_err': np.nan,
                             'b': np.nan, 'b_err': np.nan, 
                             'bNtot':np.nan, 'bNtot_err':np.nan,
                             'bTex':np.nan , 'bTex_err':np.nan
                            }
    return vib_fits
     

def Rdiag_vib_temp_uplims(rot_states, slims_df, ax1, line_width=1,
                          use_only={}, xmax = 1200, write_bootstrap=True, weights=True,
                          write=False, text_pos = [], anot_fontsize=13, uplims=False,
                          bootstrap=False, niter=1000, plot=False):
    """
    For Tvib im allowing v6 to be uplim
    """

    
    rot_fits = {}
    print('\tVibrational temperatures')
    for r, rot in enumerate(rot_states):
        slims_df['rot_selection'] =  (slims_df['jup'] == rot)
        rot_selection = slims_df[slims_df['rot_selection'] == True]
        # Only using values with no upper lims
        if uplims:
            rot_selection = rot_selection[rot_selection['uplim'] == False]
        if len(use_only)>0:
            # subSelecting only som vib states
            rot_selection['vib_selection'] = (rot_selection['vib'].isin(use_only[rot]))
            rot_selection = rot_selection[rot_selection['vib_selection'] == True]
            rot_selection.drop(rot_selection[(rot_selection.vib =='v6')].index, inplace=True)
        if weights== True:
            weights_s = 1/(rot_selection['log10_Nugu_relerr']**2)
        else:
            weights_s = [1]*len(rot_selection['log10_Nugu_relerr'])
        
        if rot ==39:
            color = '#fd3c06'
        elif rot ==26:
            color = '#0165fc'
        elif rot ==24:
            color = '#02ab2e'
        else:
            color = '0.6'
        #if (rot_selection['uplim']).values.sum() < len(rot_selection['uplim']):
        if not rot_selection.empty:
            #if ((rot_selection['uplim']).values.sum() == 0) or ((rot_selection['uplim']).values.sum() == 1 and rot_selection[(rot_selection.vib =='v6=1')]['uplim'].tolist()[0]): # All vib states are detected in that J 
            if ((rot_selection['uplim']).values.sum() == 0) or ((rot_selection['uplim']).values.sum() == 1 and not rot_selection[(rot_selection.vib =='v7=1')]['uplim'].tolist()[0]): # All v=0 and v7=1 are detected
                print(f'\tVibrational temperature for: {rot}-{rot-1}')
                if bootstrap:
                    fun = lambda p, x : p[0]*x+p[1]
                    popt = [-0.001, 15.2]
                    bpfit, bperr = utiles.fit_bootstrap(p0=popt, datax=rot_selection['Eu/k'], datay=rot_selection['log10_Nugu'], function=fun, yerr_systematic=rot_selection['log10_Nugu_relerr'], niter=niter)
                    bNtot, bNtot_err, bTex, bTex_err = HC3N_rotational_diagram_from_fit(bpfit[0], bperr[0], bpfit[1], bperr[1])
                    print('\tWith Bootstrap:')
                    print('\t\tNtot=\t'+'%1.2E' % bNtot+' +- '+'%1.2E' % bNtot_err)
                    print('\t\tTex=\t'+'%1.2f' % bTex+' +- '+'%1.6f' % bTex_err)
                else:
                    popt, pcov = curve_fit(utiles.linfit, rot_selection['Eu/k'], rot_selection['log10_Nugu'], sigma=weights_s, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov)) 
        
                    print('\tWithout Bootstrap:')
                    m = popt[0]
                    m_err = perr[0]
                    b = popt[1]
                    b_err = perr[1]
                    Ntot, Ntot_err, Tex, Tex_err = HC3N_rotational_diagram_from_fit(m, m_err, b, b_err)
                    print('\t\tNtot=\t'+'%1.2E' % Ntot+' +- '+'%1.2E' % Ntot_err)
                    print('\t\tTex=\t'+'%1.2f' % Tex+' +- '+'%1.2f' % Tex_err)
                    rot_fits[rot] = {'jup': rot,
                                     'Ntot': Ntot, 'Ntot_err': Ntot_err, 
                                    'Tex': Tex, 'Tex_err': Tex_err,  
                                    'm': m, 'm_err': m_err,
                                    'b': b, 'b_err':b
                                    }
                
                if plot:
                    xx = np.arange(np.nanmin(rot_selection['Eu/k'])-100, xmax)
                    if bootstrap==False:
                        ax1.plot(xx, utiles.linfit(xx, m, b), '-', marker=None, color=color,linewidth=line_width, zorder=2)
                    else:
                        ax1.plot(xx, utiles.linfit(xx, bpfit[0], bpfit[1]), '-', marker=None, color=color,linewidth=line_width, zorder=2)
                    xx = np.arange(np.nanmin(rot_selection['Eu/k'])-0.2*np.nanmin(rot_selection['Eu/k']), np.nanmax(rot_selection['Eu/k'])+0.2*np.nanmax(rot_selection['Eu/k']))
                if bootstrap:
                    bTex_err = np.abs(bTex_err)
                    if bTex < 0:
                        bTex = 0
                    if bTex_err > bTex:
                        bTex = 0
                    if bTex > 2000: # establishing an upper limit on temperature
                        bTex = 0
                    if bTex == 0:
                        bTex_err = 0
                    rot_fits[rot] = {'jup': rot,
                                     'bNtot':  bNtot,
                                     'bNtot_err': np.abs(bNtot_err),
                                     'bTex':  bTex,
                                     'bTex_err': np.abs(bTex_err)}
                if write:
                    ax1.plot(xx, utiles.linfit(xx, bpfit[0], bpfit[1]), '-', marker=None, color=color,linewidth=0.75, zorder=2)
                    jtext = rot_selection['J'].tolist()[0].split('_')
                    jtext2 = '-'.join(jtext)
                    text1 = r'$\rm{T}_{'+jtext2+'} ='+'%1.1f' % bTex+'\pm'+'%1.1f' % np.abs(bTex_err)+'\,K$'#+ '\n'
                    #text2 = r'$\log{N} ='+'%1.1f' % np.log10(bNtot) +' ('+'%1.1f' % np.log10(bNtot_err)+') \,{cm}^{-2}$'
                    text = text1
                    ax1.text(text_pos[0], text_pos[1],text, ha='left', va='center', color=color, fontsize=anot_fontsize, transform=ax1.transAxes)
            else:
                # All are uplims
                print('All values are uplims, skipping')
                rot_fits[rot] = {'rot': rot,
                                 'Ntot': np.nan, 'Ntot_err': np.nan, 
                                 'Tex': np.nan, 'Tex_err': np.nan,    
                                 'm': np.nan, 'm_err': np.nan,
                                 'b': np.nan, 'b_err': np.nan, 
                                 'bNtot':np.nan, 'bNtot_err':np.nan,
                                 'bTex':np.nan , 'bTex_err':np.nan
                                }
        else:
            # Empty
            rot_fits[rot] = {'rot': rot,
                                 'Ntot': np.nan, 'Ntot_err': np.nan, 
                                 'Tex': np.nan, 'Tex_err': np.nan,    
                                 'm': np.nan, 'm_err': np.nan,
                                 'b': np.nan, 'b_err': np.nan, 
                                 'bNtot':np.nan, 'bNtot_err':np.nan,
                                 'bTex':np.nan , 'bTex_err':np.nan
                                }
    return rot_fits

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