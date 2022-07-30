# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 23:10:13 2022

@author: Fer
"""

from astropy.modeling.models import BlackBody as blackbody_nu

from astrothesispy.utiles import utiles_alma
from astrothesispy.utiles import u_conversion

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants.si as _si
import matplotlib.pyplot as plt

#==============================================================================
#                    
#                   Utils: Recurrent formulas and functions for Physical stuff
#
#==============================================================================

class InputError(LookupError):
    '''Raise this when there's an error in the input parameter'''

def ion_prod_rate(freq_GHz, L_erg_s_Hz, Te):
    """
    Production rate of ionizing photons given a certain frequency and luminosity 
    at that frequency
    """
    # Murpy 2011 https://iopscience.iop.org/article/10.1088/0004-637X/737/2/67/pdf
    # Leroy 2019
    Q = (6.3E25)*((Te/1E4)**(-0.45))*(freq_GHz**0.1)*(L_erg_s_Hz)
    return Q

def ion_prod_rate_147(freq_GHz, L_erg_s_Hz, Te):
    """
    Production rate of ionizing photons given a certain frequency and luminosity 
    at that frequency
    """
    # Murpy 2011 https://iopscience.iop.org/article/10.1088/0004-637X/737/2/67/pdf
    # Leroy 2019
    Q = (147**0.1)*(6.3E25)*((Te/1E4)**(-0.45))*((freq_GHz/147)**0.1)*(L_erg_s_Hz)
    return Q

def ion_prod_rate_j(freq_GHz, Snu, Te, d_kpc):
    """
    Production rate of ionizing photons given a certain frequency and luminosity 
    at that frequency, formula by Jesus
    """
    Q = 2.14E48*((Te)**(-0.45))*(freq_GHz**0.1)*(Snu)*(d_kpc**2)
    return Q

def Paalpha_ion_prod_rate(L_pa_erg_s):
    """
    Ionizing production rate from Paschen alpha luminosity
    From Piqueras-Lopez 2016 y kennicut 1998 y osterbrock2006
    """
    Q = 6.8E-41*L_pa_erg_s*1.08E53
    return Q

def Paalpha_ion_prod_rate_v2(L_pa_erg_s):
    """
    From Herrero-Illana2014
    photons /s
    """
    Nphotos = L_pa_erg_s*6.27 *10**14
    return Nphotos

def thermal_cont_radio_from_Palpha(freq_GHz, Palpha_erg_s_cm2):
    """
    From Herrero-Illana2014
    Using standard relations between Palpha and Hbeta (Osterbrock1989) and
    Eq. 3 from Condon (1992) we can obtain the thermal radio continuum emission
    assuming T=10000 K  and Ne = 10**4 cm-3, tipical of compact (<=1pc) SB regions
    """
    Sthermal_mJy = 1.076E13 * Palpha_erg_s_cm2 * (freq_GHz**-0.1)
    return Sthermal_mJy

def spec_index(freqs, intens):
    # Spectral index between two points
    m = np.log10(intens[1]/intens[0])/np.log10(freqs[1]/freqs[0])
    return m

def ZAMSmass_from_Q0(Q0):
    # ZAMS star mass from Leroy 2019
    Mzams = Q0/4E46 # Msun
    return Mzams

def electron_density(Te, nu_GHz, dist_kpc, theta_arcmin, flux_Jy):
    """
    from Armijos-abedanos 2018
    """
    a = 0.98
    u = 0.775 # spherical geom
    ne_cm3 = 6.351E2 * u*(a**0.5)*((Te/1e4)**0.175)*(nu_GHz**0.05)*(flux_Jy**0.5)*(dist_kpc**-0.5)*(theta_arcmin**-1.5)
    return ne_cm3

def Lyman_photons(theta, n_e, n_p, alpha):
    """
    Number of Lyman ionising photons
    from Armijos-abedanos 2018
    np = proton density, equal to electron density ne under LTE
    alpha = recomb coefficient
    """
    Nlym = (4./3)*np.pi*((theta/2.)**3)*n_e*n_p*alpha
    return Nlym

def gas_mass_from_dust(I350_Jybeam, Tdust, freq_GHz, d_Mpc, k_nu_cm2_g=1.9):
    """
    Estimated gas mass from continuum emission at 350GHz from Leroy 2019
    k_nu_cm2_g = 1.9 #cm^2/g
    Berta, Lutz, Genzel et al. 2016: Measures of galaxy dust...
    """
    # Convert 350GHz intensity at the peak, to a dist optical depth
    #I350 = (1.-np.exp(-tau350))*B_nu(Tdust)
    # knu = 1.9 cm**2/g, appropiate for 350GHz and dust mixed with gas at 10^5-10^6 cm-3
    
    k_nu = k_nu_cm2_g*(u.cm**2/u.g).to(u.m**2/u.kg)
    I350 = I350_Jybeam *1E-23 # erg/cm2/s/Hz/sr
    L350 = I350*4*np.pi*(((d_Mpc*u.Mpc).to(u.cm).value)**2) # erg/s/Hz/sr
    Bnu_350 = blackbody_nu(freq_GHz*u.GHz, Tdust).value # erg/Hz/s/sr/cm^2
    Bnu_350_m2 = Bnu_350*1E4 # erg/Hz/s/sr/m^2
    Mdust = L350/(4*np.pi*k_nu*Bnu_350_m2) # kh
    Mdust_msun = (Mdust*u.kg).to(u.Msun)
    return Mdust_msun

def gas_mass_from_dust_leroy(I350_Jybeam, bmin_arsec, bmaj_arsec,
                             Tdust, freq_GHz, d_Mpc, upper_lim_size_arcsec = 0.5, lower_lim_size_arcsec = 0.024, k_nu_cm2_g=1.9):
    """
    Estimated gas mass from continuum emission at 350GHz from Leroy 2019
    k_nu_cm2_g = 1.9 #cm^2/g
    Berta, Lutz, Genzel et al. 2016: Measures of galaxy dust...
    """
    # Convert 350GHz intensity at the peak, to a dist optical depth
    #I350 = (1.-np.exp(-tau350))*B_nu(Tdust)
    beam_size_arsec2 = utiles_alma.beam_size(bmin_arsec, bmaj_arsec)
    beam_size_sr = (beam_size_arsec2*u.arcsec**2).to(u.sr)
    I350_Jysr = I350_Jybeam / beam_size_sr.value
    I350 = I350_Jysr *1E-23# I350_Jysr.value *1E-23 # erg/Hz/s/sr/cm^2
    Bnu_350 = blackbody_nu(freq_GHz*u.GHz, Tdust).value # erg/Hz/s/sr/cm^2
    tau350 = -np.log(1.-(I350/Bnu_350))
    DGR = 1/100.
    dust_sdens = tau350/(k_nu_cm2_g*DGR)*(u.g/u.cm**2)
    #dust_sdens_g_pc2 = dust_sdens*((u.g/u.cm**2).to(u.Msun/u.pc**2))
    up_size_cm = (u_conversion.lin_size(d_Mpc, upper_lim_size_arcsec, D_err = 0)).to(u.cm)
    low_size_cm  = (u_conversion.lin_size(d_Mpc, lower_lim_size_arcsec, D_err = 0)).to(u.cm) # 1" = 16.9pc
    lower_lim_area_cm2 = np.pi*((low_size_cm/2.)**2) #cm^2
    up_lim_area_cm2 = np.pi*((up_size_cm/2.)**2)
    low_Mdust_g = lower_lim_area_cm2*dust_sdens # g
    low_Mdust_Msun = (low_Mdust_g)*(1*u.g).to(u.Msun).value # Msun
    up_Mdust_g = up_lim_area_cm2*dust_sdens # g
    up_Mdust_Msun = (up_Mdust_g).to(u.Msun).value # Msun
    return tau350, low_Mdust_Msun, up_Mdust_Msun

def gas_mass_from_dust_leroy_v2(I350_Jybeam, bmin_arsec, bmaj_arsec,
                             Tdust, freq_GHz, d_Mpc, x_FWHM_arcsec = 0.5, y_FWHM_arcsec = 0.024, k_nu_cm2_g=1.9):
    """
    Estimated gas mass from continuum emission at 350GHz from Leroy 2019
    k_nu_cm2_g = 1.9 #cm^2/g
    Berta, Lutz, Genzel et al. 2016: Measures of galaxy dust...
    """
    beam_size_arsec2 =utiles_alma. beam_size(bmin_arsec, bmaj_arsec)
    beam_size_sr = (beam_size_arsec2*u.arcsec**2).to(u.sr)
    I350_Jysr = I350_Jybeam / beam_size_sr.value
    I350 = I350_Jysr *1E-23# I350_Jysr.value *1E-23 # erg/Hz/s/sr/cm^2
    Bnu_350 = blackbody_nu(freq_GHz*u.GHz, Tdust).value # erg/Hz/s/sr/cm^2
    tau350 = -np.log(1.-(I350/Bnu_350))
    DGR = 1/100.
    dust_sdens = tau350/(k_nu_cm2_g*DGR)*(u.g/u.cm**2)
    #dust_sdens_g_pc2 = dust_sdens*((u.g/u.cm**2).to(u.Msun/u.pc**2))
    up_lim_area_cm2 = utiles_alma.beam_size(u_conversion.lin_size(d_Mpc, x_FWHM_arcsec), u_conversion.lin_size(d_Mpc, y_FWHM_arcsec)).to(u.cm**2)
    up_Mdust_g = up_lim_area_cm2*dust_sdens # g
    up_Mdust_Msun = (up_Mdust_g).to(u.Msun).value # Msun
    return tau350, up_Mdust_Msun

def gas_mass_from_dust_leroy_integ_v2(I350_Jybeam, bmin_arsec, bmaj_arsec,
                             Tdust, freq_GHz, d_Mpc, x_FWHM_arcsec = 0.5, y_FWHM_arcsec = 0.024, k_nu_cm2_g=1.9):
    """
    Estimated gas mass from continuum emission at 350GHz from Leroy 2019
    k_nu_cm2_g = 1.9 #cm^2/g
    Berta, Lutz, Genzel et al. 2016: Measures of galaxy dust...
    """
    # Convert 350GHz intensity at the peak, to a dist optical depth
    #I350 = (1.-np.exp(-tau350))*B_nu(Tdust)
    beam_size_arsec2 = utiles_alma.beam_size(bmin_arsec, bmaj_arsec)
    beam_size_sr = (beam_size_arsec2*u.arcsec**2).to(u.sr)
    I350_Jysr = I350_Jybeam / beam_size_sr.value
    I350 = I350_Jysr *1E-23# I350_Jysr.value *1E-23 # erg/Hz/s/sr/cm^2
    Bnu_350 = blackbody_nu(freq_GHz*u.GHz, Tdust).value # erg/Hz/s/sr/cm^2
    tau350 = -np.log(1.-(I350/Bnu_350))
    DGR = 1/100.
    dust_sdens = tau350/(k_nu_cm2_g*DGR)*(u.g/u.cm**2)
    #dust_sdens_g_pc2 = dust_sdens*((u.g/u.cm**2).to(u.Msun/u.pc**2))
    utiles_alma.beam_size(x_FWHM_arcsec, y_FWHM_arcsec)
    up_lim_area_cm2 = utiles_alma.beam_size(u_conversion.lin_size(d_Mpc, x_FWHM_arcsec), u_conversion.lin_size(d_Mpc, y_FWHM_arcsec)).to(u.cm**2)
    up_Mdust_g = up_lim_area_cm2*dust_sdens # g
    up_Mdust_Msun = (up_Mdust_g).to(u.Msun).value # Msun
    return tau350, up_Mdust_Msun

def gas_mass_from_tau_leroy(tau350, bmin_arsec, bmaj_arsec,
                             Tdust, freq_GHz, d_Mpc, upper_lim_size_arcsec = 0.5, lower_lim_size_arcsec = 0.024, k_nu_cm2_g=1.9):
    """
    
    Estimated gas mass from continuum emission at 350GHz from Leroy 2019
    k_nu_cm2_g = 1.9 #cm^2/g
    Berta, Lutz, Genzel et al. 2016: Measures of galaxy dust...
    """
    # Convert 350GHz intensity at the peak, to a dist optical depth
    DGR = 1/100.
    dust_sdens = tau350/(k_nu_cm2_g*u.cm**2*DGR/u.g) # g/cm^2
    #dust_sdens_g_pc2 = dust_sdens.to(u.Msun/u.pc**2)
    up_size_cm = (u_conversion.lin_size(d_Mpc, upper_lim_size_arcsec, D_err = 0)).to(u.cm)
    low_size_cm  = (u_conversion.lin_size(d_Mpc, lower_lim_size_arcsec, D_err = 0)).to(u.cm) # 1" = 16.9pc
    lower_lim_area_cm2 = np.pi*((low_size_cm/2.)**2) #cm^2
    up_lim_area_cm2 = np.pi*((up_size_cm/2.)**2)
    low_Mdust_g = lower_lim_area_cm2*dust_sdens # g
    low_Mdust_Msun = (low_Mdust_g).to(u.Msun) # Msun
    up_Mdust_g = up_lim_area_cm2*dust_sdens # g
    up_Mdust_Msun = (up_Mdust_g).to(u.Msun) # Msun
    return tau350, low_Mdust_Msun, up_Mdust_Msun,

def spectralindex_func(freq, slope, intercept):
    """
    spectral index function
    slope == spectral index
    intens = freq**slope
    log(intens) = slop*log(freq)+intercept
    """
    intensity = slope*np.log10(freq)+intercept
    return intensity

def log_spectralindex_func(log_freq, slope, intercept):
    """
    spectral index function log version
    freq is in logarithmic
    slope == spectral index
    intens = freq**slope
    log(intens) = slop*log(freq)+intercept
    """
    log_intensity = slope*log_freq+intercept
    return log_intensity
    
def SFR_from_LIR_Kennicutt(Lir_Lsun):
    """
    Kenicutt 1998
    Salpeter 1955 IMF
    4.5E-44 * L_FIR (erg s-1)
    """
    SFR = (4.5E-44)*u_conversion.lsun_to_ergs(Lir_Lsun).value
    return SFR

def SFR_from_LIR_Hayward(Lir_Lsun):
    """
    Hayward 2014
    Same as Kennicutt but with a Kroupa 2001 IMF
    3E-37 * L_FIR (W)
    """
    SFR = 3E-37*u_conversion.lsun_to_watt(Lir_Lsun).value
    return SFR
    
def specific_SFR(SFR, Mstar):
    """
    Hayward 2014
    SSSFR = SFR/M*
    SSFR determines whether older stellar populations contribute significantly to the IR Luminosity
    actively star-forming : high SSFR and young stars dominate the dust heating
    not that active: lower SSFT, older stellar populations are significant
    """
    SSFR = SFR/Mstar
    return SSFR

def mass_from_nh2_number_dens(nh2_cm3, rad_pc):
    """
    Get mass in msun from number density (cm-3)
    """
    nh2cm3 = nh2_cm3#*u.cm**-3
    atomic_hydrogen_mass_g = 1.6737236E-24 #* u.g
    rad_cm = rad_pc*(1. * u.pc).to(u.cm).value
    volume_cm3 = (4./3)*np.pi*(rad_cm**3)
    dens_mass_cm3 = nh2cm3*2.*atomic_hydrogen_mass_g
    mass_msun = (dens_mass_cm3*volume_cm3)*(1.*u.g).to(u.Msun).value
    return mass_msun

def nh2_number_dens_from_mass(mass_msun, rad_pc):
    """
    Get mass in msun from number density (cm-3)
    """
    atomic_hydrogen_mass_g = 1.6737236E-24 #* u.g
    rad_cm = rad_pc*(1. * u.pc).to(u.cm).value
    volume_cm3 = (4./3)*np.pi*(rad_cm**3)
    mass_g = mass_msun*(1.*u.Msun).to(u.g).value
    dens_mass_cm3 = mass_g/volume_cm3
    dens_cm3 = dens_mass_cm3/(2.*atomic_hydrogen_mass_g)
    return dens_cm3

def mass_from_NH2_col_dens(NH2_cm2, rad_pc):
    """
    Get mass in msun from column density (cm-2)
    """
    #nh2cm3 = nh2_cm3#*u.cm**-3
    atomic_hydrogen_mass_g = 1.6737236E-24 #* u.g
    rad_cm = rad_pc*(1. * u.pc).to(u.cm).value
    surface_cm2 = np.pi*(rad_cm**2)
    col_mass_cm2 = NH2_cm2*2.*atomic_hydrogen_mass_g
    mass_msun = (col_mass_cm2*surface_cm2)*(1.*u.g).to(u.Msun).value
    return mass_msun
    
def virial_mass_leroy(FWHM, vel_disp):
    """
    Calculates the Vyrial Theorem based Dynamical mass 
    FWHM is the deconvolved size of the source in pc
    vel_disp is the velocity dispersion in km/s
    assuming dens prof ~r^-2
    http://adsabs.harvard.edu/abs/2018arXiv180402083L
    Leroy et al 2018
    """
    M = 892. * FWHM * (vel_disp**2.)
    return M

def virial_mass_mccrady(rad_pc, vel_disp):
    """
    McCrady2007
    Calculates the Vyrial Theorem based  mass 
    assuming a virial parameter of 10
    """
    viral_parameter = 10.
    Mvir = (viral_parameter*(rad_pc*((1*u.pc).to(u.m)))*((vel_disp*((1*u.km/u.s).to(u.m/u.s)))**2)/_si.G)*((1*u.kg).to(u.Msun))
    return Mvir

def virial_mass(size_pc, FHWHM_kms):
    """
    Calculates the Vyrial Theorem based Dynamical mass 
    """
    # M = 2*r(m)*(v(m/s)**2)/G(m^3/kg/s^2) = cte * 2*r(pc)*v(km/s)^2
    cte_vir = (((1*u.pc).to(u.m) * (1000.*u.m/u.s)**2)/_si.G).to(u.Msun) 
    Mvir = cte_vir *size_pc*(FHWHM_kms**2.)
    return Mvir

def virial_parameter(rad_pc, FHWHM_kms, M_Msun):
    """
    Matzner and jumper 2015
    
    """
    rad_m = rad_pc*(1.*u.pc).to(u.m)
    Mass_kg = M_Msun*(1.*u.Msun).to(u.kg)
    vir_par = (5.*((FHWHM_kms*1000)**2)*rad_m)/(_si.G*Mass_kg)
    return vir_par                        

def escape_velocity(Mass_msun, rad_pc):
    """
     Minimum speed needed for a free object to escape
     from the gravitational influence of a massive body
    """
    Mass_kg = Mass_msun*(1.*u.Msun).to(u.kg)
    rad_m = rad_pc*(1.*u.pc).to(u.m)
    v_esp = np.sqrt(2*_si.G*Mass_kg/rad_m)/1000. #km
    return v_esp

def radial_momentum(Massgas_msun, Massstars_msun, v_disp):
    """
    Leroy 2018
    pr is the max. radial momentum compatible with an observed  vel disp and a gas mass
    Normalisation by M* allows comparison with input from SNE and stellar winds, which both scale
    with stellar mass
    """
    pr_Mstars = np.sqrt(3.)*v_disp*Massgas_msun/Massstars_msun# km/s
    return pr_Mstars 

def cloud_external_pressure(Mass, v_disp_kms, rad_pc):
    """
    Elmegreen 1989 and Jhonson2015
    PI -> ne=PI<n_e> 
    they adopt PI = 0.5
    Pe units are K cm-3
    """
    kb = _si.k_B.to(u.kg * u.cm**2 / (u.s**2 * u.K)) # boltzmann constant
    Mass_kg = Mass *(1 * u.Msun).to(u.kg)
    rad_cm = rad_pc * (1. * u.pc).to(u.cm)
    v_disp_cm = v_disp_kms*(1. * u.km/u.s).to(u.cm/u.s)
    PI=0.5 
    Pe = 3.*PI*Mass_kg*(v_disp_cm**2)/(4.*np.pi*(rad_cm**3))/kb # K cm**-3
    return Pe

def freefall_time(Mtotal_msun, rad_pc):
    """
    Free fall time given a total mass and source size assuming spherical symmetry
    """
    vol_m3 = (4./3)*np.pi*((rad_pc*(1 * u.pc).to(u.m))**3)
    M_kg = Mtotal_msun*(1 * u.Msun).to(u.kg)
    t_ff = np.sqrt(3.*np.pi/(32.*_si.G.value*(M_kg/vol_m3)))*(1 *u.s).to(u.yr)
    return t_ff # years

def crossing_time(r_pc, vel_dispersion_kms):
    """
    McCrady 2007
    Crossing time: Typical time required for a star to cross the cluster.
    Age of cluster 
    if cluster age ~ 5-10 x tcr , after a few crossing times the stars are well mixed
    and the virial theorem is well satisfied (i.e. we can assume that they are self graviting)
    """
    r_km = r_pc*((1 *u.pc).to(u.km).value)
    tcr = (r_km/vel_dispersion_kms)  *(1 *u.s).to(u.yr)
    return tcr # years 

def density_Msun_pc3(Mtotal_msun, rad_pc):
    vol_pc3 = (4./3)*np.pi*(rad_pc**3)
    dens = Mtotal_msun/vol_pc3
    return dens

def SED_model(nu, Tdust, Mdust, phi, D):
    """
    SED fit from Perez-Beaupuits 2018 
    phi = beam area filling factor
    Bnu = planck function
    angular_size = source solid angle
    T = dust temperature
    Mdust = dust mass
    D = distance to the source
    phi_cold = filling factor of the coldest component
    kd = absorption coefficient
    nu = freq in GHz
    beta = 2
    """
    D = D * u.Mpc
    nu = u.Quantity(nu, u.GHz)
    Tdust = Tdust * u.K
    Mdust = Mdust * u.Msun
    angular_size_arcsec = 17.3*9.2*(u.arcsec**2) #arcsec**2
    angular_size = angular_size_arcsec.to(u.sr)
    d_m = D.to(u.m)
    Mdust_kg = Mdust.to(u.kg)
    Tcmb = 2.73 # K
    phi_cold = 5.0e-1
    beta = 2
    kd =(u.m**2 / u.kg)* 0.04*(nu/(250.*u.GHz))**beta # m^2 / kg
    tau = kd*Mdust_kg/(angular_size.value*phi_cold*d_m**2)
    Bnu_T = blackbody_nu(nu, Tdust)
    Bnu_Tcmb = blackbody_nu(nu, Tcmb)
    Snu = (1. -np.exp(-tau))*(Bnu_T-Bnu_Tcmb)*angular_size*phi
    Snu_jy = Snu.to(u.Jy)
    return Snu_jy

def Backwarming_Luminosity(NH2, Lobs, grain):
    """
    Rowan Robinson 1982
    Ivezic & Elitzur 1997
    The dust temperature scales with the radius like Td(r) = Td,i * (r_i/r)^2
    Lapp = 4pi sigma r^2 Td(r)^4
    NH2 = 2.21E21 Av
    Av  = 1.086 tauv
    Lreal = Lobs * 4/psi(taunu)
    psi(taunu) = psi0 * ()
    """
    if grain == 'carbon':
        psi0 = 6.2
        m = 1.0
    elif grain == 'silicates':
        psi0 = 2.2
        m = 1.25
    else:
        raise ValueError('Grain not defined')
    psi_taunu = (psi0+0.005*(NH2/(1.086*2.21E21))**m)
    factor = 4.  / psi_taunu
    Lreal = Lobs * factor
    return Lreal, factor


def Greenhouse_profiles(Lsunpc2, NH2, profile, radlist, rtotal):
    """
    GA Greenhouse effect temperature profile for SB
    radlist in cm
    r^-q
    log10(Tdust) = A r_n^{alpha} e^{-beta * r_n)*1/(1+b*r_n^gamma)
    """
    #script_dir = os.path.dirname(os.path.realpath(__file__))
    profiles_df = pd.read_csv('data/greenhouse/SB_Tprofiles.txt',
                              delim_whitespace= True, header=0, comment='#')
    # Converting units                        
    profiles_df['LIR_Lsunpc2'] = profiles_df['LIR_1e7_Lsunpc2']*1e7
    profiles_df['alpha'] = profiles_df['alpha_1e-2']*1e-2
    profiles_df['beta'] = profiles_df['beta_1e-2']*1e-2
    profiles_df['b'] = profiles_df['b_1e-3']*1e-3
    profiles_df['NH2'] = profiles_df['NH2_1e24']*1e24
    # Selecting parameters
    lum_condition       = profiles_df['LIR_Lsunpc2'] == Lsunpc2 
    densprof_condition  = profiles_df['q'] == profile 
    NH2_condition       = (profiles_df['NH2'] / NH2 >0.98) & (profiles_df['NH2'] / NH2 < 1.02)
    selected_profile    = profiles_df.loc[(lum_condition) & (densprof_condition) & (NH2_condition)]
    selected_profile.reset_index(drop=True, inplace=True)
    # Checking parameters
    if len(selected_profile) != 1:
        raise InputError('Number of selected profiles ({0:1d}) is not 1'.format(len(selected_profile)))
    A       = selected_profile.iloc[0]['A']
    alpha   = selected_profile.iloc[0]['alpha']
    beta    = selected_profile.iloc[0]['beta']
    b       = selected_profile.iloc[0]['b']
    gamma   = selected_profile.iloc[0]['gamma']
    templist = []
    for rad in radlist:
        rn = rad/rtotal
        # First shell has inner rad = 0
        if rad == 0:
            templist.append(0)
        else:
            logtdust = A * (rn**alpha)*np.exp(-beta*rn)/(1.+b*(rn**gamma))
            templist.append(10**logtdust)
    # Correcting first shell temperature if first inner rad = 0
    if templist[0] == 0:
        templist[0] = templist[1]
    return templist

def densprofile(NH2, radios, profile, rnube_tot, convert_from_EDr1=False):
    """
    dens_0 is the mean final density given a certain profile
    """
    radios = [0.00000E+00, 1.63376E+17, 3.37892E+17 , 5.12407E+17 , 6.86912E+17 , 8.61459E+17 ,
                          1.03595E+18, 1.21045E+18 , 1.38499E+18 , 1.55949E+18 , 1.73404E+18 , 1.90853E+18,
                          2.08308E+18 , 2.14311E+18, 2.20315E+18, 2.26324E+18, 2.32327E+18, 2.38336E+18,
                          2.44339E+18, 2.50348E+18, 2.56352E+18, 2.62355E+18, 2.68364E+18, 2.74368E+18,
                          2.74791E+18, 2.75209E+18, 2.75633E+18, 2.76056E+18, 2.76475E+18, 2.76898E+18, 
                          2.77316E+18]
    # Densities list from reference model
    v6_model_ED_nH = [1.88889E+02, 1.40824E+07, 8.30186E+06, 5.88597E+06, 4.55902E+06, 3.72036E+06,
                       3.14236E+06, 2.71981E+06, 2.39738E+06, 2.14332E+06, 1.93800E+06, 1.76851E+06,
                       1.67034E+06, 1.62418E+06, 1.58049E+06, 1.53910E+06, 1.49983E+06, 1.46251E+06, 
                       1.42700E+06, 1.39317E+06, 1.36091E+06, 1.33010E+06, 1.30067E+06, 1.28545E+06, 
                       1.28346E+06, 1.28152E+06, 1.27955E+06, 1.27761E+06, 1.27566E+06, 1.27372E+06, 
                       1.27179E+06]
    # NH2 reference column density
    from ferpy import radtransf_model_calculations as m_calc
    NH2_ed, Ndust_tot   = m_calc.Coldens_from_Abund(abundance=1E-2, nh2_list=v6_model_ED_nH,
                                                                         rnube_list=radios, rnube_tot=rnube_tot,
                                                                         lists=False)
    rad_norm = np.array(radios)/rnube_tot
    profile = 1.0 # r**-profile
    if convert_from_EDr1 == True:
        Nfactor = NH2/NH2_ed
        ff_dens_direct = np.array(v6_model_ED_nH)*Nfactor
        # keep empty layer at 1e2 cm-3
        ff_dens_direct[0] = 1e2
        NH2_direct, Ndust_tot   = m_calc.Coldens_from_Abund(abundance=1E-2, nh2_list=ff_dens_direct,
                                                                         rnube_list=radios, rnube_tot=rnube_tot,
                                                                         lists=False)
        return ff_dens_direct
    else:
        
        dens_0 = NH2/rnube_tot
        dens = [] # Density profile (1/r)**profile
        vols = [] # Volume of each shell
        v_tot = (4.*np.pi*(rnube_tot**3)/3.)
        if radios[0] != 0:
            for h, rad in enumerate(radios):
                if h < (len(radios)-1):
                    dens.append(dens_0 * (radios[0]/radios[h])**(profile))
                    vols.append(4.*np.pi*((radios[h+1]**3) - (radios[h]**3))/3.)
                else:
                    dens.append(dens_0 * (radios[0]/radios[h])**(profile))
                    vols.append(4.*np.pi*((rnube_tot**3) - (radios[h]**3))/3.)
        else:
            for h, rad in enumerate(radios):
                if h < (len(radios)-1):
                    if radios[h] == 0:
                        dens.append(0)
                        vols.append(0)
                    else:
                        dens.append(dens_0 * (radios[1]/radios[h])**(profile))
                        vols.append(4.*np.pi*((radios[h+1]**3) - (radios[h]**3))/3.)
                else:
                    dens.append(dens_0 * (radios[1]/radios[h])**(profile))
                    vols.append(4.*np.pi*((rnube_tot**3) - (radios[h]**3))/3.)
     
            #print(dens)
        # Mean density (weighting each shell by its volume)
        dens_mean = np.sum(np.array(dens)*np.array(vols))/v_tot
        # Factor to convert current mean dens to desired dens
        ff = dens_0/dens_mean
        # Final densities to give the desired mean density
        ff_dens = np.array(dens)*ff
        # Cheching mean density
        ff_dens_mean = np.sum(np.array(ff_dens)*np.array(vols))/v_tot
        
        if profile == 0: 
            # Just to keep dens of each shell to be dens_0
            # but actually mean dens is  slightlty <dens_0 
            # (dens dep with r is huge (v=r^-3))
            ff_dens = dens
        # Checking total column    
        NH2_tot, Ndust_tot   = m_calc.Coldens_from_Abund(abundance=1E-2, nh2_list=ff_dens,
                                                                         rnube_list=radios, rnube_tot=rnube_tot,
                                                                         lists=False)
        NH2_factor2 = NH2/NH2_tot
        dens_array = ff_dens*NH2_factor2
        # keep empty layer at 1e2 cm-3
        dens_array[0] =1e2
        NH2_tot_real, Ndust_tot   = m_calc.Coldens_from_Abund(abundance=1E-2, nh2_list=dens_array,
                                                                         rnube_list=radios, rnube_tot=rnube_tot,
                                                                         lists=False)
        # Checking profile
        plot_chek = False
        if plot_chek:
            ff_dens_mean = np.sum(np.array(dens_array)*np.array(vols))/v_tot
            mass_tot = ff_dens_mean * v_tot
            fig, ax= plt.subplots()
            ax.plot(rad_norm, v6_model_ED_nH, color='red')
            ax.plot(rad_norm, ff_dens_direct, color='blue')
            ax.plot(rad_norm, dens_array, color='green')
            # Drawing a pure "profile"
            xrad = np.linspace(0,1,1000)
            if profile == 1.0:
                fact = 1e6
            elif profile == 2.0:
                fact = 1e5
            fact = v6_model_ED_nH[-1]
            fact2= dens_array[-1]
            ydens = fact*xrad**(-profile)
            ydens2 = fact2*xrad**(-profile)
            ax.plot(xrad, ydens, color='k', linestyle='--')
            ax.plot(xrad, ydens2, color='k', linestyle='--')
            ax.set_yscale("log", nonposy='clip')
            
            ax.set_ylim([1e6,3e7])
            plt.show()
        return dens_array
    