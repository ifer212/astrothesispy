import os
import subprocess
from shutil import copyfile

import numpy as np
import pandas as pd
import astropy.units as u
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import ScalarFormatter

from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_plot
from astrothesispy.utiles import u_conversion

def densprofile(total_mass, radios, profile, rnube_tot):
        """ 
            Density profile of the models
            dens_0 is the mean final density given a certain profile
        """
        atomic_hydrogen_mass_kg = 1.6737236E-27*u.kg 
        dens_0 = total_mass/(2.*atomic_hydrogen_mass_kg.to(u.Msun).value)/(4/3*np.pi*rnube_tot**3)
        
        dens = [] # Density profile (1/r)**profile
        vols = [] # Volume of each shell
        v_tot = (4.*np.pi*(rnube_tot**3)/3.)
        if radios[0] != 0:
            for h, rad in enumerate(radios):
                if h < (len(radios)-1):
                    dens.append(dens_0 * (radios[0]/radios[h])**(profile))
                    vols.append(4.*np.pi*((radios[h+1]**3) - (radios[h]**3))/3.)
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
     
        # Mean density (weighting each shell by its volume)
        dens_mean = np.sum(np.array(dens)*np.array(vols))/v_tot
        # Factor to convert current mean dens to desired dens
        ff = dens_0/dens_mean
        # Final densities to give the desired mean density
        ff_dens = np.array(dens)*ff
        # Cheching mean density
        ff_dens_mean = np.sum(np.array(ff_dens)*np.array(vols))/v_tot
        mass_tot = ff_dens_mean * v_tot *2.*atomic_hydrogen_mass_kg.to(u.Msun).value
        if profile == 0: 
            # Just to keep dens of each shell to be dens_0
            # but actually mean dens is  slightlty <dens_0 
            # (dens dep with r is huge (v=r^-3))
            ff_dens = dens
        return ff_dens, ff_dens_mean, mass_tot

def get_tau100(model_path, model_name):
    """
        Gets tau100 from the molecule models
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at 100 um
    tau100 = model_taudust['taudust'].loc[model_taudust['lambda_um_u'] == 100.0]
    return tau100.values[0]

def get_tau100_dustmod(model_path, model_name):
    """
        Gets tau100 from the dust models
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.tau', delim_whitespace= True, header=None)
    model_taudust.columns = ['lambda_um', 'taudust', 'otro']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at 100 um
    tau100 = model_taudust.iloc[(model_taudust['lambda_um']-100.0).abs().argsort()[:1]]
    tau100.reset_index(inplace=True, drop=True)
    return tau100.values[0]

def get_taulambda(model_path, model_name, lambda_um):
    """
        Gets tau for the specified wavelength lambda in microns
    """
    # Optical Depths
    model_taudust = pd.read_csv(model_path+'/'+model_name+'_.taudust', delim_whitespace= True)
    model_taudust.columns = ['lambda_um', 'taudust']
    # Rounding lambda
    model_taudust['lambda_um_u']    = model_taudust['lambda_um'].apply(lambda x: utiles.rounding_exp(x, 3))
    # Finding tau  at specified wavelength in um
    taulambda = model_taudust['taudust'].loc[np.abs(model_taudust['lambda_um']-lambda_um)  < 0.1]
    # Mean value if more than one tau is found
    tau = taulambda.mean()
    return tau

def lum_from_dustmod(dustmod, my_model_path, distance_pc, r_out_pc):
    """
        Gets the luminosity from dust models
    """
    dust_inp = my_model_path+dustmod+'.inp'
    with open(dust_inp, 'r') as file:
        modelodust = file.readlines()
        file.close()
    orig_dist_pc = np.float(modelodust[16].split(' ')[6])
    factor = distance_pc/orig_dist_pc
    dust_pro = my_model_path+dustmod+'_.pro'
    df_dust = pd.read_csv(dust_pro, header=None, delim_whitespace= True)
    cols = ['lambda_um', 'I_Ic_ergscm2um', 'Ic_ergscm2um', 'Inorm_ergscm2um', 'I_ergscm2um']
    dif_len = len(df_dust.columns)-len(cols)
    string = 'abcdefgh'
    for i in range(dif_len):
        cols.append('col_'+string[i])
    df_dust.columns = cols
    luminosity_dust = 0
    for i, row in df_dust.iterrows():
        if i == 0:
            continue
        else:
            if row['lambda_um'] <= 1200 and row['lambda_um'] >= 10 :
                luminosity_dust += (df_dust['Ic_ergscm2um'][i]*
                                  (df_dust['lambda_um'][i]-df_dust['lambda_um'][i-1])*
                                  4.*3.14159*(orig_dist_pc*(1*u.pc).to(u.cm).value)**2)/(3.8e33)
    print(f'{luminosity_dust:1.2e} \t {(luminosity_dust)*(factor**2)*(r_out_pc**2):1.2e}')
    luminosity_dust_out = (luminosity_dust)*(factor**2)*(r_out_pc**2)
    return luminosity_dust_out

def model_summary(model_name, dustmod, my_model_path, distance_pc, Rcrit): 
    """
        Returns a dict with a summary of the model vars and properties
    """
    atomic_hydrogen_mass_kg = 1.6737236E-27*u.kg
    lum_to_mass = 1000 
    gas_to_dust = 100
    SHC13_MZAMS_Msun = 0.63E5 # From Leroy2018
    vel_disp = 31#utiles.fwhm_to_stdev(31,0)[0] # km/s FWHM o sigma
    sigma_disp = utiles.fwhm_to_stdev(vel_disp,0)[0]
    d_mpc = distance_pc/1e6
    kappa_mean = 10 # cm2 g-1
    
    modelom = my_model_path+model_name+'.inp'
    qprof = np.float(modelom.split('q')[-1].split('nsh')[0])
    if qprof == 1.0:
        ctedust = 2.9
    elif qprof == 1.5:
        ctedust = 1.7
    with open(modelom, 'r') as file:
        modelo = file.readlines()
        file.close()
    rout_model_hc3n_cm = np.float(list(filter(None, modelo[11].split(' ')))[1])
    rout_model_hc3n_pc = rout_model_hc3n_cm*(1*u.cm).to(u.pc).value
    Omega_arcsec2 = np.pi*rout_model_hc3n_pc**2/(distance_pc**2)*(u.sr).to(u.arcsec**2)
    sigma = np.float(model_name.split('sig')[-1].split('cd')[0])
    luminosity = sigma*np.pi*(rout_model_hc3n_pc**2)
    # Luminosity
    lum_total_model = lum_from_dustmod(dustmod, my_model_path, distance_pc, rout_model_hc3n_pc)
    lum_total_name  = luminosity
    r_profile = []
    rpc_profile = []
    nh2_profile = []
    x_profile = []
    td_profile = []
    for line in modelo[33:64]:
        a = list(filter(None, line.split(' ')))
        r_profile.append(np.float(a[2]))
        rpc_profile.append(np.float(a[2])*(1*u.cm).to(u.pc).value)
        nh2_profile.append(np.float(a[3]))
        x_profile.append(np.float(a[4]))
        td_profile.append(np.float(a[6].strip()))
    r_profile.append(rout_model_hc3n_cm)
    xd_profile = []
    r_profile_pc = np.array(r_profile)*(1*u.cm).to(u.pc).value
    nshells = np.int(list(filter(None, modelo[11].split(' ')))[0])
    for line in modelo[64:64+nshells]:
        a = list(filter(None, line.split(' ')))
        xd_profile.append(np.float(a[1]))
    NH2_profile = []
    MH2_profile = []
    vol_profile = []
    sup_profile = []
    Mdust_profile = []
    MH2_profile_corr = []
    rdif_profile = []
    sigma_profile = [] # Sigma dens profile g cm-3
    nh2_mass_profile = []
    nh2_mass_profile_corr = []
    for r,rad in enumerate(r_profile):
        if r < len(r_profile)-1:
            rdif_profile.append(r_profile[r+1]-r_profile[r])
            NH2_profile.append((r_profile[r+1]-r_profile[r])*nh2_profile[r])
            vol_profile.append(4/3*np.pi*(r_profile[r+1]**3-r_profile[r]**3))
            sup_profile.append(np.pi*(r_profile[r+1]**2-r_profile[r]**2))
            MH2_profile.append(vol_profile[r]*nh2_profile[r]*2.*atomic_hydrogen_mass_kg.to(u.Msun).value)
            nh2_mass_profile.append(nh2_profile[r]*2.*atomic_hydrogen_mass_kg.to(u.Msun).value)
            Mdust_profile.append(MH2_profile[r]*xd_profile[r])
            sigma_profile.append(MH2_profile[r]*(1*u.Msun).to(u.g).value/sup_profile[r])
            MH2_profile_corr.append(Mdust_profile[r]/0.01)
            nh2_mass_profile_corr.append(MH2_profile_corr[r]*(1*u.Msun).to(u.g)/vol_profile[r])
    # Correcting gas mass for r>0.5 due to changes in Xd
    # for r<Rcrit we can assume that changes in Xd are only for free-free emission
    nh2_profile_corr = []
    sigma_profile_corr = []
    NH2_profile_corr = []
    Mgas_fromnH2_Msun_corrected = 0
    Mgas_fromnH2_Msun_profile = []
    for r,rad in enumerate(r_profile_pc[1:]):
        if rad>=Rcrit:
            Mgas_fromnH2_Msun_profile.append(MH2_profile_corr[r])
            Mgas_fromnH2_Msun_corrected += MH2_profile_corr[r]
            sigma_profile_corr.append(MH2_profile_corr[r]*(1*u.Msun).to(u.g).value/sup_profile[r])
        else:
            Mgas_fromnH2_Msun_corrected += MH2_profile[r]
            Mgas_fromnH2_Msun_profile.append(MH2_profile[r])
            sigma_profile_corr.append(MH2_profile[r]*(1*u.Msun).to(u.g).value/sup_profile[r])
        nh2_profile_corr.append(Mgas_fromnH2_Msun_profile[r]/(2.*atomic_hydrogen_mass_kg.to(u.Msun).value)/vol_profile[r])
        NH2_profile_corr.append(rdif_profile[r]*nh2_profile_corr[r])
        
    Stars = [] # propto dens*MH2
    Stars_nocorr = [] # propto dens*MH2
    Stars_nocorrv2 = [] # propto dens*MH2
    Stars_nocorr_norm = [] # Lir/dens_sup
    Stars_nocorr_norm2 = [] # dens_sup*dens*M
    for m, mass in enumerate(Mgas_fromnH2_Msun_profile):
        Stars.append(nh2_profile_corr[m]*Mgas_fromnH2_Msun_profile[m])
        Stars_nocorr.append(nh2_profile_corr[m]*MH2_profile[m])
        Stars_nocorrv2.append(nh2_profile[m]*MH2_profile[m])
        Stars_nocorr_norm.append(lum_total_model/sigma_profile[m])
        Stars_nocorr_norm2.append(sigma_profile[m]*nh2_mass_profile[m]*MH2_profile[m]*(1*u.Msun).to(u.g).value)
    Stars_total = np.nansum(Stars)
    st = 0
    all_st = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        all_st += Stars[r]
        if st <= Stars_total/2:
            st += Stars[r]
        else:
            half_stars_rad = r_profile_pc[r]
            break
    Stars_total_nocorr = np.nansum(Stars_nocorr)
    st_nocorr = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        if st_nocorr <= Stars_total_nocorr/2:
            st_nocorr += Stars_nocorr[r]
        else:
            half_stars_rad_nocorr = r_profile_pc[r]
            break
    
    Stars_total_nocorrv2 = np.nansum(Stars_nocorrv2)
    st_nocorrv2 = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        if st_nocorrv2 <= Stars_total_nocorrv2/2:
            st_nocorrv2 += Stars_nocorrv2[r]
        else:
            half_stars_rad_nocorrv2 = r_profile_pc[r]
            break
        
    st_nocorrv2 = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        if st_nocorrv2 <= 0.8*Stars_total_nocorrv2:
            st_nocorrv2 += Stars_nocorrv2[r]
        else:
            trescuartos_stars_rad_nocorrv2 = r_profile_pc[r]
            break
    print(f'{model_name} \t {trescuartos_stars_rad_nocorrv2:1.2f}')
    Stars_total_nocorr_norm = np.nansum(Stars_nocorr_norm[1:])
    st_nocorr_norm = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        if st_nocorr_norm <= Stars_total_nocorr_norm/2:
            st_nocorr_norm += Stars_nocorr_norm[r+1]
        else:
            half_stars_rad_nocorr_norm = r_profile_pc[r]
            break
    Stars_total_nocorr_norm2 = np.nansum(Stars_nocorr_norm2)
    st_nocorr_norm2 = 0
    for r, rad in enumerate(r_profile_pc[1:]):
        if st_nocorr_norm2 <= Stars_total_nocorr_norm2/2:
            st_nocorr_norm2 += Stars_nocorr_norm2[r]
        else:
            half_stars_rad_nocorr_norm2 = r_profile_pc[r]
            break 
    NHC3N_profile = np.array(NH2_profile)*np.array(x_profile)
    NHC3N_profile_corr = np.array(NH2_profile_corr)*np.array(x_profile)
    Nd_profile = np.array(NH2_profile)*np.array(xd_profile)
    Nd_profile_corr = np.array(NH2_profile_corr)**np.array(xd_profile)
    # Col. Densities cm -1
    NH2_total = np.nansum(NH2_profile)
    NH2_total_corr = np.nansum(NH2_profile_corr)
    NHC3N_total = np.nansum(NHC3N_profile)
    NHC3N_total_corr = np.nansum(NHC3N_profile_corr)
    Nd_total = np.nansum(Nd_profile) # From Xdust, bad
    Nd_total_corr = np.nansum(Nd_profile_corr) # From Xdust, bad
    # Mass from NH2  --> Mal
    Mgas_fromNH2_Msun_total = utiles.mass_from_NH2_col_dens(NH2_total, rout_model_hc3n_pc)
    # Mass from density (nH2) --> Esta seria la buena si no hubiera corregido Xd
    Mgas_fromnH2_Msun_total = np.nansum(MH2_profile)
    Mgas_Msun_total = Mgas_fromnH2_Msun_corrected # --> Esta es la buena es la que uso para todo
    # Mdust  (eq.3 from GA19)
    Mdust_Msun_total = ctedust*1e6*(NH2_total/1e25)*((d_mpc/100)**2)*(Omega_arcsec2/1.1e-2)
    # Mass from Mdust --> Mal al haber cambiado Xd en las capas
    Mgas_fromMdust_Msun_total = Mdust_Msun_total*gas_to_dust
    # H2 Density from NH2 column density in cm-3 --> mal
    n_aver_fromNH2 = NH2_total/(rout_model_hc3n_pc*(1*u.pc).to(u.cm).value) 
    # H2 Density from corrected Mgas in cm-3
    volume_cm3 = 4/3*np.pi*(rout_model_hc3n_pc*(1*u.pc).to(u.cm)**3)
    surface_cm2 = np.pi*((rout_model_hc3n_pc*(1*u.pc).to(u.cm))**2)
    n_aver_fromMgas = Mgas_Msun_total*(1*u.Msun).to(u.kg)/(2.*atomic_hydrogen_mass_kg)/volume_cm3
    n_surf_aver_fromMgas = Mgas_Msun_total*(1*u.Msun).to(u.kg)/(2.*atomic_hydrogen_mass_kg)/surface_cm2
    # H2 Density from corrected Mgas in Msun pc-3 
    volume_m3 = 4/3*np.pi*((rout_model_hc3n_pc*(1*u.pc).to(u.m))**3)
    surface_pc2 = np.pi*((rout_model_hc3n_pc*(1*u.pc).to(u.pc))**2)
    total_dens_kg_m3 = Mgas_Msun_total*(1*u.Msun).to(u.kg)/volume_m3
    total_dens_Msun_pc3 = total_dens_kg_m3.to(u.Msun/u.pc**3)
    surf_dens_Msun_pc2 = Mgas_Msun_total*(1*u.Msun)/surface_pc2
    # Opacity 
    tau100   = get_tau100(my_model_path, model_name) # Al variar Xdust esta no vale
    d_tau100 = get_tau100_dustmod(my_model_path, dustmod) # Esta es la buena
    
    model_dict = {'model'      : model_name,
                  'R_pc'       : rout_model_hc3n_pc,
                  'q'          : qprof,
                  'NH2'        : NH2_total, 
                  'NH2_corr'   : NH2_total_corr,   # Columna corregida por cambiar Xd
                  'NHC3N'      : NHC3N_total,
                  'NHC3N_corr' : NHC3N_total_corr, # Columna corregida por cambiar Xd
                  'Nd'         : Nd_total, # Nd de NH2, no es el del modelo de verdad
                  'Nd_corr'    : Nd_total_corr, # Nd de NH2, no es el del modelo de verdad
                  'HC3Ntau100'     : tau100, # Mala, hay que calcularlo en el modelo del polvo
                  'dusttau100'     : d_tau100[1], # Buena, proviene del modelo del polvo
                  'Mgas_Msun'      : Mgas_fromnH2_Msun_total,     # Es la original del modelo (sin corregir por cambiar Xd)
                  'Mgas_Msun_corr' : Mgas_fromnH2_Msun_corrected, # Masa corregida por cambiar Xd)
                  'Ltot_Lsun'      : lum_total_model,      # Lum from dust model
                  'Ltot_name_Lsun' : lum_total_name,       # Lum from model name
                  'nH2_cm3'        : n_aver_fromMgas.value,      # Density in cm-3 after corrcting from Xd
                  'nH2_Msun_pc3'        : total_dens_Msun_pc3.value,      # Density in Msun pc-3 after corrcting from Xd
                  'nH2_kg_m3'          : total_dens_kg_m3.value, 
                  'SigmaH2_cm2'          : n_surf_aver_fromMgas, 
                  'SigmaH2_Msun_pc2'    : surf_dens_Msun_pc2.value,  # Surface density in Msun pc-2 after corrcting from Xd
                  'half_rad_pc'    : half_stars_rad_nocorr, # Wrong it uses the corrected density profile # Radius with half the luminosity (i.e. Lir \propto \rho*M_gas)
                  'half_rad_pcv2'   :half_stars_rad_nocorrv2, # Vals without correction
                  'half_rad_pc_corr'    : half_stars_rad,  # Radius with half the luminosity after correction for Xd(i.e. Lir \propto \rho*M_gas)
                  'half_rad_norm_pc' : half_stars_rad_nocorr_norm,
                  'half_rad_norm2_pc' : half_stars_rad_nocorr_norm2,
                  # Profiles
                  'NHC3N_corr_profile': NHC3N_profile_corr,
                  'Nd_corr_profile': Nd_profile_corr,
                  'NH2_corr_profile': NH2_profile_corr,
                  'rad_profile_pc': r_profile_pc[1:]
                  # half_rad_pc_corr is not the half "SFR" radii because the increased MH2 (from Xd) is in the HC3N model
                      # and not in the dust model, hence the new "mass" is not taken into account for star formation
                  }
    return model_dict

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

def read_model_input(modelo, my_model_path, results_path, Rcrit):
    """
     Reads model parameters from .inp and also de observed values
    """
    atomic_hydrogen_mass_kg = 1.6737236E-27*u.kg
    model_name = modelo[0]
    modelom = my_model_path+model_name+'.inp'
    qprof = modelom.split('q')[-1].split('nsh')[0]
    with open(modelom, 'r') as file:
        modelo = file.readlines()
        file.close()
    rout_model_hc3n_cm = np.float(list(filter(None, modelo[11].split(' ')))[1])
    rout_model_hc3n_pc = rout_model_hc3n_cm*(1*u.cm).to(u.pc).value
    sigma = np.float(model_name.split('sig')[-1].split('cd')[0])
    luminosity = sigma*np.pi*(rout_model_hc3n_pc**2)
    r_profile = []
    rpc_profile = []
    nh2_profile = []
    x_profile = []
    td_profile = []
    nshells = np.int(list(filter(None, modelo[11].split(' ')))[0])
    for line in modelo[33:64]:
        a = list(filter(None, line.split(' ')))
        r_profile.append(np.float(a[2]))
        rpc_profile.append(np.float(a[2])*(1*u.cm).to(u.pc).value)
        nh2_profile.append(np.float(a[3]))
        x_profile.append(np.float(a[4]))
        td_profile.append(np.float(a[6].strip()))
    r_profile.append(rout_model_hc3n_cm)
    r_profile_pc = np.array(r_profile)*(1*u.cm).to(u.pc).value
    xd_profile = []
    for line in modelo[64:64+nshells]:
        a = list(filter(None, line.split(' ')))
        xd_profile.append(np.float(a[1]))
    NH2_profile = []
    MH2_profile = []
    vol_profile = []
    Mdust_profile = []
    MH2_profile_corr = []
    rdif_profile = []
    for r,rad in enumerate(r_profile):
        if r < len(r_profile)-1:
            rdif_profile.append(r_profile[r+1]-r_profile[r])
            NH2_profile.append((r_profile[r+1]-r_profile[r])*nh2_profile[r])
            vol_profile.append(4/3*np.pi*(r_profile[r+1]**3-r_profile[r]**3))
            MH2_profile.append(vol_profile[r]*nh2_profile[r]*2.*atomic_hydrogen_mass_kg.to(u.Msun).value)
            Mdust_profile.append(MH2_profile[r]*xd_profile[r])
            MH2_profile_corr.append(Mdust_profile[r]/0.01)
    # Correcting dens and col denst with new mass correction
    # for r<Rcrit we can assume that changes in Xd are only for free-free emission
    nh2_profile_corr = []
    NH2_profile_corr = []
    Mgas_fromnH2_Msun_corrected = 0
    Mgas_fromnH2_Msun_profile_corr = []
    for r,rad in enumerate(r_profile_pc[1:]):
        if rad>=Rcrit:
            Mgas_fromnH2_Msun_profile_corr.append(MH2_profile_corr[r])
            Mgas_fromnH2_Msun_corrected += MH2_profile_corr[r]
        else:
            Mgas_fromnH2_Msun_corrected += MH2_profile[r]
            Mgas_fromnH2_Msun_profile_corr.append(MH2_profile[r])
        nh2_profile_corr.append(Mgas_fromnH2_Msun_profile_corr[r]/(2.*atomic_hydrogen_mass_kg.to(u.Msun).value)/vol_profile[r])
        NH2_profile_corr.append(rdif_profile[r]*nh2_profile_corr[r])
        
    # Column density profiles
    NHC3N_profile = np.array(NH2_profile)*np.array(x_profile)
    NHC3N_profile_corr = np.array(NH2_profile_corr)*np.array(x_profile)
    Nd_profile = np.array(NH2_profile)*np.array(xd_profile)
    Nd_profile_corr = np.array(NH2_profile_corr)**np.array(xd_profile)
    logNHC3N_profile = np.log10(NHC3N_profile)
    logNH2_profile = np.log10(NH2_profile)
    logNHC3N_profile_corr = np.log10(NHC3N_profile_corr)
    logNH2_profile_corr = np.log10(NH2_profile_corr)
    
    obs_df = pd.read_csv(f'{results_path}SHC_13_SLIM_Tex_and_logN_profiles.csv', sep=';')
    obs_df['Dist_mean_cm'] = obs_df['Dist_mean_pc']*(1*u.pc).to(u.cm).value
    poly = np.polyfit(obs_df['Dist_mean_pc'], obs_df['Col_det'], deg=6)
    fit_logNHC3N = np.polyval(poly, rpc_profile)
    fit_XHC3N = (10**fit_logNHC3N)/np.array(NH2_profile)
    return obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof

def plot_model_input(modelo, my_model_path, figinp_path, results_path):
    """
     Plots model input properties
    """
    model_name = modelo[0]
    modelom = my_model_path+model_name+'.inp'
    with open(modelom, 'r') as file:
        modelo = file.readlines()
        file.close()
    rout_model_hc3n_cm = np.float(list(filter(None, modelo[11].split(' ')))[1])
    rout_model_hc3n_pc = rout_model_hc3n_cm*(1*u.cm).to(u.pc).value
    r_profile = []
    rpc_profile = []
    nh2_profile = []
    x_profile = []
    td_profile = []
    for line in modelo[33:64]:
        a = list(filter(None, line.split(' ')))
        r_profile.append(np.float(a[2]))
        rpc_profile.append(np.float(a[2])*(1*u.cm).to(u.pc).value)
        nh2_profile.append(np.float(a[3]))
        x_profile.append(np.float(a[4]))
        td_profile.append(np.float(a[6].strip()))
    r_profile.append(rout_model_hc3n_cm)
    #rpc_profile.append(rout_model_hc3n_pc)
    NH2_profile = []
    for r,rad in enumerate(r_profile):
        if r < len(r_profile)-1:
            NH2_profile.append((r_profile[r+1]-r_profile[r])*nh2_profile[r])
    logNH2_profile = np.log10(NH2_profile)
    NHC3N_profile = np.array(NH2_profile)*np.array(x_profile)
    logNHC3N_profile = np.log10(NHC3N_profile)
    obs_df = pd.read_csv(f'{results_path}SHC_13_SLIM_Tex_and_logN_profiles.csv', sep=';')
    obs_df['dist_ring_cm'] = obs_df['dist_ring_pc']*(1*u.pc).to(u.cm).value
    poly = np.polyfit(obs_df['dist_ring_pc'], obs_df['Col_det'], deg=6)
    fit_logNHC3N = np.polyval(poly, rpc_profile)
    fit_XHC3N = (10**fit_logNHC3N)/np.array(NH2_profile)
    
    figsize = 20
    naxis = 2
    maxis = 2
    labelsize = 18
    ticksize = 16
    fontsize = 14
    fg = plt.figure(figsize=(figsize*1.15, figsize*0.85))
    gs = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    ydict = {'T': [td_profile, r'T$_{dust}$ (K)', 'Tex_det', '', []],
             'dens' : [nh2_profile, r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', '', 'log', []],
             'X': [x_profile, r'$X$ (HC$_{3}$N)', '', 'log', fit_XHC3N],
             'N' : [logNHC3N_profile, r'log N(HC$_{3}$N)  (cm$^{-2}$)', 'Col_det', '', fit_logNHC3N],
            }
    
    axis=[]
    for v,var in enumerate(ydict):
        axis.append(fg.add_subplot(gs[v]))
        axis[v].set_ylabel(ydict[var][1], fontsize = labelsize, labelpad=12)
        if ydict[var][2] != '':
            axis[v].plot(obs_df['dist_ring_pc'], obs_df[ydict[var][2]], marker='o', color='k', linestyle='')
        axis[v].plot(rpc_profile, ydict[var][0], color='r')
        if ydict[var][3] == 'log': 
            axis[v].set_yscale('log')
        if len(ydict[var][4])>0: 
            axis[v].plot(rpc_profile, ydict[var][4], color='b')

        
        minor_locator = AutoMinorLocator(2)
        axis[v].set_xlim([0.0, 1.42])
        
        axis[v].tick_params(direction='in')
        axis[v].tick_params(axis="both", which='major', length=8)
        axis[v].tick_params(axis="both", which='minor', length=4)
        axis[v].xaxis.set_tick_params(which='both', top ='on')
        axis[v].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axis[v].tick_params(axis='both', which='major', labelsize=ticksize)
        axis[v].xaxis.set_minor_locator(minor_locator)
        axis[v].tick_params(labelleft=True,
                       labelright=False)
        if v >1:
            axis[v].set_xlabel(r'r (pc)', fontsize = labelsize)
        else: 
            axis[v].tick_params(
                       labelbottom=False)
    fg.savefig(figinp_path+'NGC253_SHC13_'+model_name+'_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    return fit_XHC3N, x_profile, obs_df['dist_ring_pc'].tolist(), rpc_profile

def line_profiles_chi2(hb_df, line_column, modelos, fort_paths, my_model_path, Rcrit, results_path, D_Mpc = 3.5):
    """
        Estimates de chi2 from the modelled line profiles
    """
    distance_pc = D_Mpc*1e6
    for l,line in enumerate(line_column):
        if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:
            for i,row in hb_df.iterrows():
                if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                    hb_df.loc[i, line+'_uplim'] = True
                else:
                    hb_df.loc[i, line+'_uplim'] = False
                    
    for m, mod in enumerate(modelos):
        modelo = modelos[mod]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        # Modelled emission
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        # Model parameters
        bs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof =  read_model_input(modelo, my_model_path, results_path, Rcrit)
        # Model luminosity
        model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
        lines_chi = 0
        for l,line in enumerate(line_column):
            if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:
                chi2 = line_chi2(hb_df, m_molec, line, beam='_beam_orig')
                print(f'\t{line}\t{chi2:1.2f}')
                lines_chi += chi2
        print(f'Total Chi2 : Sum {lines_chi:1.2f} Mean {lines_chi/11:1.2f}')
        
def cont_difs(Leroy36df_new, cont_df, line_column, modelos, fort_paths, my_model_path, results_path, savefig_path, D_Mpc = 3.5):
    """
     Difference between continuum from dust model and from HC3N model (i.e. free-free)
         Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit'] # Size at 219GHz at the 345GHz resolution
         Leroy36df_new['FWHM_pc']                     # Sizes from 36GHz emission from Leroy2018
         Leroy36df_new['350GHz_deconvFWHM_pc_fit']    # Sizes from 36GHz emission from Leroy2018
         Leroy36df_new['Size_limit']                  # Size of the 111GHz (free-free) emission
         Leroy36df_new['Size_limit_opt_thin']         # Size of the 111GHz (free-free) emission assuming opt thin
    """
    distance_pc = D_Mpc*1e6
    results_path = '/Users/frico/Documents/data/NGC253_HR/Results_v2/'
    modelos = { 'model2': ['m28_LTHC3Nsbsig1.3E+07cd1.0E+25q1.5nsh30rad1.5vt5_b9','dustsblum1.2E+10cd1.0E+25exp1.5nsh1003rad17',
                            1.5, utiles_plot.redpink , [1, 1.0, 1.9]]}
    Rtotal = 1.5 # pc
    FWHM219_deconv_size_pc = 0.5
    FWHM345_deconv_size_pc = 0.8
    sigma219_deconv_size_pc = utiles.fwhm_to_stdev(FWHM219_deconv_size_pc, 0)[0]
    sigma345_deconv_size_pc = utiles.fwhm_to_stdev(FWHM345_deconv_size_pc, 0)[0]
    ring_width_pc = 0.1
    w_ring_pc = ring_width_pc/2
    FWHM = utiles.stdev_to_fwhm(Rtotal/3, 0)[0] # Rtotal = 3sigma
    # Converting FWHM to sigma values
    Leroy36df_new = Leroy36df_new.dropna(subset = ['Source_altern_sub_final'])
    Leroy36df_new['219GHz_SM_deconvsigma_pc_fit'] = utiles.fwhm_to_stdev(Leroy36df_new['219GHz_SM_deconvFWHM_pc_fit'],0)[0]
    Leroy36df_new['36GHz_sigma_pc'] = utiles.fwhm_to_stdev(Leroy36df_new['FWHM_pc'],0)[0]            
    Leroy36df_new['350GHz_SM_deconvsigma_pc_fit'] = utiles.fwhm_to_stdev(Leroy36df_new['350GHz_deconvFWHM_pc_fit'],0)[0]
    Leroy36df_new['Size_sigma_limit_pc'] = utiles.fwhm_to_stdev(Leroy36df_new['Size_limit'],0)[0] 
    Leroy36df_new['Size_sigma_limit_opt_thin'] = utiles.fwhm_to_stdev(Leroy36df_new['Size_limit_opt_thin'],0)[0] 
    # Printing all sigma sizes
    for i,row in Leroy36df_new.iterrows():
        print(f'{row["Source_altern_sub_final"]}   \t {row["36GHz_sigma_pc"]:1.2f}   \t {3*row["219GHz_SM_deconvsigma_pc_fit"]:1.2f}    \t {3*row["350GHz_SM_deconvsigma_pc_fit"]:1.2f}    \t {3*row["Size_sigma_limit_pc"]:1.2f}    \t {3*row["Size_sigma_limit_opt_thin"]:1.2f}')
    # Ratio between 230GHz/345GHz cont emission. For r<Rcrit this ratio should increase and indicate the size of the free-free emission
    cont_df['235/345_ratio'] = cont_df['F235GHz_mjy_beam345']/cont_df['F345GHz_mjy_beam']
    cont_df['px_count'] = [1, 8, 12, 16, 20, 24, 28, 36, 32, 44, 56, 48, 64, 60]
    total_radlist = list(-1*np.array(cont_df['dist'].tolist()[::-1]))+cont_df['dist'].tolist()
    total_cont235list =cont_df['F235GHz_mjy_beam345'].tolist()[::-1]+cont_df['F235GHz_mjy_beam345'].tolist()
    total_cont345list =cont_df['F345GHz_mjy_beam'].tolist()[::-1]+cont_df['F345GHz_mjy_beam'].tolist()
    # Rcrit (min value of the ratio)
    Rcrit_ind = cont_df['235/345_ratio'].idxmin(axis=0, skipna=True)
    plot = True
    if plot:
        fig = plt.figure()
        plt.plot(cont_df['dist'], cont_df['235/345_ratio']**-1)
        fig.savefig(f'{savefig_path}NGC253_SHC13_235_345_ratio.pdf', bbox_inches='tight', transparent=True, dpi=400)

    Rcrit_pc = cont_df['dist'].iloc[Rcrit_ind]
    Rcrit_pc_art = 0.5
    # At 219GHz resolution
    beam_orig = np.pi*0.022*0.020/(4*np.log(2))
    pc2_beam_orig = beam_orig*(distance_pc*np.pi/(180.0*3600.0))**2
    fwhm_beam = u_conversion.lin_size(distance_pc/1e6,0.022).to(u.pc).value
    beam_345  = np.pi*0.028*0.034/(4*np.log(2))
    fwhm_beam345 = u_conversion.lin_size(distance_pc/1e6,0.034).to(u.pc).value
    pc2_beam_345  = beam_345*(distance_pc*np.pi/(180.0*3600.0))**2
    FWHM219_deconv_size_pc2 = np.pi*FWHM219_deconv_size_pc**2/(4*np.log(2))
    FWHM345_deconv_size_pc2 = np.pi*FWHM345_deconv_size_pc**2/(4*np.log(2))
    FWHM219_conv_size_pc = np.sqrt(np.sqrt(FWHM219_deconv_size_pc2**2+pc2_beam_345**2)/np.pi)
    FWHM345_conv_size_pc = np.sqrt(np.sqrt(FWHM345_deconv_size_pc2**2+pc2_beam_345**2)/np.pi)
    FWHM345_conv_size2_pc = np.sqrt(FWHM345_deconv_size_pc**2+fwhm_beam345**2)
    sigma219_conv_size_pc = utiles.fwhm_to_stdev(np.sqrt(FWHM219_deconv_size_pc**2+fwhm_beam345**2), 0)[0]
    sigma345_conv_size_pc = utiles.fwhm_to_stdev(np.sqrt(FWHM345_deconv_size_pc**2+fwhm_beam345**2), 0)[0]
    
    cont_df['F235GHz_mJy'] = (cont_df['F235GHz_mjy_beam']/pc2_beam_orig)*np.pi*((cont_df['dist']+w_ring_pc)**2-(cont_df['dist']-w_ring_pc)**2)
    cont_df['F235GHz_beam345_mJy'] = (cont_df['F235GHz_mjy_beam345']/pc2_beam_345)*np.pi*((cont_df['dist']+w_ring_pc)**2-(cont_df['dist']-w_ring_pc)**2)
    amp_fit = []
    sigma_fit = []
    FWHM_fit = []
    size_fit_pc2 = []
    for c,cont in enumerate([total_cont235list, total_cont345list]):
        n = len(total_radlist)                          #the number of data
        mean = sum(np.array(cont)*np.array(total_radlist))/n     #note this correction
        sigma = sum(np.array(cont)*(np.array(total_radlist)-mean)**2)/n        #note this correction
        def gaus(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
        popt,pcov = curve_fit(gaus,np.array(total_radlist),np.array(cont),p0=[1,mean,sigma])
        amp_fit.append(np.abs(popt[0]))
        sigma_fit.append(np.abs(popt[2]))
        FWHM_fit.append(utiles.stdev_to_fwhm(sigma_fit[c], 0)[0])
        size_fit_pc2.append(np.pi*sigma_fit[c]**2)

    # Observed sizes, they match the original sizes from 2D fit
    deconv_sizefit_pc2 = utiles.deconv_sizes(np.array(size_fit_pc2), pc2_beam_345)
    deconv_sizefit_pc = np.sqrt(deconv_sizefit_pc2)
    size_Rcrit_pc2 = np.pi*(Rcrit_pc+0.05)**2
    deconv_Rcrit_pc = utiles.deconv_sizes(Rcrit_pc+0.05, np.sqrt(pc2_beam_345))
    deconv_FWHM_pc = utiles.deconv_sizes(FWHM, np.sqrt(pc2_beam_345))
    deconv_Rtotal_pc = utiles.deconv_sizes(Rtotal**2, np.sqrt(pc2_beam_345))
    
    
    cubo345_rms = 4e-5
    cubo345_pixlen_pc = 0.0848
    cubo219_pixlen_pc = 0.119
    radprofile345 = pd.read_csv(f'{results_path}SHC13_cont345_profile2.csv')
    radprofile345.columns = ['px', 'jy_beam']
    Rmax = radprofile345['jy_beam'].idxmax(axis=0, skipna=True)
    rmax = (radprofile345['px'].iloc[Rmax]+radprofile345['px'].iloc[Rmax-1])/2
    radprofile345['px_res'] = radprofile345['px']-rmax
    radprofile345['dist_pc'] = radprofile345['px_res']*cubo345_pixlen_pc
    n = len(radprofile345['dist_pc'])                          #the number of data
    mean = sum(radprofile345['jy_beam']*radprofile345['dist_pc'])/n     #note this correction
    sigma = 0.23
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    a = radprofile345['jy_beam'].iloc[Rmax]
    popt,pcov = curve_fit(gaus,radprofile345['dist_pc'],radprofile345['jy_beam'],p0=[a,mean,sigma])
    fig = plt.figure()
    plt.plot(radprofile345['dist_pc'],radprofile345['jy_beam'],color= 'b', linestyle='-',label='data', marker='')
    plt.plot(radprofile345['dist_pc'],gaus(radprofile345['dist_pc'],*popt),color='r',linestyle='--',label='fit', marker='')
    plt.axhline(3*cubo345_rms, color='g')
    plt.ylabel(r'345GHz (mJy beam$^{-1}$)')
    plt.xlabel(r'$r$ (pc)')
    fig.savefig(f'{savefig_path}cont345_radprofile.pdf')
    radprofile345_sub = radprofile345[radprofile345['jy_beam']>=3*cubo345_rms]
    
    cubo345_rms = 4e-5
    cubo345_pixlen_pc = 0.0848
    cubo219_pixlen_pc = 0.119
    radprofilehc3n = pd.read_csv(f'{results_path}SHC13_HC3Nv0_2625_profile2.csv')
    radprofilehc3n.columns = ['px', 'jy_beam']
    Rmax = radprofilehc3n['jy_beam'].idxmax(axis=0, skipna=True)
    rmax = (radprofilehc3n['px'].iloc[Rmax]+radprofilehc3n['px'].iloc[Rmax-1])/2
    radprofilehc3n['px_res'] = radprofilehc3n['px']-rmax
    radprofilehc3n['dist_pc'] = radprofilehc3n['px_res']*cubo219_pixlen_pc
    n = len(radprofilehc3n['dist_pc'])                          #the number of data
    mean = sum(radprofilehc3n['jy_beam']*radprofilehc3n['dist_pc'])/n     #note this correction
    sigma = 0.23#sum(radprofile345['jy_beam']*(radprofile345['dist_pc']-mean)**2)/n        #note this correction
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    a = radprofilehc3n['jy_beam'].iloc[Rmax]
    popt,pcov = curve_fit(gaus,radprofilehc3n['dist_pc'],radprofilehc3n['jy_beam'],p0=[a,mean,sigma])
    plt.plot(radprofilehc3n['dist_pc'],radprofilehc3n['jy_beam'],'b+',label='data')
    plt.plot(radprofilehc3n['dist_pc'],gaus(radprofilehc3n['dist_pc'],*popt),'ro:',label='fit')
    plt.axhline(3*cubo345_rms, color='g')
    radprofilehc3n_sub = radprofilehc3n[radprofilehc3n['jy_beam']>=3*cubo345_rms]
    
    obsflux_upto_Rcrit_pc_art = 0
    obsflux_upto_Rcrit_pc = 0
    obsflux_upto_convsize_pc = 0
    for i,row in cont_df.iterrows():
        if row['dist'] < Rcrit_pc:
            obsflux_upto_Rcrit_pc += row['F235GHz_beam345_mJy']
        if row['dist'] < Rcrit_pc_art:
            obsflux_upto_Rcrit_pc_art += row['F235GHz_beam345_mJy']
        if row['dist'] < FWHM219_conv_size_pc:
            obsflux_upto_convsize_pc += row['F235GHz_beam345_mJy']
            

    for m, mod in enumerate(modelos):
        modelo = modelos[mod]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        # Model parameters
        obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
        # Modelled emission
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE=True, read_only=True)
        # Model luminosity
        model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
        
       
        for i,row in m_molec.iterrows():
            if i<len(m_molec)-1:
                m_molec.loc[i, 'F235GHz_mjy'] = row['F235GHz_mjy_beam']/pc2_beam_orig*np.pi*(m_molec.loc[i+1, 0]**2-m_molec.loc[i, 0]**2)
                m_molec345.loc[i, 'F235GHz_beam345_mjy'] = m_molec345.loc[i, 'F235GHz_mjy_beam345']/pc2_beam_345*np.pi*(m_molec345.loc[i+1, 0]**2-m_molec345.loc[i, 0]**2)
                mdust.loc[i, 'F235GHz_mjy'] = mdust.loc[i, 'F235GHz_mjy_beam']/pc2_beam_orig*np.pi*(mdust.loc[i+1, 0]**2-mdust.loc[i, 0]**2)
                mdust.loc[i, 'F235GHz_beam345_mjy'] = mdust.loc[i, 'F235GHz_mjy_beam345']/pc2_beam_345*np.pi*(mdust.loc[i+1, 0]**2-mdust.loc[i, 0]**2)
        
        # Difference btw dust model and HC3N model cont emission (i.e. free-free)
        m_molec['diff_235GHz_mjy'] = m_molec['F235GHz_mjy'] - mdust['F235GHz_mjy']
        m_molec['diff_235GHz_beam345_mjy'] = m_molec345['F235GHz_beam345_mjy'] - mdust['F235GHz_beam345_mjy']
        diff_235GHz_mjy_sumRcrit = 0
        diff_235GHz_mjy_beam345_sum_Rcrit = 0
        modflux235GHz_upto_Rcrit_pc_beam345 = 0
        diff_235GHz_mjy_sumRcrit_art = 0
        
        diff_235GHz_mjy_beam345_sum_Rcrit_art = 0
        diff_235GHz_mjy_sumconvsize = 0
        diff_235GHz_mjy_beam345_sumconvsize = 0
        modflux235GHz_upto_Rcrit_pc_art_beam345 = 0
        modflux235GHz_upto_Rcrit_pc_art = 0
        for i,row in m_molec.iterrows():
            if row[0] < Rcrit_pc:
                diff_235GHz_mjy_sumRcrit += row['diff_235GHz_mjy']
                diff_235GHz_mjy_beam345_sum_Rcrit += row['diff_235GHz_beam345_mjy']
                modflux235GHz_upto_Rcrit_pc_beam345 += m_molec345.loc[i, 'F235GHz_beam345_mjy']
            if row[0] < Rcrit_pc_art:
                diff_235GHz_mjy_sumRcrit_art += row['diff_235GHz_mjy']
                diff_235GHz_mjy_beam345_sum_Rcrit_art += row['diff_235GHz_beam345_mjy']
                modflux235GHz_upto_Rcrit_pc_art_beam345 += m_molec345.loc[i, 'F235GHz_beam345_mjy']
                modflux235GHz_upto_Rcrit_pc_art += m_molec.loc[i, 'F235GHz_mjy']
            if row[0] < FWHM219_conv_size_pc:
                diff_235GHz_mjy_sumconvsize += row['diff_235GHz_mjy']
                diff_235GHz_mjy_beam345_sumconvsize += row['diff_235GHz_beam345_mjy']
            
        ff_fraction_upto_Rcrit_pc_art_beam345 = 100*diff_235GHz_mjy_beam345_sum_Rcrit_art/modflux235GHz_upto_Rcrit_pc_art_beam345
        onlydust_upto_Rcrit_pc_art_beam345 = modflux235GHz_upto_Rcrit_pc_art_beam345-diff_235GHz_mjy_beam345_sum_Rcrit_art
        ff_fraction_upto_Rcrit_pc_beam345 = 100*diff_235GHz_mjy_beam345_sum_Rcrit/modflux235GHz_upto_Rcrit_pc_beam345
        onlydust_upto_Rcrit_pc_beam345 = modflux235GHz_upto_Rcrit_pc_beam345-diff_235GHz_mjy_beam345_sum_Rcrit
        
        def densprof(r,q,a):
            return a*r**(-q)
        def densprof2(r,q,a,q2, b):
            return a*r**(-q)+b*r**(-q2)
        a = 5.5e5
        q = 1.5
        q2 = -0.3
        b =1e6
        popt,pcov = curve_fit(densprof, np.array(rpc_profile[1:17]), np.array(nh2_profile_corr[1:17]),p0=[q, a])
        popt2,pcov2 = curve_fit(densprof, np.array(rpc_profile[17:]), np.array(nh2_profile_corr[17:]),p0=[q, a])
        popt3,pcov3 = curve_fit(densprof2, np.array(rpc_profile[17:]), np.array(nh2_profile_corr[17:]),p0=[q, a, q2, b])

        y2 = densprof(np.array(rpc_profile[1:]), *popt)
        
        ysub = np.array(nh2_profile_corr[1:]) - y2
        popt2,pcov2 = curve_fit(densprof, np.array(rpc_profile[1:]), ysub,p0=[q, a])

        nh2_plot = densprof(np.array(rpc_profile[1:]),1.5,5.7e5)
        ysum = densprof(np.array(rpc_profile[1:]),*popt) + densprof(np.array(rpc_profile[1:]),*popt2)
        plt.plot(rpc_profile[1:], nh2_profile_corr[1:], 'k')
        plt.plot(np.array(rpc_profile[1:]),densprof(np.array(rpc_profile[1:]),*popt),'b:',label='fit')
        plt.plot(np.array(rpc_profile[1:]),densprof(np.array(rpc_profile[1:]),*popt2),'g:',label='fit')
        plt.plot(np.array(rpc_profile[1:]),ysum,'r',label='fit')
        
def line_chi2(obs, mod, line, beam='_beam_orig'):
    """
        Calculates the chi2 for one transition
    """
    nch = np.count_nonzero(line+'_uplim')
    chi2 = 0
    for i, row in obs.iterrows():
        # Finding closest modeled rad value 
        mod_cl = mod.iloc[(mod[0]-row['dist']).abs().argsort()[:1]]
        mod_cl.reset_index(inplace=True, drop=True)
        mod_cl.values[0]
        if ~np.isnan(row[line+'_mJy_kms'+beam]):
            chi2 += ((row[line+'_mJy_kms'+beam]-mod_cl[line+beam][0])**2)/(7*nch*(0.3*row[line+'_mJy_kms'+beam]**2))
    return chi2                
    
def plot_models_and_inp_finalfig(Rcrit, line_column, modelos, hb_df, cont_df, my_model_path, figmod_path, figrt_path, fort_paths, results_path,
                                 writename = True, plot_CH3CN = False, plot_col = True, plot_opacity = False, D_Mpc = 3.5):
    """
       Plots model parameters and observations
    """
    distance_pc = D_Mpc*1e6
    mykeys = list(line_column.keys())
    plot_only_cont = ['model7']#['model2']
    plot_corr_cols = False
    plot_corr_abun = True
    if 'plot_T' not in list(line_column.keys()):
        mykeys.insert(1, 'plot_T')
        line_column['plot_T'] = []
        if plot_col:
            mykeys.insert(2, 'plot_col')
            line_column['plot_col'] = []
        else:
            mykeys.insert(2, 'plot_dens')
            line_column['plot_dens'] = []
        mykeys.insert(3, 'plot_x')
        line_column['plot_x'] = []
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    else:
        # Already ran, getting problems with ordered dict
        if plot_col:
            mykeys = ['plot_conts', 'plot_T', 'plot_col', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        else:
            mykeys = ['plot_conts', 'plot_T', 'plot_dens', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM'],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM'],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM'],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM'],
                   }
    
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
        hb_df['ratio_'+ratio+'_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']**2))**2)
        hb_df['ratio_'+ratio+'_beam345'] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']
        hb_df['ratio_'+ratio+'_beam345_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']**2))**2)

    figsize = 20
    tstring = 'Tex_SM_ave_ring' #Tex_det' #'Tex_ave_ring' # 'Tex_det'
    colstring = 'Col_SM_ave_ring' #'Col_det' #'Col_ave_ring' #'Col_det'
    naxis = 2
    maxis = 2
    labelsize = 28
    ticksize = 20
    fontsize = 18
    contms = 8
    axcolor = 'k'
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'None'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    axes=[]
    mykeys_conts = ['plot_conts', 'plot_T', 'plot_col', 'plot_x']
    
    for l,line in enumerate(mykeys_conts):
        axes.append(fig.add_subplot(gs1[l]))
        if line == 'plot_conts':
            axes[l].set_ylim([0.05, 20])  
            for i,row in cont_df.iterrows():
                axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam'], 
                                             yerr=row['F235GHz_mjy_beam_err'],
                                             marker='o', markersize=contms,
                                             markerfacecolor='k',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                
                axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam345'], 
                                             yerr=row['F235GHz_mjy_beam345_err'],
                                             marker='o', markersize=contms,
                                             markerfacecolor='w',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                axes[l].plot(row['dist'], row['F235GHz_mjy_beam345'],  linestyle='',
                                             marker='o', markersize=contms,
                                             markerfacecolor='w',
                                             markeredgecolor='k',
                                             zorder=2)
                axes[l].errorbar(row['dist'], row['F345GHz_mjy_beam'], 
                                             yerr=row['F345GHz_mjy_beam_err'],
                                             marker='o', markersize=contms,
                                             markerfacecolor='0.5',
                                             markeredgecolor='0.5', markeredgewidth=0.8,
                                             ecolor='0.5',
                                             color = '0.5',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                color = '0.5',
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                axes[l].text(0.95,  0.95-0.045, r'Cont. 235 GHz',
                                color = 'k',
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                
                axes[l].set_yscale('log')
                axes[l].set_ylabel(r'$\text{Flux density}\:(\text{mJy}\:\text{beam}^{-1})$', fontsize=labelsize)
                axes[l].yaxis.set_major_formatter(ScalarFormatter())
        
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        
        if l <2:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    save_name = ''
    ytext = 0.95
    ytext2 = 0.95
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
        model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
        mtau100 = get_tau100(my_model_path, modelo[0])
        total_NH2 = np.nansum(10**logNH2_profile)
        total_NH2_corr = np.nansum(10**logNH2_profile_corr)
        for l,line in enumerate(mykeys_conts):
            if line == 'plot_conts':
                axes[l].set_ylim([0.05, 20])  
                if mod in plot_only_cont: # Plotting only one cont lines. Too complicated Fig.
                    axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord)
                    axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                    axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord)
                    axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '--', zorder=mzord)
                elif len(plot_only_cont)<1:
                    axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord)
                    axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                    axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord)
                    axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '--', zorder=mzord)
            elif line == 'plot_T':
                
                if writename:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    qval = f'{np.float(qprof):1.1f}'
                    qstr = r'$q='+qval+'$'
                    #Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2)}'+r'\text{cm}^{-2}$'
                    Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                    Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                    axes[l].text(0.95, ytext, modstr+':  '+Lstr +' \; '+Nstr+' \; '+qstr,
                                color = mod_color,
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                ytext = ytext -0.045
                axes[l].set_ylabel(r'T$_{\text{dust}}$ (K)', fontsize = labelsize)#, labelpad=12)
                for i,row in obs_df.iterrows():
                    if i <= 2 or i>= 10:
                        axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                     uplims = True,
                                                     yerr=65,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    else:
                        axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                     yerr=row[tstring+'_err'],
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    if plot_CH3CN:
                        if row['Tex_err_CH3CN'] < 0:
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                     uplims = True,
                                                     yerr=200,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='b',
                                                     markeredgecolor='b', markeredgewidth=0.8,
                                                     ecolor='b',
                                                     color = 'b',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                     lolims = True,
                                                     yerr=200,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='b',
                                                     markeredgecolor='b', markeredgewidth=0.8,
                                                     ecolor='b',
                                                     color = 'b',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'], 
                                                         yerr=row['Tex_err_CH3CN'],
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='b',
                                                         markeredgecolor='b', markeredgewidth=0.8,
                                                         ecolor='b',
                                                         color = 'b',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                
                axes[l].plot(rpc_profile, td_profile, color=mod_color, zorder=mzord)
            elif line == 'plot_dens':
                
                axes[l].set_ylabel(r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', fontsize = labelsize)#, labelpad=12)
                axes[l].plot(rpc_profile[1:], nh2_profile[1:], color=mod_color, zorder=mzord)
                axes[l].set_yscale('log')
            elif line == 'plot_col':
                
                for i, row in obs_df.iterrows():
                    if row[colstring+'_err']>10:
                        col_err = (10**(row[colstring+'_err']+0.75))*(1/np.log(10))/(10**row[colstring])
                    else:
                        col_err = row[colstring+'_err']
                
                
                    if i>=10:
                        axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                     uplims = True,
                                                     yerr=0.25,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    else:
                        axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                     yerr=col_err,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        

                ytext2 = ytext2 -0.045
                axes[l].set_ylabel(r'$\log{N(\text{HC}_{3}\text{N})}$ (cm$^{-2}$)', fontsize = labelsize)#, labelpad=12)
                if plot_corr_cols:
                    axes[l].plot(rpc_profile[1:], logNHC3N_profile_corr[1:], color=mod_color, zorder=mzord)
                else:
                    axes[l].plot(rpc_profile[1:], logNHC3N_profile[1:], color=mod_color, zorder=mzord)
            elif line == 'plot_x':
                axes[l].set_ylabel(r'$X$ (HC$_{3}$N)', fontsize = labelsize)#, labelpad=12)
                if plot_corr_abun:
                    x_profile_corr = np.array(x_profile)*(10**logNH2_profile)/(10**logNH2_profile_corr)
                    axes[l].plot(rpc_profile[1:], x_profile_corr[1:], color=mod_color, zorder=mzord)
                else:
                    axes[l].plot(rpc_profile[1:], x_profile[1:], color=mod_color, zorder=mzord)
                axes[l].set_yscale('log')
            
        for l,line in enumerate(mykeys_conts):
            axes[l].tick_params(which='both',
                       labelright=False)
    if plot_opacity:
         # Plotting opacity
        gs2 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs2.update(wspace = 0.23, hspace=0.23, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        axtau = fig.add_subplot(gs2[-1])
        for m, mod in enumerate(modelos):
            mod_name = modelos[mod][0]
            mod_color = modelos[mod][3]
            mod_taudust = pd.read_csv(my_model_path+'/'+mod_name+'_.taudust', delim_whitespace= True)
            mod_taudust.columns = ['lambda_um', 'taudust']
            axtau.plot(mod_taudust['lambda_um'], mod_taudust['taudust'], color=mod_color)
            minor_locator = AutoMinorLocator(2)
            axtau.set_xlim([0.0, 100])
            axtau.tick_params(direction='in')
            axtau.tick_params(axis="both", which='major', length=8)
            axtau.tick_params(axis="both", which='minor', length=4)
            axtau.xaxis.set_tick_params(which='both', top ='on')
            axtau.yaxis.set_tick_params(which='both', right='on', labelright='off')
            axtau.tick_params(axis='both', which='major', labelsize=ticksize)
            axtau.xaxis.set_minor_locator(minor_locator)
            axtau.tick_params(labelleft=True,
                           labelright=False)
            axtau.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize = labelsize)
            axtau.set_ylabel(r'$\tau$', fontsize = labelsize)
        
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
        fig.savefig(figmod_path+'NGC253_'+save_name+'_conts_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_conts_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
    
    # Corrected densities and masses
    figsize = 10
    naxis = 2
    maxis = 1
    labelsize = 28
    ticksize = 20
    fontsize = 18
    contms = 8
    axcolor = 'k'
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'None'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    xlabel_ind = -1
    xtextpos = 0.95
    ytextpos = 0.95
    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    mykeys_flux = [r'$n_{\text{H}_2}$ (cm$^{-3}$)',r'$M_{\text{H}_2}$ (M$_\odot$)']
    for l,line in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
        model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
        for l,line in enumerate(mykeys_flux):
            if l==0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
                ytextpos = ytextpos -0.052
                # Uncorrected densities
                axes[l].plot(rpc_profile[1:], nh2_profile[1:], linestyle='--', color=mod_color, zorder=mzord)
                # Corrected densities
                axes[l].plot(rpc_profile[1:], nh2_profile_corr[1:], linestyle='-', color=mod_color, zorder=mzord)
            elif l==1:
                # Uncorrected Masses
                axes[l].plot(rpc_profile[1:], MH2_profile[1:], linestyle='--', color=mod_color, zorder=mzord)
                # Corrected Masses
                axes[l].plot(rpc_profile[1:], Mgas_fromnH2_Msun_profile_corr[1:], linestyle='-', color=mod_color, zorder=mzord)
            axes[l].set_ylabel(mykeys_flux[l], fontsize = labelsize)
    for l,line in enumerate(mykeys_flux):
        axes[l].set_yscale('log')
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_dens_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_dens_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()

    # Line profiles
    figsize = 20
    naxis = 3
    maxis = 4
    
    labelsize = 18
    ticksize = 16
    fontsize = 14
    linems = 6
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'w'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])   
    if maxis ==3:
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 9
        mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                   'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.15
        ytextpos = 0.95
    else:
        gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 8
        mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                    'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.90
        ytextpos = 0.85

    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    for l,line in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
            for i,row in hb_df.iterrows():
                if row['dist']<=line_column[line][0]:
                    ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                    if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                        hb_df.loc[i, line+'_uplim'] = True
                        axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                             uplims=True,
                                             yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                             marker='o', markersize=linems,
                                             markerfacecolor=color_beam_orig,
                                             markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                             ecolor=color_beam_orig,
                                             color = color_beam_orig,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                    else:
                        # Adding 10% error to the inner rings
                        if row['dist']<0.35:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                        else:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']
                        hb_df.loc[i, line+'_uplim'] = False
                        axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                         yerr=errplot,
                                         marker='o', markersize=linems,
                                         markerfacecolor=color_beam_orig,
                                         markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                         ecolor=color_beam_orig,
                                         color =color_beam_orig,
                                         elinewidth= 0.7,
                                         barsabove= True,
                                         zorder=2)
                    if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                        axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                             uplims=True,
                                             yerr=ysep,
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                             ecolor=color_beam_345,
                                             color = color_beam_345,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                        axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                             linestyle='',
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345,
                                             color = color_beam_345,
                                             zorder=2)
                    else:
                        # Adding 10% error to the inner rings
                        if row['dist']<0.35:
                            errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                        else:
                            errplot = row[line+'_mJy_kms_beam_345_errcont']
                        axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                         yerr=errplot,
                                         marker='o', markersize=linems,
                                         markerfacecolor=facecolor_beam_345,
                                         markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                         ecolor=color_beam_345,
                                         color =color_beam_345,
                                         elinewidth= 0.7,
                                         barsabove= True,
                                         zorder=1)
                        axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                             linestyle='',
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345,
                                             color = color_beam_345,
                                             zorder=2) 
        
            axes[l].set_ylim(line_column[line][2])  
            yminor_locator = AutoMinorLocator(2)
            axes[l].yaxis.set_minor_locator(yminor_locator)
            axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[l].transAxes)
            a = line_column[line][1].split(' ')[0]
            axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
        elif line == 'ratio_v6_v6v7':
            plot_ratio345 = False
            for i, row in hb_df.iterrows():
                if row['v6=v7=1_26_2_25_-2_SM_uplim']:
                    continue
                else:
                    axes[l].errorbar(row['dist'], row['ratio_v6_v6v7'], 
                                                 yerr=row['ratio_v6_v6v7_err'],
                                                 marker='o', markersize=linems,
                                                 markerfacecolor='k',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    if plot_ratio345:
                        axes[l].errorbar(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                                     yerr=row['ratio_v6_v6v7_beam345_err'],
                                                     marker='o', markersize=linems,
                                                     markerfacecolor='w',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        axes[l].plot(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                         linestyle='',
                                         marker='o', markersize=linems,
                                         markerfacecolor='w',
                                         markeredgecolor='k',
                                         color = 'k',
                                         zorder=2)
                    
                    axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                    axes[l].set_ylim([0.5, 4.7]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        for l,line in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
                ytextpos = ytextpos -0.052
            if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
            elif line in ['ratio_v6_v6v7']:
                mol_ratio = m_molec['v6=1_24_-1_23_1_SM_beam_orig']/m_molec['v6=v7=1_26_2_25_-2_SM_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    # Line ratios
    figsize = 20
    naxis = 2
    maxis = 2
    labelsize = 28
    ticksize = 20
    fontsize = 18
    contms = 8
    axcolor = 'k'
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'None'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    xlabel_ind = 1
    mykeys_flux = ['v0_v71_2625', 'v71_v61_2423', 'v71_v61_2625', 'v6_v6v7']
    # Making upper lim ratios
    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM', r'$v_{6}=1/v_{6}=v_{7}=1$', [1.0, 4.7]],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (24-23)$', [0.5, 4.7]],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (26-25)$', [0.5, 5.7]],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', r'$v=0/v_{7}=1 \:\: (26-25)$', [0.0, 8.7]],
                   }
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio+'_uplim'] = False
    for i, row in hb_df.iterrows():
        for r, ratio in enumerate(ratio_lines):
            uplim1 = 3*row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']
            uplim2 = 3*row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
            if uplim1 or uplim2:
                hb_df.loc[i, 'ratio_'+ratio+'_uplim'] = True
    xtextpos = 0.15
    ytextpos = 0.95
    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    
    for l,ratio in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        plot_ratio345 = False
        for i, row in hb_df.iterrows():
            if row['ratio_'+ratio+'_uplim']: # At least one of the lines is an upper limit
                continue
            else:
                axes[l].errorbar(row['dist'], row['ratio_'+ratio], 
                                             yerr=row['ratio_'+ratio+'_err'],
                                             marker='o', markersize=linems,
                                             markerfacecolor='k',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                if plot_ratio345:
                    axes[l].errorbar(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                                 yerr=row['ratio_'+ratio+'_beam345_err'],
                                                 marker='o', markersize=linems,
                                                 markerfacecolor='w',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    axes[l].plot(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                     linestyle='',
                                     marker='o', markersize=linems,
                                     markerfacecolor='w',
                                     markeredgecolor='k',
                                     color = 'k',
                                     zorder=2)
                
                axes[l].set_ylabel(ratio_lines[ratio][2], fontsize = labelsize)
                axes[l].set_ylim(ratio_lines[ratio][3]) 
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)     
    # Model ratios
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        for l,ratio in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
            mol_ratio = m_molec[ratio_lines[ratio][0]+'_beam_orig']/m_molec[ratio_lines[ratio][1]+'_beam_orig']
            axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
            if plot_ratio345:
                mol_ratio345 = m_molec345[ratio_lines[ratio][0]+'_beam_345']/m_molec345[ratio_lines[ratio][1]+'_beam_345']
                axes[l].plot(m_molec345[0], mol_ratio345, color=mod_color, linestyle= '--', zorder=2)
        ytextpos = ytextpos -0.045
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_ratios_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_ratios_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
    
    figsize = 10
    naxis = 2
    maxis = 1
    labelsize = 28
    ticksize = 20
    fontsize = 18
    contms = 8
    axcolor = 'k'
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'None'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    xlabel_ind = -1
    
    mykeys_flux = ['v71_v61_2423', 'v6_v6v7']
    
    xtextpos = 0.15
    ytextpos = 0.95

    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    
    # Making upper lim ratios
    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM', r'$v_{6}=1/v_{6}=v_{7}=1$', [1.1, 4.7]],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM', r'$v_{7}=1/v_{6}=1$', [0.6, 4.7]],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (26-25)$', [0.5, 5.7]],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', r'$v=0/v_{7}=1 \:\: (26-25)$', [0.0, 8.7]],
                   }
    
    for l,ratio in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        plot_ratio345 = False
        for i, row in hb_df.iterrows():
            
            if row['ratio_'+ratio+'_uplim']: # At least one of the lines is an upper limit
                continue
            else:
                axes[l].errorbar(row['dist'], row['ratio_'+ratio], 
                                             yerr=row['ratio_'+ratio+'_err'],
                                             marker='o', markersize=linems,
                                             markerfacecolor='k',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                if plot_ratio345:
                    axes[l].errorbar(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                                 yerr=row['ratio_'+ratio+'_beam345_err'],
                                                 marker='o', markersize=linems,
                                                 markerfacecolor='w',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    axes[l].plot(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                     linestyle='',
                                     marker='o', markersize=linems,
                                     markerfacecolor='w',
                                     markeredgecolor='k',
                                     color = 'k',
                                     zorder=2)
                
                axes[l].set_ylabel(ratio_lines[ratio][2], fontsize = labelsize)
                axes[l].set_ylim(ratio_lines[ratio][3]) 
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
                
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    # Model ratios
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        for l,ratio in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
            mol_ratio = m_molec[ratio_lines[ratio][0]+'_beam_orig']/m_molec[ratio_lines[ratio][1]+'_beam_orig']
            axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
            if plot_ratio345:
                mol_ratio345 = m_molec345[ratio_lines[ratio][0]+'_beam_345']/m_molec345[ratio_lines[ratio][1]+'_beam_345']
                axes[l].plot(m_molec345[0], mol_ratio345, color=mod_color, linestyle= '--', zorder=2)
        ytextpos = ytextpos -0.045
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_ratios2_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_ratios2_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()   

def plot_models_and_inp_comp(Rcrit, line_column, modelos, hb_df, cont_df, my_model_path, figmod_path, figrt_path, fort_paths, results_path,
                             writename = True, plot_CH3CN = False, plot_col = True, plot_opacity = False, D_Mpc = 3.5):
    """
        Plots model comparisson parameters
    """
    distance_pc = D_Mpc*1e6
    # Figures
    plot_conts_and_tex = False
    plot_corr_dens = False
    plot_line_profiles = True
    plot_line_opacities = True
    plot_line_Tex = True
    plot_line_ratios = False
    
    mykeys = list(line_column.keys())
    plot_only_cont = ['LTE 3E17']#['model7']
    plot_corr_cols = False
    plot_corr_abun = True
    
    if 'plot_T' not in list(line_column.keys()):
        mykeys.insert(1, 'plot_T')
        line_column['plot_T'] = []
        if plot_col:
            mykeys.insert(2, 'plot_col')
            line_column['plot_col'] = []
        else:
            mykeys.insert(2, 'plot_dens')
            line_column['plot_dens'] = []
        mykeys.insert(3, 'plot_x')
        line_column['plot_x'] = []
        if plot_opacity == False:
            mykeys.append('ratio_v0_v0')
    else:
        # Already ran, getting problems with ordered dict
        if plot_col:
            mykeys = ['plot_conts', 'plot_T', 'plot_col', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        else:
            mykeys = ['plot_conts', 'plot_T', 'plot_dens', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    
    ratio_lines = {'v0_v0':  ['v=0_24_23_SM', 'v=0_26_25_SM'],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM'],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM'],
                   'v0_v71_2625':  ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM'],
                   }
    
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
        hb_df['ratio_'+ratio+'_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']**2))**2)
        hb_df['ratio_'+ratio+'_beam345'] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']
        hb_df['ratio_'+ratio+'_beam345_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']**2))**2)

    # Conts and Tex figure
    if plot_conts_and_tex:
        figsize = 20
        tstring = 'Tex_SM_ave_ring' #Tex_det' #'Tex_ave_ring' # 'Tex_det'
        colstring = 'Col_SM_ave_ring' #'Col_det' #'Col_ave_ring' #'Col_det'
        naxis = 2
        maxis = 2
        labelsize = 28
        ticksize = 20
        fontsize = 18
        contms = 8
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        axes=[]
        mykeys_conts = ['plot_conts', 'plot_T', 'plot_col', 'plot_x']
        
        for l,line in enumerate(mykeys_conts):
            axes.append(fig.add_subplot(gs1[l]))
            if line == 'plot_conts':
                axes[l].set_ylim([0.05, 20])  
                for i,row in cont_df.iterrows():
                    axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam'], 
                                                 yerr=row['F235GHz_mjy_beam_err'],
                                                 marker='o', markersize=contms,
                                                 markerfacecolor='k',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    
                    axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam345'], 
                                                 yerr=row['F235GHz_mjy_beam345_err'],
                                                 marker='o', markersize=contms,
                                                 markerfacecolor='w',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    axes[l].plot(row['dist'], row['F235GHz_mjy_beam345'],  linestyle='',
                                                 marker='o', markersize=contms,
                                                 markerfacecolor='w',
                                                 markeredgecolor='k',
                                                 zorder=2)
                    axes[l].errorbar(row['dist'], row['F345GHz_mjy_beam'], 
                                                 yerr=row['F345GHz_mjy_beam_err'],
                                                 marker='o', markersize=contms,
                                                 markerfacecolor='0.5',
                                                 markeredgecolor='0.5', markeredgewidth=0.8,
                                                 ecolor='0.5',
                                                 color = '0.5',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                    color = '0.5',
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    axes[l].text(0.95,  0.95-0.045, r'Cont. 235 GHz',
                                    color = 'k',
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    axes[l].set_yscale('log')
                    axes[l].set_ylabel(r'$\text{Flux density}\:(\text{mJy}\:\text{beam}^{-1})$', fontsize=labelsize)
                    axes[l].yaxis.set_major_formatter(ScalarFormatter())
            
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            
            if l <2:
                axes[l].tick_params(
                           labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        save_name = ''
        ytext = 0.95
        ytext2 = 0.95
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            linestlye   = modelo[5]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc, tau_molec, tau_dust = model_reader_comp(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
            obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
            model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
            mtau100 = get_tau100(my_model_path, modelo[0])
            total_NH2 = np.nansum(10**logNH2_profile)
            total_NH2_corr = np.nansum(10**logNH2_profile_corr)
            for l,line in enumerate(mykeys_conts):
                if line == 'plot_conts':
                    axes[l].set_ylim([0.05, 20])  
                    if mod in plot_only_cont: # Plotting only one cont lines. Too complicated Fig.
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord)
                        axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= '--', zorder=mzord)
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '--', zorder=mzord)
                    elif len(plot_only_cont)<1:
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord)
                        axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= '--', zorder=mzord)
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '--', zorder=mzord)
                elif line == 'plot_T':
                    
                    if writename:
                        if 'model' in mod:
                            modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                        else:
                            modstr = mod
                        qval = f'{np.float(qprof):1.1f}'
                        qstr = r'$q='+qval+'$'
                        #Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2)}'+r'\text{cm}^{-2}$'
                        Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                        Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                        axes[l].text(0.95, ytext, modstr+':  '+Lstr +' \; '+Nstr,#+' \; '+qstr,
                                    color = mod_color,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    ytext = ytext -0.045
                    axes[l].set_ylabel(r'T$_{\text{dust}}$ (K)', fontsize = labelsize)#, labelpad=12)
                    for i,row in obs_df.iterrows():
                        if i <= 2 or i>= 10:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                         uplims = True,
                                                         yerr=65,
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='k',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                         yerr=row[tstring+'_err'],
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='k',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                        if plot_CH3CN:
                            if row['Tex_err_CH3CN'] < 0:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                         uplims = True,
                                                         yerr=200,
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='b',
                                                         markeredgecolor='b', markeredgewidth=0.8,
                                                         ecolor='b',
                                                         color = 'b',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                         lolims = True,
                                                         yerr=200,
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='b',
                                                         markeredgecolor='b', markeredgewidth=0.8,
                                                         ecolor='b',
                                                         color = 'b',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                            else:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'], 
                                                             yerr=row['Tex_err_CH3CN'],
                                                             marker='o', markersize=contms,
                                                             markerfacecolor='b',
                                                             markeredgecolor='b', markeredgewidth=0.8,
                                                             ecolor='b',
                                                             color = 'b',
                                                             elinewidth= 0.7,
                                                             barsabove= True,
                                                             zorder=1)
                    
                    axes[l].plot(rpc_profile, td_profile, color=mod_color, zorder=mzord, linestyle=linestlye)
                elif line == 'plot_dens':
                    
                    axes[l].set_ylabel(r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', fontsize = labelsize)#, labelpad=12)
                    axes[l].plot(rpc_profile[1:], nh2_profile[1:], color=mod_color, zorder=mzord, linestyle=linestlye)
                    axes[l].set_yscale('log')
                elif line == 'plot_col':
                    
                    for i, row in obs_df.iterrows():
                        if row[colstring+'_err']>10:
                            col_err = (10**(row[colstring+'_err']+0.75))*(1/np.log(10))/(10**row[colstring])
                        else:
                            col_err = row[colstring+'_err']
                    
                    
                        if i>=10:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                         uplims = True,
                                                         yerr=0.25,
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='k',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                         yerr=col_err,
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='k',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                            
    
                    ytext2 = ytext2 -0.045
                    axes[l].set_ylabel(r'$\log{N(\text{HC}_{3}\text{N})}$ (cm$^{-2}$)', fontsize = labelsize)#, labelpad=12)
                    if plot_corr_cols:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile_corr[1:], color=mod_color, zorder=mzord, linestyle=linestlye)
                    else:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile[1:], color=mod_color, zorder=mzord, linestyle=linestlye)
                elif line == 'plot_x':
                    axes[l].set_ylabel(r'$X$ (HC$_{3}$N)', fontsize = labelsize)#, labelpad=12)
                    if plot_corr_abun:
                        x_profile_corr = np.array(x_profile)*(10**logNH2_profile)/(10**logNH2_profile_corr)
                        axes[l].plot(rpc_profile[1:], x_profile_corr[1:], color=mod_color, zorder=mzord, linestyle=linestlye)
                    else:
                        axes[l].plot(rpc_profile[1:], x_profile[1:], color=mod_color, zorder=mzord, linestyle=linestlye)
                    axes[l].set_yscale('log')
                
            for l,line in enumerate(mykeys_conts):
                axes[l].tick_params(which='both',
                           labelright=False)
        if plot_opacity:
             # Plotting opacity
            gs2 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
            gs2.update(wspace = 0.23, hspace=0.23, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            axtau = fig.add_subplot(gs2[-1])
            for m, mod in enumerate(modelos):
                mod_name = modelos[mod][0]
                mod_color = modelos[mod][3]
                mod_taudust = pd.read_csv(my_model_path+'/'+mod_name+'_.taudust', delim_whitespace= True)
                mod_taudust.columns = ['lambda_um', 'taudust']
                axtau.plot(mod_taudust['lambda_um'], mod_taudust['taudust'], color=mod_color)
                minor_locator = AutoMinorLocator(2)
                axtau.set_xlim([0.0, 100])
                axtau.tick_params(direction='in')
                axtau.tick_params(axis="both", which='major', length=8)
                axtau.tick_params(axis="both", which='minor', length=4)
                axtau.xaxis.set_tick_params(which='both', top ='on')
                axtau.yaxis.set_tick_params(which='both', right='on', labelright='off')
                axtau.tick_params(axis='both', which='major', labelsize=ticksize)
                axtau.xaxis.set_minor_locator(minor_locator)
                axtau.tick_params(labelleft=True,
                               labelright=False)
                axtau.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize = labelsize)
                axtau.set_ylabel(r'$\tau$', fontsize = labelsize)
            
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_conts_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_conts_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()

    # Corrected densities and masses fig
    if plot_corr_dens:
        figsize = 10
        naxis = 2
        maxis = 1
        labelsize = 28
        ticksize = 20
        fontsize = 18
        contms = 8
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = -1
        
        xtextpos = 0.95
        ytextpos = 0.95
    
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        mykeys_flux = [r'$n_{\text{H}_2}$ (cm$^{-3}$)',r'$M_{\text{H}_2}$ (M$_\odot$)']
        for l,line in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc, tau_molec, tau_dust = model_reader_comp(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
            obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
            model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
            for l,line in enumerate(mykeys_flux):
                if l==0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                    ytextpos = ytextpos -0.052
                    # Uncorrected densities
                    axes[l].plot(rpc_profile[1:], nh2_profile[1:], linestyle='--', color=mod_color, zorder=mzord)
                    # Corrected densities
                    axes[l].plot(rpc_profile[1:], nh2_profile_corr[1:], linestyle='-', color=mod_color, zorder=mzord)
                elif l==1:
                    # Uncorrected Masses
                    axes[l].plot(rpc_profile[1:], MH2_profile[1:], linestyle='--', color=mod_color, zorder=mzord)
                    # Corrected Masses
                    axes[l].plot(rpc_profile[1:], Mgas_fromnH2_Msun_profile_corr[1:], linestyle='-', color=mod_color, zorder=mzord)
                axes[l].set_ylabel(mykeys_flux[l], fontsize = labelsize)
        for l,line in enumerate(mykeys_flux):
            axes[l].set_yscale('log')
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_dens_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_dens_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()

    # line profiles fig
    if plot_line_profiles:
        plot345 = False
        figsize = 20
        naxis = 2
        maxis = 3
        
        labelsize = 18
        ticksize = 16
        fontsize = 14
        linems = 6
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'w'
        # Fig for fluxes
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)
        
        if maxis ==3:
            gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 3
            #mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
            #           'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
            #           'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
            #           'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v0_v0']
            mykeys_flux = ['v=0_24_23_SM', 'v=0_26_25_SM', 
                           'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 
                           'v6=1_24_-1_23_1_SM', 'v6=1_26_-1_25_1_SM',
                           ]
            xtextpos = 0.25
            ytextpos = 0.95
        else:
            gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 8
            mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                       'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                       'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                        'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v0_v0']
            xtextpos = 0.90
            ytextpos = 0.85
    
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        
                
        for l,line in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v0_v0']:   
                for i,row in hb_df.iterrows():
                    if row['dist']<=line_column[line][0]:
                        ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                        if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                            hb_df.loc[i, line+'_uplim'] = True
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                                 uplims=True,
                                                 yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=color_beam_orig,
                                                 markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                                 ecolor=color_beam_orig,
                                                 color = color_beam_orig,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']
                            hb_df.loc[i, line+'_uplim'] = False
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                             yerr=errplot,
                                             marker='o', markersize=linems,
                                             markerfacecolor=color_beam_orig,
                                             markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                             ecolor=color_beam_orig,
                                             color =color_beam_orig,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=2)
                        if plot345:
                            if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                                axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                     uplims=True,
                                                     yerr=ysep,#3*row[line+'_mJy_kms_beam_345_errcont']*0.15,
                                                     marker='o', markersize=linems,
                                                     markerfacecolor=facecolor_beam_345,
                                                     markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                     ecolor=color_beam_345,
                                                     color = color_beam_345,
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                                axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                     linestyle='',
                                                     marker='o', markersize=linems,
                                                     markerfacecolor=facecolor_beam_345,
                                                     markeredgecolor=color_beam_345,
                                                     color = color_beam_345,
                                                     zorder=2)
                            else:
                                # Adding 10% error to the inner rings
                                if row['dist']<0.35:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                                else:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']
                                axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                 yerr=errplot,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                 ecolor=color_beam_345,
                                                 color =color_beam_345,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                                axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                     linestyle='',
                                                     marker='o', markersize=linems,
                                                     markerfacecolor=facecolor_beam_345,
                                                     markeredgecolor=color_beam_345,
                                                     color = color_beam_345,
                                                     zorder=2) 
            
                axes[l].set_ylim(line_column[line][2])  
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
                axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                a = line_column[line][1].split(' ')[0]
                axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
            elif line == 'ratio_v0_v0':
                plot_ratio345 = False
                for i, row in hb_df.iterrows():
                    if row['v6=v7=1_26_2_25_-2_SM_uplim']:
                        continue
                    else:
                        axes[l].errorbar(row['dist'], row['ratio_v6_v6v7'], 
                                                     yerr=row['ratio_v6_v6v7_err'],
                                                     marker='o', markersize=linems,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        
                        if plot_ratio345:
                            axes[l].errorbar(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                                         yerr=row['ratio_v6_v6v7_beam345_err'],
                                                         marker='o', markersize=linems,
                                                         markerfacecolor='w',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                            axes[l].plot(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                             linestyle='',
                                             marker='o', markersize=linems,
                                             markerfacecolor='w',
                                             markeredgecolor='k',
                                             color = 'k',
                                             zorder=2)
                        
                        axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                        axes[l].set_ylim([0.5, 4.7]) 
                        yminor_locator = AutoMinorLocator(2)
                        axes[l].yaxis.set_minor_locator(yminor_locator)
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                           labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    
                                
        for m, mod in enumerate(modelos):
            mykeys_flux = ['v=0_24_23_SM', 'v=0_26_25_SM', 
                           'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 
                           'v6=1_24_-1_23_1_SM', 'v6=1_26_-1_25_1_SM',
                           ]
            mykeys_flux_beam_orig = []
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            linestlye   = modelo[5]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc, tau_molec, tau_dust = model_reader_comp(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
            # Saving line fluxes and opacities in the corresponding dist rings to file
            model_rings = m_molec[m_molec[0].isin(hb_df['dist'])]
            model_rings['r_pc'] = model_rings[0]*1.0
            model_rings = model_rings[['r_pc']+[fl+'_beam_orig' for fl in mykeys_flux]]
            model_rings.to_excel(figrt_path+'_'+modelos[mod][0]+'_fluxes.xlsx')
            if not tau_molec.empty:
                # Finding closest modeled rad value ro rings
                subind = []
                for r,row in hb_df.iterrows():
                    subind.append(tau_molec.iloc[(tau_molec[0]-row['dist']).abs().argsort()[:1]].index[0])
                tau_molec = tau_molec.iloc[subind]
                tau_molec.reset_index(inplace=True, drop=True)
                model_opacit_rings = tau_molec
                model_opacit_rings['r_pc'] = model_opacit_rings[0]*1.0
                model_opacit_rings = model_opacit_rings[['r_pc']+[fl+'_tau' for fl in mykeys_flux]]
                model_opacit_rings.to_excel(figrt_path+'_'+modelos[mod][0]+'_opacities.xlsx')
                
            # Cont opacities
            # Finding closest modeled rad value ro rings
            subind = []
            for r,row in hb_df.iterrows():
                subind.append(tau_dust.iloc[(tau_dust[0]-row['dist']).abs().argsort()[:1]].index[0])
            tau_dust = tau_dust.iloc[subind]
            tau_dust.reset_index(inplace=True, drop=True)
            model_contopacit_rings = tau_dust
            model_contopacit_rings['r_pc'] = model_contopacit_rings[0]*1.0
            model_contopacit_rings = model_contopacit_rings[['r_pc', 'tau_235', 'tau_345']]
            model_contopacit_rings.to_excel(figrt_path+'_'+modelos[mod][0]+'_cont_opacities.xlsx')
            for l,line in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                    ytextpos = ytextpos -0.052
                if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v0_v0']: 
                    if plot345:
                        axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= linestlye, zorder=mzord)
                    
                
                elif line in ['ratio_v0_v0']:
                    mol_ratio = m_molec['v=0_24_23_SM_beam_orig']/m_molec['v=0_26_25_SM_beam_orig']
                    axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= linestlye, zorder=mzord)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()

    # Line and cont. Opacities
    if plot_line_opacities:
        labelsize = 18
        ticksize = 16
        fontsize = 14
        xtextmod = 0.90
        ytextmod = 0.95
        xtextpos = 0.85
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            linestlye   = modelo[5]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc, tau_molec, tau_dust = model_reader_comp(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
            
            if not tau_molec.empty:
                # Models in LTE have vel recubrimiento = 0 and do not write opacities in the output
                mykeys_flux = [['tau_235', 'tau_345'],
                               ['v=0_24_23_SM', 'v=0_26_25_SM'],
                               ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM'],
                               ['v6=1_24_-1_23_1_SM', 'v6=1_26_-1_25_1_SM'],
                               ]
                if mod == 'NLTE 1E16':
                    lines_lims = [[0.01,1],
                              [0.02, 5],
                              [0.02, 2],
                              [0.02, 0.5]]
                else:
                    lines_lims = [[0.01,1],
                                  [0.03, 50],
                                  [0.03, 20],
                                  [0.03, 10]]
                axes = []
                linestyles = ['-', '--']
                cont_col = ['r','b']
                cont_str = ['235GHz','345GHz']
                figsize = 20
                naxis = 2
                maxis = 2
                fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
                gs = gridspec.GridSpec(maxis, naxis)
                gs.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
                xlabel_ind = 1
                for p,linepair in enumerate(mykeys_flux):
                    axes.append(fig.add_subplot(gs[p]))
                    ytextpos = 0.85
                    if p == 0:
                        if 'model' in mod:
                            modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                        else:
                            modstr = mod
                        axes[p].text(xtextmod, ytextmod, modstr,
                                    color = mod_color,  
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[p].transAxes)
                        for l, line in enumerate(linepair):
                            axes[p].plot(tau_dust[0], tau_dust[line], color=cont_col[l],
                                        linestyle= '-', zorder=mzord)
                            axes[p].text(xtextpos, ytextpos, cont_str[l],
                                                color = cont_col[l],  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[p].transAxes)
                            ytextpos = ytextpos -0.052
                        axes[p].set_ylabel(r'$\tau$\:\ Cont.', fontsize=labelsize)
                        
                    else:
                        for l, line in enumerate(linepair):
                            axes[p].plot(tau_molec[0], tau_molec[line+'_tau'], color=mod_color,
                                        linestyle= linestyles[l], zorder=mzord)
                        a = line_column[linepair[0]][1].split(' ')[0]
                        axes[p].set_ylabel(r'$\tau$\:\,'+a, fontsize=labelsize)
                    axes[p].set_ylim(lines_lims[p])  
                    yminor_locator = AutoMinorLocator(2)
                    axes[p].yaxis.set_minor_locator(yminor_locator)
                    minor_locator = AutoMinorLocator(2)
                    axes[p].set_xlim([0.0, 1.42])
                    axes[p].tick_params(direction='in')
                    axes[p].tick_params(axis="both", which='major', length=8)
                    axes[p].tick_params(axis="both", which='minor', length=4)
                    axes[p].xaxis.set_tick_params(which='both', top ='on')
                    axes[p].yaxis.set_tick_params(which='both', right='on', labelright='off')
                    axes[p].tick_params(axis='both', which='major', labelsize=ticksize)
                    axes[p].xaxis.set_minor_locator(minor_locator)
                    axes[p].tick_params(labelleft=True,
                                   labelright=False)
                    if p <=xlabel_ind:
                        axes[p].tick_params(
                                   labelbottom=False)
                    else:
                        axes[p].set_xlabel(r'r (pc)', fontsize = labelsize)
                fig.savefig(figrt_path+'NGC253_'+modstr+'_opacities.pdf', bbox_inches='tight', transparent=True, dpi=400)
                plt.close()
                
    # Tex figures
    if plot_line_Tex:
        labelsize = 18
        ticksize = 16
        fontsize = 14
        for m, mod in enumerate(modelos):
            if 'model' in mod:
                modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
            else:
                modstr = mod
            modelo = modelos[mod]
            model_name = modelo[0]
            dust_model = modelo[1]
            source_rad = modelo[2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if not LTE:
                xtextpos = 0.85
                ytextpos = 0.90
                figsize = 20
                naxis = 1
                maxis = 1
                axes = []
                fig = plt.figure(figsize=(figsize*naxis/maxis*1.05, figsize*0.95))
                gs1 = gridspec.GridSpec(maxis, naxis)
                axes.append(fig.add_subplot(gs1[0]))
                # Dust temperatures (from cont and line models, should be the same)
                ## From dust models
                modelo_dust = my_model_path+dust_model+'.inp'
                with open(modelo_dust, 'r') as file:
                    modelo_dustlines = file.readlines()
                    file.close()
                #dist_model_dust = np.float(modelo_dustlines[16].split(' ')[6])
                rout_model_dust_pc = np.float(list(filter(None, modelo_dustlines[2].split(' ')))[1])*(1*u.cm).to(u.pc).value
                Tdust_cont = pd.read_csv(my_model_path+dust_model+'_.tem', header=None, delim_whitespace=True) 
                Tdust_cont['r_pc'] = Tdust_cont[0]/rout_model_dust_pc/(1*u.pc).to(u.cm).value*source_rad
                Tdust_cont['Tdust'] = Tdust_cont[1]*1.0
                axes[0].plot(Tdust_cont['r_pc'], Tdust_cont['Tdust'], color='k', linestyle='-')
                ## From line models 
                Tdust_line = pd.read_csv(my_model_path+model_name+'_.temdust', header=None, delim_whitespace=True) 
                Tdust_line['r_pc'] = Tdust_line[0]*1.0
                Tdust_line['Tdust'] = Tdust_line[1]*1.0
                axes[0].plot(Tdust_line['r_pc'], Tdust_line['Tdust'], color='grey', linestyle='--')
                axes[0].text(xtextpos, ytextpos, 'Tdust',
                            color = 'k',  
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[0].transAxes)
                ytextpos = ytextpos - 0.026
                # Lines Tex
                Tex_df = pd.read_csv(my_model_path+model_name+'_1rec.tex', header=None, delim_whitespace=True) 
                Tex_df['r_pc'] = Tex_df[0]*(1*u.cm).to(u.pc).value
                for l,line in enumerate(line_column):
                    if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x']:
                        Tex_df[line+'_Tex'] = Tex_df[line_column[line][3]]
                        
                        if 'v=0' in line:
                            line_str = 'v=0'
                            color = 'b'
                        elif 'v7=1' in line:
                            color = 'r'
                            line_str = 'v7=1'
                        elif 'v6=1' in line:
                            color = 'lime'
                            line_str = 'v6=1'
                        if l%2 == 0: 
                            ls = '--'
                            axes[0].text(xtextpos, ytextpos, line_str,
                                        color = color,  
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[0].transAxes)
                            ytextpos = ytextpos - 0.026
                        else:
                            ls = '-'
                        axes[0].plot(Tex_df['r_pc'], Tex_df[line+'_Tex'], color=color, linestyle=ls, zorder = 1)
                        
                        
                
                axes[0].set_ylabel(r'T (K)', fontsize=labelsize)
                axes[0].set_ylim([10, 500])  
                yminor_locator = AutoMinorLocator(2)
                axes[0].yaxis.set_minor_locator(yminor_locator)
                minor_locator = AutoMinorLocator(2)
                axes[0].set_xlim([0.0, 1.42])
                axes[0].tick_params(direction='in')
                axes[0].tick_params(axis="both", which='major', length=8)
                axes[0].tick_params(axis="both", which='minor', length=4)
                axes[0].xaxis.set_tick_params(which='both', top ='on')
                axes[0].yaxis.set_tick_params(which='both', right='on', labelright='off')
                axes[0].tick_params(axis='both', which='major', labelsize=ticksize)
                axes[0].xaxis.set_minor_locator(minor_locator)
                axes[0].tick_params(labelleft=True,
                               labelright=False)
                axes[0].set_xlabel(r'r (pc)', fontsize = labelsize)
                fig.savefig(figrt_path+'NGC253_'+modstr+'_Tex.pdf', bbox_inches='tight', transparent=True, dpi=400)
                plt.close()
            
    if plot_line_ratios:
        # Line ratios
        figsize = 20
        naxis = 2
        maxis = 2
        labelsize = 28
        ticksize = 20
        fontsize = 18
        contms = 8
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 1
        mykeys_flux = ['v0_v71_2625', 'v71_v61_2423', 'v71_v61_2625', 'v0_v0']
        # Making upper lim ratios
        ratio_lines = {'v0_v0': ['v=0_24_23_SM', 'v=0_26_25_SM', r'$v=0 (24-23)/v=0 (26-25)$', [0.24, 1.5]],
                       'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (24-23)$', [0.5, 4.7]],
                       'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (26-25)$', [0.5, 5.7]],
                       'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', r'$v=0/v_{7}=1 \:\: (26-25)$', [0.0, 8.7]],
                       }
        
        for r, ratio in enumerate(ratio_lines):
            hb_df['ratio_'+ratio+'_uplim'] = False
        for i, row in hb_df.iterrows():
            for r, ratio in enumerate(ratio_lines):
                uplim1 = 3*row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']
                uplim2 = 3*row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
                if uplim1 or uplim2:
                    hb_df.loc[i, 'ratio_'+ratio+'_uplim'] = True
        xtextpos = 0.15
        ytextpos = 0.95
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        
        for l,ratio in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            plot_ratio345 = False
            for i, row in hb_df.iterrows():
                if row['ratio_'+ratio+'_uplim']: # At least one of the lines is an upper limit
                    continue
                else:
                    axes[l].errorbar(row['dist'], row['ratio_'+ratio], 
                                                 yerr=row['ratio_'+ratio+'_err'],
                                                 marker='o', markersize=linems,
                                                 markerfacecolor='k',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    if plot_ratio345:
                        axes[l].errorbar(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                                     yerr=row['ratio_'+ratio+'_beam345_err'],
                                                     marker='o', markersize=linems,
                                                     markerfacecolor='w',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        axes[l].plot(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                         linestyle='',
                                         marker='o', markersize=linems,
                                         markerfacecolor='w',
                                         markeredgecolor='k',
                                         color = 'k',
                                         zorder=2)
                    
                    axes[l].set_ylabel(ratio_lines[ratio][2], fontsize = labelsize)
                    axes[l].set_ylim(ratio_lines[ratio][3]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
                    
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                           labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
               
        # Model ratios
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            linestlye   = modelo[5]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
            for l,ratio in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                mol_ratio = m_molec[ratio_lines[ratio][0]+'_beam_orig']/m_molec[ratio_lines[ratio][1]+'_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= linestlye, zorder=mzord)
                if plot_ratio345:
                    mol_ratio345 = m_molec345[ratio_lines[ratio][0]+'_beam_345']/m_molec345[ratio_lines[ratio][1]+'_beam_345']
                    axes[l].plot(m_molec345[0], mol_ratio345, color=mod_color, linestyle= '--', zorder=2)
            ytextpos = ytextpos -0.045
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_ratios_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_ratios_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()

def plot_models_and_inp_finalfig_diap(Rcrit, line_column, modelos, hb_df, cont_df, my_model_path, figmod_path, figrt_path, fort_paths, results_path,
                                      writename = True, plot_CH3CN = False, plot_col = True, plot_opacity = False, D_Mpc = 3.5):
    """
        Plots models profiles and values in a more clear figure
    """
    distance_pc = D_Mpc*1e6
    mykeys = list(line_column.keys())
    plot_only_cont = []
    plot_corr_cols = False
    plot_corr_abun = True
    plot_smoothed = False
    if 'plot_T' not in list(line_column.keys()):
        mykeys.insert(1, 'plot_T')
        line_column['plot_T'] = []
        if plot_col:
            mykeys.insert(2, 'plot_col')
            line_column['plot_col'] = []
        else:
            mykeys.insert(2, 'plot_dens')
            line_column['plot_dens'] = []
        mykeys.insert(3, 'plot_x')
        line_column['plot_x'] = []
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    else:
        # Already ran, getting problems with ordered dict
        if plot_col:
            mykeys = ['plot_conts', 'plot_T', 'plot_col', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        else:
            mykeys = ['plot_conts', 'plot_T', 'plot_dens', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    
    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM'],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM'],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM'],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM'],
                   }
    
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
        hb_df['ratio_'+ratio+'_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']**2))**2)
        hb_df['ratio_'+ratio+'_beam345'] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']
        hb_df['ratio_'+ratio+'_beam345_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']**2))**2)
    
    figsize = 20
    tstring = 'Tex_SM_ave_ring' #Tex_det' #'Tex_ave_ring' # 'Tex_det'
    colstring = 'Col_SM_ave_ring' #'Col_det' #'Col_ave_ring' #'Col_det'
    naxis = 2
    maxis = 2
    labelsize = 28
    ticksize = 20
    fontsize = 18
    contms = 8
    axcolor = 'k'
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'None'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
    gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
    axes=[]
    mykeys_conts = ['plot_conts', 'plot_T', 'plot_col', 'plot_x']
    
    for l,line in enumerate(mykeys_conts):
        axes.append(fig.add_subplot(gs1[l]))
        if line == 'plot_conts':
            axes[l].set_ylim([0.05, 20])  
            for i,row in cont_df.iterrows():
                axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam'], 
                                             yerr=row['F235GHz_mjy_beam_err'],
                                             marker='o', markersize=contms,
                                             markerfacecolor='k',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                
                
                axes[l].text(0.95, 0.95, r'Cont. 235GHz 0.02"',
                                color = 'k',
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                
                if plot_smoothed:
                    axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam345'], 
                                             yerr=row['F235GHz_mjy_beam345_err'],
                                             marker='o', markersize=contms,
                                             markerfacecolor='0.65',
                                             markeredgecolor='k', markeredgewidth=0.8,
                                             ecolor='k',
                                             color = 'k',
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                    axes[l].plot(row['dist'], row['F235GHz_mjy_beam345'],  linestyle='',
                                             marker='o', markersize=contms,
                                             markerfacecolor='0.65',
                                             markeredgecolor='k',
                                             zorder=2)
                    axes[l].text(0.95,  0.95-0.045, r'Cont. 235GHz 0.03"',
                                    color = '0.5',
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                
                axes[l].set_yscale('log')
                axes[l].set_ylabel(r'$\text{Flux density}\:(\text{mJy}\:\text{beam}^{-1})$', fontsize=labelsize)
                axes[l].yaxis.set_major_formatter(ScalarFormatter())
        
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        
        if l <2:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    save_name = ''
    ytext = 0.95
    ytext2 = 0.95
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
        model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
        mtau100 = get_tau100(my_model_path, modelo[0])
        total_NH2 = np.nansum(10**logNH2_profile)
        total_NH2_corr = np.nansum(10**logNH2_profile_corr)
        for l,line in enumerate(mykeys_conts):
            if line == 'plot_conts':
                axes[l].set_ylim([0.05, 20])  
                if mod in plot_only_cont: # Plotting only one cont lines. Too complicated Fig.
                    axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord)
                    if plot_smoothed:
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                elif len(plot_only_cont)<1:
                    axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '-', zorder=mzord)
                    if plot_smoothed:
                        axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= '--', zorder=mzord)
            elif line == 'plot_T':
                if writename:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    qval = f'{np.float(qprof):1.1f}'
                    qstr = r'$q='+qval+'$'
                    Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                    Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                    axes[l].text(0.95, ytext, modstr+':  '+Lstr +' \; '+Nstr+' \; '+qstr,
                                color = mod_color,
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                ytext = ytext -0.045
                axes[l].set_ylabel(r'T$_{\text{dust}}$ (K)', fontsize = labelsize)#, labelpad=12)
                for i,row in obs_df.iterrows():
                    if i <= 2 or i>= 10:
                        axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                     uplims = True,
                                                     yerr=65,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    else:
                        axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                     yerr=row[tstring+'_err'],
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    if plot_CH3CN:
                        if row['Tex_err_CH3CN'] < 0:
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                     uplims = True,
                                                     yerr=200,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='b',
                                                     markeredgecolor='b', markeredgewidth=0.8,
                                                     ecolor='b',
                                                     color = 'b',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                     lolims = True,
                                                     yerr=200,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='b',
                                                     markeredgecolor='b', markeredgewidth=0.8,
                                                     ecolor='b',
                                                     color = 'b',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'], 
                                                         yerr=row['Tex_err_CH3CN'],
                                                         marker='o', markersize=contms,
                                                         markerfacecolor='b',
                                                         markeredgecolor='b', markeredgewidth=0.8,
                                                         ecolor='b',
                                                         color = 'b',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                axes[l].plot(rpc_profile, td_profile, color=mod_color, zorder=mzord)
            elif line == 'plot_dens':
                axes[l].set_ylabel(r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', fontsize = labelsize)#, labelpad=12)
                axes[l].plot(rpc_profile[1:], nh2_profile[1:], color=mod_color, zorder=mzord)
                axes[l].set_yscale('log')
            elif line == 'plot_col':
                
                for i, row in obs_df.iterrows():
                    if row[colstring+'_err']>10:
                        col_err = (10**(row[colstring+'_err']+0.75))*(1/np.log(10))/(10**row[colstring])
                    else:
                        col_err = row[colstring+'_err']
                
                
                    if i>=10:
                        axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                     uplims = True,
                                                     yerr=0.25,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                    else:
                        axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                     yerr=col_err,
                                                     marker='o', markersize=contms,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                ytext2 = ytext2 -0.045
                axes[l].set_ylabel(r'$\log{N(\text{HC}_{3}\text{N})}$ (cm$^{-2}$)', fontsize = labelsize)#, labelpad=12)
                if plot_corr_cols:
                    axes[l].plot(rpc_profile[1:], logNHC3N_profile_corr[1:], color=mod_color, zorder=mzord)
                else:
                    axes[l].plot(rpc_profile[1:], logNHC3N_profile[1:], color=mod_color, zorder=mzord)
            elif line == 'plot_x':
                axes[l].set_ylabel(r'$X$ (HC$_{3}$N)', fontsize = labelsize)#, labelpad=12)
                if plot_corr_abun:
                    x_profile_corr = np.array(x_profile)*(10**logNH2_profile)/(10**logNH2_profile_corr)
                    axes[l].plot(rpc_profile[1:], x_profile_corr[1:], color=mod_color, zorder=mzord)
                else:
                    axes[l].plot(rpc_profile[1:], x_profile[1:], color=mod_color, zorder=mzord)
                axes[l].set_yscale('log')
        for l,line in enumerate(mykeys_conts):
            axes[l].tick_params(which='both',
                       labelright=False)
    if plot_opacity:
         # Plotting opacity
        gs2 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs2.update(wspace = 0.23, hspace=0.23, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        axtau = fig.add_subplot(gs2[-1])
        for m, mod in enumerate(modelos):
            mod_name = modelos[mod][0]
            mod_color = modelos[mod][3]
            mod_taudust = pd.read_csv(my_model_path+'/'+mod_name+'_.taudust', delim_whitespace= True)
            mod_taudust.columns = ['lambda_um', 'taudust']
            axtau.plot(mod_taudust['lambda_um'], mod_taudust['taudust'], color=mod_color)
            minor_locator = AutoMinorLocator(2)
            axtau.set_xlim([0.0, 100])
            axtau.tick_params(direction='in')
            axtau.tick_params(axis="both", which='major', length=8)
            axtau.tick_params(axis="both", which='minor', length=4)
            axtau.xaxis.set_tick_params(which='both', top ='on')
            axtau.yaxis.set_tick_params(which='both', right='on', labelright='off')
            axtau.tick_params(axis='both', which='major', labelsize=ticksize)
            axtau.xaxis.set_minor_locator(minor_locator)
            axtau.tick_params(labelleft=True,
                           labelright=False)
            axtau.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize = labelsize)
            axtau.set_ylabel(r'$\tau$', fontsize = labelsize)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
        fig.savefig(figmod_path+'NGC253_'+save_name+'_conts_SM_prese.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_conts_SM_presen.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()

    # line profiles
    figsize = 20
    naxis = 3
    maxis = 4
    labelsize = 18
    ticksize = 16
    fontsize = 14
    linems = 6
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'w'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)  
    if maxis ==3:
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 9
        mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                   'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.15
        ytextpos = 0.95
    else:
        gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 8
        mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                    'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.90
        ytextpos = 0.85

    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    
            
    for l,line in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
            for i,row in hb_df.iterrows():
                if row['dist']<=line_column[line][0]:
                    ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                    if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                        hb_df.loc[i, line+'_uplim'] = True
                        axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                             uplims=True,
                                             yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                             marker='o', markersize=linems,
                                             markerfacecolor=color_beam_orig,
                                             markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                             ecolor=color_beam_orig,
                                             color = color_beam_orig,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                    else:
                        # Adding 10% error to the inner rings
                        if row['dist']<0.35:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                        else:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']
                        hb_df.loc[i, line+'_uplim'] = False
                        axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                         yerr=errplot,
                                         marker='o', markersize=linems,
                                         markerfacecolor=color_beam_orig,
                                         markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                         ecolor=color_beam_orig,
                                         color =color_beam_orig,
                                         elinewidth= 0.7,
                                         barsabove= True,
                                         zorder=2)
                    if plot_smoothed:
                        if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 uplims=True,
                                                 yerr=ysep,#3*row[line+'_mJy_kms_beam_345_errcont']*0.15,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                 ecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                            axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                             yerr=errplot,
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                             ecolor=color_beam_345,
                                             color =color_beam_345,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                            axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2) 
        
            axes[l].set_ylim(line_column[line][2])  
            yminor_locator = AutoMinorLocator(2)
            axes[l].yaxis.set_minor_locator(yminor_locator)
            axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[l].transAxes)
            a = line_column[line][1].split(' ')[0]
            axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
        elif line == 'ratio_v6_v6v7':
            plot_ratio345 = False
            for i, row in hb_df.iterrows():
                if row['v6=v7=1_26_2_25_-2_SM_uplim']:
                    continue
                else:
                    axes[l].errorbar(row['dist'], row['ratio_v6_v6v7'], 
                                                 yerr=row['ratio_v6_v6v7_err'],
                                                 marker='o', markersize=linems,
                                                 markerfacecolor='k',
                                                 markeredgecolor='k', markeredgewidth=0.8,
                                                 ecolor='k',
                                                 color = 'k',
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                    if plot_ratio345:
                        axes[l].errorbar(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                                     yerr=row['ratio_v6_v6v7_beam345_err'],
                                                     marker='o', markersize=linems,
                                                     markerfacecolor='w',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        axes[l].plot(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                         linestyle='',
                                         marker='o', markersize=linems,
                                         markerfacecolor='w',
                                         markeredgecolor='k',
                                         color = 'k',
                                         zorder=2)
                    
                    axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                    axes[l].set_ylim([0.5, 4.7]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)

                            
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        for l,line in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
                ytextpos = ytextpos -0.052
            if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                if plot_smoothed:
                    axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
            
            elif line in ['ratio_v6_v6v7']:
                mol_ratio = m_molec['v6=1_24_-1_23_1_SM_beam_orig']/m_molec['v6=v7=1_26_2_25_-2_SM_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM_prese.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM_presen.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
    # line profiles
    figsize = 20
    naxis = 3
    maxis = 2
    
    labelsize = 32
    ticksize = 24
    fontsize = 28
    linems = 6
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'w'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])   
    if maxis ==3:
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 9
        mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                   'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.15
        ytextpos = 0.95
    else:
        gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 2
        mykeys_flux = ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM',
                       'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM', 'v4=1_26_25_SM']
        xtextpos = 0.90
        ytextpos = 0.85

    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    
            
    for l,line in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
            for i,row in hb_df.iterrows():
                if row['dist']<=line_column[line][0]:
                    ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                    if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                        hb_df.loc[i, line+'_uplim'] = True
                        axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                             uplims=True,
                                             yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                             marker='o', markersize=linems,
                                             markerfacecolor=color_beam_orig,
                                             markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                             ecolor=color_beam_orig,
                                             color = color_beam_orig,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                    else:
                        # Adding 10% error to the inner rings
                        if row['dist']<0.35:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                        else:
                            errplot = row[line+'_mJy_kms_beam_orig_errcont']
                        hb_df.loc[i, line+'_uplim'] = False
                        axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                         yerr=errplot,
                                         marker='o', markersize=linems,
                                         markerfacecolor=color_beam_orig,
                                         markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                         ecolor=color_beam_orig,
                                         color =color_beam_orig,
                                         elinewidth= 0.7,
                                         barsabove= True,
                                         zorder=2)
                    if plot_smoothed:
                        if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 uplims=True,
                                                 yerr=ysep,#3*row[line+'_mJy_kms_beam_345_errcont']*0.15,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                 ecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                            axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                             yerr=errplot,
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                             ecolor=color_beam_345,
                                             color =color_beam_345,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                            axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2) 
        
            axes[l].set_ylim(line_column[line][2])  
            yminor_locator = AutoMinorLocator(2)
            axes[l].yaxis.set_minor_locator(yminor_locator)
            axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=fontsize,
                            transform=axes[l].transAxes)
            a = line_column[line][1].split(' ')[0]
            axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)

        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.42])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)

                            
    for m, mod in enumerate(modelos):
        if mod == 'model2':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE)
        for l,line in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                axes[l].text(xtextpos, ytextpos, modstr,
                                            color = mod_color,  
                                            horizontalalignment='right',
                                            verticalalignment='top',
                                            fontsize=fontsize,
                                            transform=axes[l].transAxes)
                ytextpos = ytextpos -0.052
            if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                if plot_smoothed:
                    axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM_prese_subset.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM_presen_subset.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
def plot_models_and_inp_finalpaperfig(convolve, Rcrit, line_column, modelos, hb_df, cont_df,
                                      my_model_path, figmod_path, figrt_path, fort_paths, results_path,
                                      writename = True, plot_CH3CN = False, plot_col = True,
                                      cont_modelplot = 'model2',
                                      plot_opacity = False, D_Mpc = 3.5, source_rad=0, fortcomp=False):
    """ 
        Plots nLTE models for the final version of the paper
        convolve = False gets the model data without convolving by the beam
        fortcomp = False avoids fortran compilation to convolve by beam the modelled data (need the compiled .31 files)
    """
    distance_pc = D_Mpc*1e6
    cont_Tex_N_X_fig = True
    cont_Tex_N_X_fig_BIG = True
    line_profiles = True
    line_profiles_subset = True
    line_ratios = True
    line_ratios_BIG = True
    if convolve:
        convstr = ''
    else:
        convstr = '_noconv'
    print(f'source rad first: {source_rad:1.1f}')
    mykeys = list(line_column.keys())
    plot_only_cont = [cont_modelplot]#['model7']
    plot_corr_cols = False
    plot_corr_abun = True
    plot_smoothed = False
    if 'plot_T' not in list(line_column.keys()):
        mykeys.insert(1, 'plot_T')
        line_column['plot_T'] = []
        if plot_col:
            mykeys.insert(2, 'plot_col')
            line_column['plot_col'] = []
        else:
            mykeys.insert(2, 'plot_dens')
            line_column['plot_dens'] = []
        mykeys.insert(3, 'plot_x')
        line_column['plot_x'] = []
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    else:
        # Already ran, getting problems with ordered dict
        if plot_col:
            mykeys = ['plot_conts', 'plot_T', 'plot_col', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        else:
            mykeys = ['plot_conts', 'plot_T', 'plot_dens', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM'],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM'],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM'],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM'],
                   }
    
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
        hb_df['ratio_'+ratio+'_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']**2))**2)
        hb_df['ratio_'+ratio+'_beam345'] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']
        hb_df['ratio_'+ratio+'_beam345_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']**2))**2)

    # Continuum, Tex, logN and X profile figure
    if cont_Tex_N_X_fig:
        figsize = 20
        tstring = 'Tex_SM_ave_ring' 
        colstring = 'Col_SM_ave_ring'
        naxis = 2
        maxis = 2
        labelsize = 28
        ticksize = 22
        fontsize = 18
        contms = 8
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)  
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        axes=[]
        mykeys_conts = ['plot_conts', 'plot_T', 'plot_col', 'plot_x']
        
        for l,line in enumerate(mykeys_conts):
            axes.append(fig.add_subplot(gs1[l]))
            if line == 'plot_conts':
                axes[l].set_ylim([0.05, 20])  
                for i,row in cont_df.iterrows():
                    axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam'], 
                                                yerr=row['F235GHz_mjy_beam_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='k',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)                
                    axes[l].text(0.95, 0.95-0.045, r'Cont. 235 GHz',
                                    color = 'k',
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                        color = '0.5',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                    axes[l].errorbar(row['dist'], row['F345GHz_mjy_beam'], 
                                                yerr=row['F345GHz_mjy_beam_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.5',
                                                markeredgecolor='0.5', markeredgewidth=0.8,
                                                ecolor='0.5',
                                                color = '0.5',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                    if plot_smoothed:
                        axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam345'], 
                                                yerr=row['F235GHz_mjy_beam345_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.65',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                        axes[l].plot(row['dist'], row['F235GHz_mjy_beam345'],  linestyle='',
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.65',
                                                markeredgecolor='k',
                                                zorder=2)
                        axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                        color = '0.5',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                        axes[l].text(0.95,  0.95-0.045, r'Cont. 235 GHz',
                                        color = 'k',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                    axes[l].set_yscale('log')
                    axes[l].set_ylabel(r'$\text{Flux density}\:(\text{mJy}\:\text{beam}^{-1})$', fontsize=labelsize)
                    axes[l].yaxis.set_major_formatter(ScalarFormatter())
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <2:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        save_name = ''
        ytext = 0.95
        ytext2 = 0.95
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad=source_rad, read_only=fortcomp)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
            obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
            model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
            mtau100 = get_tau100(my_model_path, modelo[0])
            total_NH2 = np.nansum(10**logNH2_profile)
            total_NH2_corr = np.nansum(10**logNH2_profile_corr)
            for l,line in enumerate(mykeys_conts):
                if line == 'plot_conts':
                    axes[l].set_ylim([0.05, 20])  
                    if mod in plot_only_cont: # Plotting only one cont lines. Too complicated Fig.
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)), linewidth=2.0)
                        if plot_smoothed:
                            axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)), linewidth=2.0)
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '-', zorder=mzord)
                    elif len(plot_only_cont)<1:
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)))
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)))
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '-', zorder=mzord)
                        if plot_smoothed:
                            axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= (0, (5, 10)), zorder=mzord)
                elif line == 'plot_T':
                    if writename:
                        if 'model' in mod:
                            modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                        else:
                            modstr = mod
                        qval = f'{np.float(qprof):1.1f}'
                        qstr = r'$q='+qval+'$'
                        Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                        Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                        axes[l].text(0.95, ytext, modstr+':  '+Lstr +' \; '+Nstr+' \; '+qstr,
                                    color = mod_color,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    ytext = ytext -0.045
                    axes[l].set_ylabel(r'T$_{\text{dust}}$ (K)', fontsize = labelsize)#, labelpad=12)
                    for i,row in obs_df.iterrows():
                        if i <= 2 or i>= 10:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                        uplims = True,
                                                        yerr=65,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                        yerr=row[tstring+'_err'],
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        if plot_CH3CN:
                            if row['Tex_err_CH3CN'] < 0:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                        uplims = True,
                                                        yerr=200,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='b',
                                                        markeredgecolor='b', markeredgewidth=0.8,
                                                        ecolor='b',
                                                        color = 'b',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                        lolims = True,
                                                        yerr=200,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='b',
                                                        markeredgecolor='b', markeredgewidth=0.8,
                                                        ecolor='b',
                                                        color = 'b',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                            else:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'], 
                                                            yerr=row['Tex_err_CH3CN'],
                                                            marker='o', markersize=contms,
                                                            markerfacecolor='b',
                                                            markeredgecolor='b', markeredgewidth=0.8,
                                                            ecolor='b',
                                                            color = 'b',
                                                            elinewidth= 0.7,
                                                            barsabove= True,
                                                            zorder=1)
                    
                    axes[l].plot(rpc_profile, td_profile, color=mod_color, zorder=mzord)
                elif line == 'plot_dens':
                    axes[l].set_ylabel(r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', fontsize = labelsize)#, labelpad=12)
                    axes[l].plot(rpc_profile[1:], nh2_profile[1:], color=mod_color, zorder=mzord)
                    axes[l].set_yscale('log')
                elif line == 'plot_col':
                    
                    for i, row in obs_df.iterrows():
                        if row[colstring+'_err']>10:
                            col_err = (10**(row[colstring+'_err']+0.75))*(1/np.log(10))/(10**row[colstring])
                        else:
                            col_err = row[colstring+'_err']
                        if i>=10:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                        uplims = True,
                                                        yerr=0.25,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                        yerr=col_err,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                    ytext2 = ytext2 -0.045
                    axes[l].set_ylabel(r'$\log{N(\text{HC}_{3}\text{N})}$ (cm$^{-2}$)', fontsize = labelsize)#, labelpad=12)
                    if plot_corr_cols:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile_corr[1:], color=mod_color, zorder=mzord)
                    else:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile[1:], color=mod_color, zorder=mzord)
                elif line == 'plot_x':
                    axes[l].set_ylabel(r'$X$ (HC$_{3}$N)', fontsize = labelsize)#, labelpad=12)
                    if plot_corr_abun:
                        x_profile_corr = np.array(x_profile)*(10**logNH2_profile)/(10**logNH2_profile_corr)
                        axes[l].plot(rpc_profile[1:], x_profile_corr[1:], color=mod_color, zorder=mzord)
                    else:
                        axes[l].plot(rpc_profile[1:], x_profile[1:], color=mod_color, zorder=mzord)
                    axes[l].set_yscale('log')
            for l,line in enumerate(mykeys_conts):
                axes[l].tick_params(which='both',
                        labelright=False)
        if plot_opacity:
            # Plotting opacity
            gs2 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
            gs2.update(wspace = 0.23, hspace=0.23, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            axtau = fig.add_subplot(gs2[-1])
            for m, mod in enumerate(modelos):
                mod_name = modelos[mod][0]
                mod_color = modelos[mod][3]
                mod_taudust = pd.read_csv(my_model_path+'/'+mod_name+'_.taudust', delim_whitespace= True)
                mod_taudust.columns = ['lambda_um', 'taudust']
                axtau.plot(mod_taudust['lambda_um'], mod_taudust['taudust'], color=mod_color)
                minor_locator = AutoMinorLocator(2)
                axtau.set_xlim([0.0, 100])
                axtau.tick_params(direction='in')
                axtau.tick_params(axis="both", which='major', length=8)
                axtau.tick_params(axis="both", which='minor', length=4)
                axtau.xaxis.set_tick_params(which='both', top ='on')
                axtau.yaxis.set_tick_params(which='both', right='on', labelright='off')
                axtau.tick_params(axis='both', which='major', labelsize=ticksize)
                axtau.xaxis.set_minor_locator(minor_locator)
                axtau.tick_params(labelleft=True,
                            labelright=False)
                axtau.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize = labelsize)
                axtau.set_ylabel(r'$\tau$', fontsize = labelsize)
        D_Mpc = 3.5
        beam_size = 0.020/2 #arcsec
        xstart = 0.05
        ypos = 0.08
        beam_size_pc = u_conversion.lin_size(D_Mpc, beam_size).to(u.pc).value
        axes[0].hlines(ypos, xmin=xstart, xmax=xstart+beam_size_pc, color='k', linestyle='-', lw=1.2)
        axes[0].annotate('FWHM/2', xy=((xstart+beam_size_pc)/2,ypos), xytext=((xstart+beam_size_pc)/2+0.027,ypos+0.015), weight='bold',
                            fontsize=fontsize, color='k',
                            horizontalalignment='center',
                            verticalalignment='center',)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_conts_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_conts_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # Continuum, Tex, logN and X profile figure big labels
    if cont_Tex_N_X_fig_BIG:
        figsize = 20
        tstring = 'Tex_SM_ave_ring'
        colstring = 'Col_SM_ave_ring'
        naxis = 2
        maxis = 2
        labelsize = 35
        ticksize = 28
        fontsize = 25
        contms = 10
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05)#, left=0.05, right=0.80)
        axes=[]
        mykeys_conts = ['plot_conts', 'plot_T', 'plot_col', 'plot_x']
        for l,line in enumerate(mykeys_conts):
            axes.append(fig.add_subplot(gs1[l]))
            if line == 'plot_conts':
                axes[l].set_ylim([0.05, 20])  
                for i,row in cont_df.iterrows():
                    axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam'], 
                                                yerr=row['F235GHz_mjy_beam_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='k',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                    axes[l].text(0.95, 0.95-0.045, r'Cont. 235 GHz',
                                    color = 'k',
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                        color = '0.5',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                    axes[l].errorbar(row['dist'], row['F345GHz_mjy_beam'], 
                                                yerr=row['F345GHz_mjy_beam_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.5',
                                                markeredgecolor='0.5', markeredgewidth=0.8,
                                                ecolor='0.5',
                                                color = '0.5',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                    if plot_smoothed:
                        axes[l].errorbar(row['dist'], row['F235GHz_mjy_beam345'], 
                                                yerr=row['F235GHz_mjy_beam345_err'],
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.65',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                        axes[l].plot(row['dist'], row['F235GHz_mjy_beam345'],  linestyle='',
                                                marker='o', markersize=contms,
                                                markerfacecolor='0.65',
                                                markeredgecolor='k',
                                                zorder=2)
                        axes[l].text(0.95, 0.95, r'Cont. 345 GHz',
                                        color = '0.5',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                        axes[l].text(0.95,  0.95-0.045, r'Cont. 235 GHz',
                                        color = 'k',
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        fontsize=fontsize,
                                        transform=axes[l].transAxes)
                    axes[l].set_yscale('log')
                    axes[l].set_ylabel(r'$\text{Flux density}\:(\text{mJy}\:\text{beam}^{-1})$', fontsize=labelsize)
                    axes[l].yaxis.set_major_formatter(ScalarFormatter())
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <2:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        save_name = ''
        ytext = 0.95
        ytext2 = 0.95
        ytext3 = 0.95
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad=source_rad)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
            obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
            model_lum = lum_from_dustmod(modelo[1], my_model_path, distance_pc, rout_model_hc3n_pc)
            mtau100 = get_tau100(my_model_path, modelo[0])
            total_NH2 = np.nansum(10**logNH2_profile)
            total_NH2_corr = np.nansum(10**logNH2_profile_corr)
            for l,line in enumerate(mykeys_conts):
                if line == 'plot_conts':
                    axes[l].set_ylim([0.05, 20])  
                    if mod in plot_only_cont: # Plotting only one cont lines. Too complicated Fig.
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)), linewidth=2.0)
                        if plot_smoothed:
                            axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam345'], color=mod_color, zorder=mzord)
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)), linewidth=2.0)
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '-', zorder=mzord)
                    elif len(plot_only_cont)<1:
                        axes[l].plot(mdust[0],mdust['F235GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)))
                        axes[l].plot(mdust[0],mdust['F345GHz_mjy_beam'], color=mod_color, zorder=mzord, linestyle= (0, (5, 10)))
                        axes[l].plot(m_molec[0], m_molec['F235GHz_mjy_beam'], color=mod_color, linestyle= '-', zorder=mzord)
                        if plot_smoothed:
                            axes[l].plot(m_molec345[0], m_molec345['F235GHz_mjy_beam345'], color=mod_color, linestyle= (0, (5, 10)), zorder=mzord)
                elif line == 'plot_T':
                    if writename:
                        if 'model' in mod:
                            modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                        else:
                            modstr = mod
                        qval = f'{np.float(qprof):1.1f}'
                        qstr = r'$q='+qval+'$'
                        #Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2)}'+r'\text{cm}^{-2}$'
                        Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                        Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                        axes[l].text(0.95, ytext, modstr+':  '+Lstr +' \; '+qstr,
                                    color = mod_color,
                                    horizontalalignment='right',
                                    verticalalignment='top',
                                    fontsize=fontsize,
                                    transform=axes[l].transAxes)
                    ytext = ytext -0.045
                    axes[l].set_ylabel(r'T$_{\text{dust}}$ (K)', fontsize = labelsize)#, labelpad=12)
                    for i,row in obs_df.iterrows():
                        if i <= 2 or i>= 10:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                        uplims = True,
                                                        yerr=65,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[tstring], 
                                                        yerr=row[tstring+'_err'],
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        if plot_CH3CN:
                            if row['Tex_err_CH3CN'] < 0:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                        uplims = True,
                                                        yerr=200,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='b',
                                                        markeredgecolor='b', markeredgewidth=0.8,
                                                        ecolor='b',
                                                        color = 'b',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'],
                                                        lolims = True,
                                                        yerr=200,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='b',
                                                        markeredgecolor='b', markeredgewidth=0.8,
                                                        ecolor='b',
                                                        color = 'b',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                            else:
                                axes[l].errorbar(row['dist_ring_pc'], row['Tex_CH3CN'], 
                                                            yerr=row['Tex_err_CH3CN'],
                                                            marker='o', markersize=contms,
                                                            markerfacecolor='b',
                                                            markeredgecolor='b', markeredgewidth=0.8,
                                                            ecolor='b',
                                                            color = 'b',
                                                            elinewidth= 0.7,
                                                            barsabove= True,
                                                            zorder=1)
                    
                    axes[l].plot(rpc_profile, td_profile, color=mod_color, zorder=mzord)
                elif line == 'plot_dens':
                    axes[l].set_ylabel(r'$n_{\text{H}_{2}}$ (cm$^{-3}$)', fontsize = labelsize)#, labelpad=12)
                    axes[l].plot(rpc_profile[1:], nh2_profile[1:], color=mod_color, zorder=mzord)
                    axes[l].set_yscale('log')
                elif line == 'plot_col':
                    for i, row in obs_df.iterrows():
                        if row[colstring+'_err']>10:
                            col_err = (10**(row[colstring+'_err']+0.75))*(1/np.log(10))/(10**row[colstring])
                        else:
                            col_err = row[colstring+'_err']
                        if i>=10:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                        uplims = True,
                                                        yerr=0.25,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                        else:
                            axes[l].errorbar(row['dist_ring_pc'], row[colstring], 
                                                        yerr=col_err,
                                                        marker='o', markersize=contms,
                                                        markerfacecolor='k',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                    qval = f'{np.float(qprof):1.1f}'
                    qstr = r'$q='+qval+'$'
                    Nstr = r'$N_{\text{H}_2}='+f'{utiles.latex_float(total_NH2_corr)}'+r'\text{cm}^{-2}$'
                    Lstr = r'$L_\text{IR}='+f'{utiles.latex_float(model_lum)}'+r'\text{L}_{\odot}$'
                    axes[l].text(0.95, ytext3, modstr+':  '+Nstr,
                                color = mod_color,
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                    ytext3 = ytext3 -0.045
                    axes[l].set_ylabel(r'$\log{N(\text{HC}_{3}\text{N})}$ (cm$^{-2}$)', fontsize = labelsize)#, labelpad=12)
                    if plot_corr_cols:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile_corr[1:], color=mod_color, zorder=mzord)
                    else:
                        axes[l].plot(rpc_profile[1:], logNHC3N_profile[1:], color=mod_color, zorder=mzord)
                elif line == 'plot_x':
                    axes[l].set_ylabel(r'$X$ (HC$_{3}$N)', fontsize = labelsize, labelpad=-3)
                    if plot_corr_abun:
                        x_profile_corr = np.array(x_profile)*(10**logNH2_profile)/(10**logNH2_profile_corr)
                        axes[l].plot(rpc_profile[1:], x_profile_corr[1:], color=mod_color, zorder=mzord)
                    else:
                        axes[l].plot(rpc_profile[1:], x_profile[1:], color=mod_color, zorder=mzord)
                    axes[l].set_yscale('log')
            for l,line in enumerate(mykeys_conts):
                axes[l].tick_params(which='both',
                        labelright=False)
        if plot_opacity:
            # Plotting opacity
            gs2 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
            gs2.update(wspace = 0.23, hspace=0.23, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            axtau = fig.add_subplot(gs2[-1])
            for m, mod in enumerate(modelos):
                mod_name = modelos[mod][0]
                mod_color = modelos[mod][3]
                mod_taudust = pd.read_csv(my_model_path+'/'+mod_name+'_.taudust', delim_whitespace= True)
                mod_taudust.columns = ['lambda_um', 'taudust']
                axtau.plot(mod_taudust['lambda_um'], mod_taudust['taudust'], color=mod_color)
                minor_locator = AutoMinorLocator(2)
                axtau.set_xlim([0.0, 100])
                axtau.tick_params(direction='in')
                axtau.tick_params(axis="both", which='major', length=8)
                axtau.tick_params(axis="both", which='minor', length=4)
                axtau.xaxis.set_tick_params(which='both', top ='on')
                axtau.yaxis.set_tick_params(which='both', right='on', labelright='off')
                axtau.tick_params(axis='both', which='major', labelsize=ticksize)
                axtau.xaxis.set_minor_locator(minor_locator)
                axtau.tick_params(labelleft=True,
                            labelright=False)
                axtau.set_xlabel(r'$\lambda$ ($\mu$m)', fontsize = labelsize)
                axtau.set_ylabel(r'$\tau$', fontsize = labelsize)
        D_Mpc = 3.5
        beam_size = 0.020/2 #arcsec
        xstart = 0.05
        ypos = 0.08
        beam_size_pc = u_conversion.lin_size(D_Mpc, beam_size).to(u.pc).value
        axes[0].hlines(ypos, xmin=xstart, xmax=xstart+beam_size_pc, color='k', linestyle='-', lw=1.2)
        axes[0].annotate('FWHM/2', xy=((xstart+beam_size_pc)/2,ypos), xytext=((xstart+beam_size_pc)/2+0.027,ypos+0.015), weight='bold',
                            fontsize=fontsize, color='k',
                            horizontalalignment='center',
                            verticalalignment='center',)
            
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_conts_big_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_conts_big_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # Line profiles figure
    if line_profiles:
        # Line profiles
        print('line profiles')
        plot_smoothed_lines = True
        figsize = 20
        naxis = 3
        maxis = 4
        labelsize = 18
        ticksize = 16
        fontsize = 14
        linems = 6
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'w'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)  
        if maxis ==3:
            gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 9
            mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                    'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                    'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                    'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
            xtextpos = 0.15
            ytextpos = 0.95
        else:
            gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 7
            mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                    'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                    'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                        'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']#, 'ratio_v6_v6v7']
            xtextpos = 0.90
            ytextpos = 0.85
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        for l,line in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
                for i,row in hb_df.iterrows():
                    if row['dist']<=line_column[line][0]:
                        ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                        if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                            hb_df.loc[i, line+'_uplim'] = True
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                                uplims=True,
                                                yerr=ysep,
                                                marker='o', markersize=linems,
                                                markerfacecolor=color_beam_orig,
                                                markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                                ecolor=color_beam_orig,
                                                color = color_beam_orig,
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']
                            hb_df.loc[i, line+'_uplim'] = False
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                            yerr=errplot,
                                            marker='o', markersize=linems,
                                            markerfacecolor=color_beam_orig,
                                            markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                            ecolor=color_beam_orig,
                                            color =color_beam_orig,
                                            elinewidth= 0.7,
                                            barsabove= True,
                                            zorder=2)
                        if plot_smoothed_lines:
                            if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                                axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                    uplims=True,
                                                    yerr=ysep,#3*row[line+'_mJy_kms_beam_345_errcont']*0.15,
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                    ecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    elinewidth= 0.7,
                                                    barsabove= True,
                                                    zorder=1)
                                axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                    linestyle='',
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    zorder=2)
                            else:
                                # Adding 10% error to the inner rings
                                if row['dist']<0.35:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                                else:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']
                                axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                yerr=errplot,
                                                marker='o', markersize=linems,
                                                markerfacecolor=facecolor_beam_345,
                                                markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                ecolor=color_beam_345,
                                                color =color_beam_345,
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                                axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                    linestyle='',
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    zorder=2) 
            
                axes[l].set_ylim(line_column[line][2])  
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
                axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                a = line_column[line][1].split(' ')[0]
                axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
            elif line == 'ratio_v6_v6v7':
                plot_ratio345 = False
                for i, row in hb_df.iterrows():
                    if row['v6=v7=1_26_2_25_-2_SM_uplim']:
                        continue
                    else:
                        axes[l].errorbar(row['dist'], row['ratio_v6_v6v7'], 
                                                    yerr=row['ratio_v6_v6v7_err'],
                                                    marker='o', markersize=linems,
                                                    markerfacecolor='k',
                                                    markeredgecolor='k', markeredgewidth=0.8,
                                                    ecolor='k',
                                                    color = 'k',
                                                    elinewidth= 0.7,
                                                    barsabove= True,
                                                    zorder=1)
                        if plot_ratio345:
                            axes[l].errorbar(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                                        yerr=row['ratio_v6_v6v7_beam345_err'],
                                                        marker='o', markersize=linems,
                                                        markerfacecolor='w',
                                                        markeredgecolor='k', markeredgewidth=0.8,
                                                        ecolor='k',
                                                        color = 'k',
                                                        elinewidth= 0.7,
                                                        barsabove= True,
                                                        zorder=1)
                            axes[l].plot(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                            linestyle='',
                                            marker='o', markersize=linems,
                                            markerfacecolor='w',
                                            markeredgecolor='k',
                                            color = 'k',
                                            zorder=2)
                        
                        axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                        axes[l].set_ylim([0.5, 4.7]) 
                        yminor_locator = AutoMinorLocator(2)
                        axes[l].yaxis.set_minor_locator(yminor_locator)
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        beam_size = 0.020/2 #arcsec
        xstart = 0.2
        ypos = 0.0
        beam_size_pc = u_conversion.lin_size(D_Mpc, beam_size).to(u.pc).value
        axes[0].hlines(ypos, xmin=xstart, xmax=xstart+beam_size_pc, color='k', linestyle='-', lw=1.2)
        axes[0].annotate('FWHM/2', xy=((xstart+beam_size_pc)/2,ypos), xytext=((xstart+beam_size_pc)/2+0.11,ypos+5.5), weight='bold',
                            fontsize=fontsize-4, color='k',
                            horizontalalignment='center',
                            verticalalignment='center',)
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad=source_rad, read_only=fortcomp)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
            for l,line in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                    ytextpos = ytextpos -0.052
                if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                    if plot_smoothed_lines:
                        axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
                
                elif line in ['ratio_v6_v6v7']:
                    mol_ratio = m_molec['v6=1_24_-1_23_1_SM_beam_orig']/m_molec['v6=v7=1_26_2_25_-2_SM_beam_orig']
                    axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM_prese'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # subset of Line profiles figure
    if line_profiles_subset:
        # line profiles
        print('line profiles subset')
        figsize = 20
        naxis = 3
        maxis = 2
        labelsize = 32
        ticksize = 24
        fontsize = 28
        linems = 6
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'w'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])   
        if maxis ==3:
            gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 9
            mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                    'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                    'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                    'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
            xtextpos = 0.15
            ytextpos = 0.95
        else:
            gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 2
            mykeys_flux = ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM',
                        'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM', 'v4=1_26_25_SM']
            xtextpos = 0.90
            ytextpos = 0.85
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        for l,line in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
                for i,row in hb_df.iterrows():
                    if row['dist']<=line_column[line][0]:
                        ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                        if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                            hb_df.loc[i, line+'_uplim'] = True
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                                uplims=True,
                                                yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                                marker='o', markersize=linems,
                                                markerfacecolor=color_beam_orig,
                                                markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                                ecolor=color_beam_orig,
                                                color = color_beam_orig,
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']
                            hb_df.loc[i, line+'_uplim'] = False
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                            yerr=errplot,
                                            marker='o', markersize=linems,
                                            markerfacecolor=color_beam_orig,
                                            markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                            ecolor=color_beam_orig,
                                            color =color_beam_orig,
                                            elinewidth= 0.7,
                                            barsabove= True,
                                            zorder=2)
                        if plot_smoothed:
                            if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                                axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                    uplims=True,
                                                    yerr=ysep,
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                    ecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    elinewidth= 0.7,
                                                    barsabove= True,
                                                    zorder=1)
                                axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                    linestyle='',
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    zorder=2)
                            else:
                                # Adding 10% error to the inner rings
                                if row['dist']<0.35:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                                else:
                                    errplot = row[line+'_mJy_kms_beam_345_errcont']
                                axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                yerr=errplot,
                                                marker='o', markersize=linems,
                                                markerfacecolor=facecolor_beam_345,
                                                markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                ecolor=color_beam_345,
                                                color =color_beam_345,
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                                axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                    linestyle='',
                                                    marker='o', markersize=linems,
                                                    markerfacecolor=facecolor_beam_345,
                                                    markeredgecolor=color_beam_345,
                                                    color = color_beam_345,
                                                    zorder=2) 
                axes[l].set_ylim(line_column[line][2])  
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
                axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                a = line_column[line][1].split(' ')[0]
                axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)                            
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad=source_rad, read_only=fortcomp)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
            for l,line in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                    ytextpos = ytextpos -0.052
                if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                    if plot_smoothed:
                        axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM_prese_subset'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM_presen_subset'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # Line ratios figure   
    if line_ratios:
        # Line ratios
        figsize = 10
        naxis = 2
        maxis = 1
        labelsize = 28
        ticksize = 20
        fontsize = 18
        contms = 8
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)   
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = -1
        mykeys_flux = ['v71_v61_2423', 'v6_v6v7']
        xtextpos = 0.15
        ytextpos = 0.95
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        # Making upper lim ratios
        ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM', r'$v_{6}=1/v_{6}=v_{7}=1$', [1.1, 4.7]],
                    'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM', r'$v_{7}=1/v_{6}=1$', [0.6, 4.7]],
                    'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (26-25)$', [0.5, 5.7]],
                    'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', r'$v=0/v_{7}=1 \:\: (26-25)$', [0.0, 8.7]],
                    }
        
        for r, ratio in enumerate(ratio_lines):
            hb_df['ratio_'+ratio+'_uplim'] = False
        for i, row in hb_df.iterrows():
            for r, ratio in enumerate(ratio_lines):
                uplim1 = 3*row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']
                uplim2 = 3*row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
                if uplim1 or uplim2:
                    hb_df.loc[i, 'ratio_'+ratio+'_uplim'] = True
        
        for l,ratio in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            plot_ratio345 = False
            for i, row in hb_df.iterrows():
                if row['ratio_'+ratio+'_uplim']: # At least one of the lines is an upper limit
                    continue
                else:
                    axes[l].errorbar(row['dist'], row['ratio_'+ratio], 
                                                yerr=row['ratio_'+ratio+'_err'],
                                                marker='o', markersize=linems,
                                                markerfacecolor='k',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                    if plot_ratio345:
                        axes[l].errorbar(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                                    yerr=row['ratio_'+ratio+'_beam345_err'],
                                                    marker='o', markersize=linems,
                                                    markerfacecolor='w',
                                                    markeredgecolor='k', markeredgewidth=0.8,
                                                    ecolor='k',
                                                    color = 'k',
                                                    elinewidth= 0.7,
                                                    barsabove= True,
                                                    zorder=1)
                        axes[l].plot(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                        linestyle='',
                                        marker='o', markersize=linems,
                                        markerfacecolor='w',
                                        markeredgecolor='k',
                                        color = 'k',
                                        zorder=2)
                    
                    axes[l].set_ylabel(ratio_lines[ratio][2], fontsize = labelsize)
                    axes[l].set_ylim(ratio_lines[ratio][3]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
                    
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        # Model ratios
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, read_only=fortcomp)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)

            for l,ratio in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=fontsize,
                                                transform=axes[l].transAxes)
                mol_ratio = m_molec[ratio_lines[ratio][0]+'_beam_orig']/m_molec[ratio_lines[ratio][1]+'_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
                if plot_ratio345:
                    mol_ratio345 = m_molec345[ratio_lines[ratio][0]+'_beam_345']/m_molec345[ratio_lines[ratio][1]+'_beam_345']
                    axes[l].plot(m_molec345[0], mol_ratio345, color=mod_color, linestyle= '--', zorder=2)
            ytextpos = ytextpos -0.045
        
        D_Mpc = 3.5
        beam_size = 0.020/2 #arcsec
        xstart = 1.15
        ypos = 0.85
        beam_size_pc = u_conversion.lin_size(D_Mpc, beam_size).to(u.pc).value
        axes[0].hlines(ypos, xmin=xstart, xmax=xstart+beam_size_pc, color='k', linestyle='-', lw=1.2)
        axes[0].annotate('FWHM/2', xy=((xstart+beam_size_pc)/2+0.57,ypos), xytext=((xstart+beam_size_pc)/2+0.57,ypos+0.1), weight='bold',
                            fontsize=fontsize-4, color='k',
                            horizontalalignment='center',
                            verticalalignment='center',)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_ratios2_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_ratios2_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # Line ratios figure big labels
    if line_ratios_BIG:
        # Line ratios
        figsize = 10
        naxis = 2
        maxis = 1
        labelsize = 35
        tickfontsize = 28
        fontsize = 25
        modelname_fontsize = 30
        contms = 10
        linems = 10
        axcolor = 'k'
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'None'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])    
        gs1.update(wspace = 0.125, hspace=0.0, top=0.95, bottom = 0.05)#, left=0.05, right=0.80)
        xlabel_ind = -1
        mykeys_flux = ['v71_v61_2423', 'v6_v6v7']
        xtextpos = 0.18
        ytextpos = 0.95
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        # Making upper lim ratios
        ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM', r'$v_{6}=1/v_{6}=v_{7}=1$', [1.1, 4.7]],
                    'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM', r'$v_{7}=1/v_{6}=1$', [0.6, 4.7]],
                    'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM', r'$v_{7}=1/v_{6}=1 \:\: (26-25)$', [0.5, 5.7]],
                    'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM', r'$v=0/v_{7}=1 \:\: (26-25)$', [0.0, 8.7]],
                    }
        for r, ratio in enumerate(ratio_lines):
            hb_df['ratio_'+ratio+'_uplim'] = False
        for i, row in hb_df.iterrows():
            for r, ratio in enumerate(ratio_lines):
                uplim1 = 3*row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']
                uplim2 = 3*row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont'] > row[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
                if uplim1 or uplim2:
                    hb_df.loc[i, 'ratio_'+ratio+'_uplim'] = True
        for l,ratio in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            plot_ratio345 = False
            for i, row in hb_df.iterrows():
                
                if row['ratio_'+ratio+'_uplim']: # At least one of the lines is an upper limit
                    continue
                else:
                    axes[l].errorbar(row['dist'], row['ratio_'+ratio], 
                                                yerr=row['ratio_'+ratio+'_err'],
                                                marker='o', markersize=linems,
                                                markerfacecolor='k',
                                                markeredgecolor='k', markeredgewidth=0.8,
                                                ecolor='k',
                                                color = 'k',
                                                elinewidth= 0.7,
                                                barsabove= True,
                                                zorder=1)
                    if plot_ratio345:
                        axes[l].errorbar(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                                    yerr=row['ratio_'+ratio+'_beam345_err'],
                                                    marker='o', markersize=linems,
                                                    markerfacecolor='w',
                                                    markeredgecolor='k', markeredgewidth=0.8,
                                                    ecolor='k',
                                                    color = 'k',
                                                    elinewidth= 0.7,
                                                    barsabove= True,
                                                    zorder=1)
                        axes[l].plot(row['dist'], row['ratio_'+ratio+'_beam345'], 
                                        linestyle='',
                                        marker='o', markersize=linems,
                                        markerfacecolor='w',
                                        markeredgecolor='k',
                                        color = 'k',
                                        zorder=2)
                    
                    axes[l].set_ylabel(ratio_lines[ratio][2], fontsize = labelsize)
                    axes[l].set_ylim(ratio_lines[ratio][3]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=tickfontsize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                        labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                        labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
        # Model ratios
        for m, mod in enumerate(modelos):
            if mod == 'model2':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            if convolve:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, read_only=fortcomp)
            else:
                mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)

            for l,ratio in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    axes[l].text(xtextpos, ytextpos, modstr,
                                                color = mod_color,  
                                                horizontalalignment='right',
                                                verticalalignment='top',
                                                fontsize=modelname_fontsize,
                                                transform=axes[l].transAxes)
                mol_ratio = m_molec[ratio_lines[ratio][0]+'_beam_orig']/m_molec[ratio_lines[ratio][1]+'_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
                if plot_ratio345:
                    mol_ratio345 = m_molec345[ratio_lines[ratio][0]+'_beam_345']/m_molec345[ratio_lines[ratio][1]+'_beam_345']
                    axes[l].plot(m_molec345[0], mol_ratio345, color=mod_color, linestyle= '--', zorder=2)
            ytextpos = ytextpos -0.049
        D_Mpc = 3.5
        beam_size = 0.020/2 #arcsec
        xstart = 1.15
        ypos = 0.85
        beam_size_pc = u_conversion.lin_size(D_Mpc, beam_size).to(u.pc).value
        axes[0].hlines(ypos, xmin=xstart, xmax=xstart+beam_size_pc, color='k', linestyle='-', lw=1.2)
        axes[0].annotate('FWHM/2', xy=((xstart+beam_size_pc)/2+0.57,ypos), xytext=((xstart+beam_size_pc)/2+0.57,ypos+0.11), weight='bold',
                            fontsize=fontsize, color='k',
                            horizontalalignment='center',
                            verticalalignment='center',)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_ratios2_big_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_ratios2_big_SM_papfin'+convstr+'.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()

def plot_models_and_inp_abscompfig(Rcrit, line_column, modelos, hb_df, cont_df, my_model_path, figmod_path, figrt_path, fort_paths, results_path, writename = True, plot_CH3CN = False, plot_col = True, plot_opacity = False, distance_pc = 3.5e6):
    source_rad = 1.5
    mykeys = list(line_column.keys())
    plot_only_cont = []#['model2']
    plot_corr_cols = False
    plot_corr_abun = True
    if 'plot_T' not in list(line_column.keys()):
        mykeys.insert(1, 'plot_T')
        line_column['plot_T'] = []
        if plot_col:
            mykeys.insert(2, 'plot_col')
            line_column['plot_col'] = []
        else:
            mykeys.insert(2, 'plot_dens')
            line_column['plot_dens'] = []
        mykeys.insert(3, 'plot_x')
        line_column['plot_x'] = []
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')
    else:
        # Already ran, getting problems with ordered dict
        if plot_col:
            mykeys = ['plot_conts', 'plot_T', 'plot_col', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        else:
            mykeys = ['plot_conts', 'plot_T', 'plot_dens', 'plot_x',
             'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
             'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
             'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
             'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM']
        if plot_opacity == False:
            mykeys.append('ratio_v6_v6v7')

    ratio_lines = {'v6_v6v7': ['v6=1_24_-1_23_1_SM', 'v6=v7=1_26_2_25_-2_SM'],
                   'v71_v61_2423': ['v7=1_24_1_23_-1_SM', 'v6=1_24_-1_23_1_SM'],
                   'v71_v61_2625': ['v7=1_26_1_25_-1_SM', 'v6=1_26_-1_25_1_SM'],
                   'v0_v71_2625': ['v=0_26_25_SM', 'v7=1_26_1_25_-1_SM'],
                   }
    for r, ratio in enumerate(ratio_lines):
        hb_df['ratio_'+ratio] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']
        hb_df['ratio_'+ratio+'_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_orig']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_orig']**2))**2)
        hb_df['ratio_'+ratio+'_beam345'] = hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']
        hb_df['ratio_'+ratio+'_beam345_err'] = np.sqrt((hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345_errcont']/hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345'])**2+(hb_df[ratio_lines[ratio][0]+'_mJy_kms_beam_345']*hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345_errcont']/(hb_df[ratio_lines[ratio][1]+'_mJy_kms_beam_345']**2))**2)
    plot_lineabs = False
    if plot_lineabs:
        # line profiles
        figsize = 20
        naxis = 3
        maxis = 4
        labelsize = 18
        ticksize = 16
        fontsize = 14
        linems = 6
        color_beam_orig = 'k'
        color_beam_345 = 'k'
        facecolor_beam_345 = 'w'
        fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
        gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])   
        if maxis ==3:
            gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 9
            mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                       'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                       'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                       'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
            xtextpos = 0.15
            ytextpos = 0.95
        else:
            gs1.update(wspace = 0.20, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
            xlabel_ind = 8
            mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                       'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                       'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                        'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
            xtextpos = 0.90
            ytextpos = 0.85
    
        axes=[]
        ytext = 0.95
        ytext2 = 0.95
        save_name = ''
        
                
        for l,line in enumerate(mykeys_flux):
            axes.append(fig.add_subplot(gs1[l]))
            if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
                for i,row in hb_df.iterrows():
                    if row['dist']<=line_column[line][0]:
                        ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04
                        if 3*row[line+'_mJy_kms_beam_orig_errcont'] > row[line+'_mJy_kms_beam_orig']:
                            hb_df.loc[i, line+'_uplim'] = True
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_orig_errcont'], 
                                                 uplims=True,
                                                 yerr=ysep,#3*row[line+'_mJy_kms_beam_orig_err']*0.15,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=color_beam_orig,
                                                 markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                                 ecolor=color_beam_orig,
                                                 color = color_beam_orig,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_orig_errcont']
                            hb_df.loc[i, line+'_uplim'] = False
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_orig'], 
                                             yerr=errplot,
                                             marker='o', markersize=linems,
                                             markerfacecolor=color_beam_orig,
                                             markeredgecolor=color_beam_orig, markeredgewidth=0.8,
                                             ecolor=color_beam_orig,
                                             color =color_beam_orig,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=2)
                        if 3*row[line+'_mJy_kms_beam_345_errcont'] > row[line+'_mJy_kms_beam_345']:
                            axes[l].errorbar(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 uplims=True,
                                                 yerr=ysep,#3*row[line+'_mJy_kms_beam_345_errcont']*0.15,
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                                 ecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 elinewidth= 0.7,
                                                 barsabove= True,
                                                 zorder=1)
                            axes[l].plot(row['dist'], 3*row[line+'_mJy_kms_beam_345_errcont'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2)
                        else:
                            # Adding 10% error to the inner rings
                            if row['dist']<0.35:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']*1.25
                            else:
                                errplot = row[line+'_mJy_kms_beam_345_errcont']
                            axes[l].errorbar(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                             yerr=errplot,
                                             marker='o', markersize=linems,
                                             markerfacecolor=facecolor_beam_345,
                                             markeredgecolor=color_beam_345, markeredgewidth=0.8,
                                             ecolor=color_beam_345,
                                             color =color_beam_345,
                                             elinewidth= 0.7,
                                             barsabove= True,
                                             zorder=1)
                            axes[l].plot(row['dist'], row[line+'_mJy_kms_beam_345'], 
                                                 linestyle='',
                                                 marker='o', markersize=linems,
                                                 markerfacecolor=facecolor_beam_345,
                                                 markeredgecolor=color_beam_345,
                                                 color = color_beam_345,
                                                 zorder=2) 
            
                axes[l].set_ylim(line_column[line][2])  
                yminor_locator = AutoMinorLocator(2)
                axes[l].yaxis.set_minor_locator(yminor_locator)
                axes[l].text(0.9, 0.95, line_column[line][1].split('$')[-1],
                                horizontalalignment='right',
                                verticalalignment='top',
                                fontsize=fontsize,
                                transform=axes[l].transAxes)
                a = line_column[line][1].split(' ')[0]
                axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1}\:\,\text{beam}^{-1})$', fontsize=labelsize)
            elif line == 'ratio_v6_v6v7':
                plot_ratio345 = False
                for i, row in hb_df.iterrows():
                    if row['v6=v7=1_26_2_25_-2_SM_uplim']:
                        continue
                    else:
                        axes[l].errorbar(row['dist'], row['ratio_v6_v6v7'], 
                                                     yerr=row['ratio_v6_v6v7_err'],
                                                     marker='o', markersize=linems,
                                                     markerfacecolor='k',
                                                     markeredgecolor='k', markeredgewidth=0.8,
                                                     ecolor='k',
                                                     color = 'k',
                                                     elinewidth= 0.7,
                                                     barsabove= True,
                                                     zorder=1)
                        if plot_ratio345:
                            axes[l].errorbar(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                                         yerr=row['ratio_v6_v6v7_beam345_err'],
                                                         marker='o', markersize=linems,
                                                         markerfacecolor='w',
                                                         markeredgecolor='k', markeredgewidth=0.8,
                                                         ecolor='k',
                                                         color = 'k',
                                                         elinewidth= 0.7,
                                                         barsabove= True,
                                                         zorder=1)
                            axes[l].plot(row['dist'], row['ratio_v6_v6v7_beam345'], 
                                             linestyle='',
                                             marker='o', markersize=linems,
                                             markerfacecolor='w',
                                             markeredgecolor='k',
                                             color = 'k',
                                             zorder=2)
                        
                        axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                        axes[l].set_ylim([0.5, 4.7]) 
                        yminor_locator = AutoMinorLocator(2)
                        axes[l].yaxis.set_minor_locator(yminor_locator)
            minor_locator = AutoMinorLocator(2)
            axes[l].set_xlim([0.0, 1.42])
            axes[l].tick_params(direction='in')
            axes[l].tick_params(axis="both", which='major', length=8)
            axes[l].tick_params(axis="both", which='minor', length=4)
            axes[l].xaxis.set_tick_params(which='both', top ='on')
            axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
            axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
            axes[l].xaxis.set_minor_locator(minor_locator)
            axes[l].tick_params(labelleft=True,
                           labelright=False)
            if l <=xlabel_ind:
                axes[l].tick_params(
                           labelbottom=False)
            else:
                axes[l].set_xlabel(r'r (pc)', fontsize = labelsize)
    
                                
        for m, mod in enumerate(modelos):
            if mod == '0':
                mzord = 4
            else:
                mzord = 2
            save_name += mod+'_'
            modelo = modelos[mod]
            mod_color = modelo[3]
            factor_model_hc3n = modelo[4][0]
            factor_model_dust = modelo[4][1]
            factor_model_ff   = modelo[4][2]
            if len(modelo)<=5:
                LTE = True
            else:
                LTE = modelo[6]
            mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
            obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
            rhc3n = [i for i in range(len(x_profile)) if x_profile[i] < 1e-11]
            for l,line in enumerate(mykeys_flux):
                if l == 0:
                    if 'model' in mod:
                        modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                    else:
                        modstr = mod
                    ytextpos = ytextpos -0.052
                if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                    axes[l].plot(m_molec345[0], m_molec345[line+'_beam_345'], color=mod_color, linestyle= '--', zorder=mzord)
                    axes[l].plot(m_molec[0], m_molec[line+'_beam_orig'], color=mod_color, linestyle= '-', zorder=mzord)
                
                elif line in ['ratio_v6_v6v7']:
                    mol_ratio = m_molec['v6=1_24_-1_23_1_SM_beam_orig']/m_molec['v6=v7=1_26_2_25_-2_SM_beam_orig']
                    axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
        colorbar = True      
        if colorbar:
            import matplotlib.cm as cm
            ax = axes[0]
            pos = ax.get_position()
            # Horizontal colorbar
            a = np.array([[0,1]]) 
            sm = plt.cm.ScalarMappable(cmap=cm.gist_rainbow, norm=plt.Normalize(vmin=0, vmax=len(modelos)))
            sm._A = []
            axis = 'x'
            orien = 'horizontal'
            sep=-0.043
            width=0.01
            labelpad = -45
            ticksize=8
            framewidth=1
            tickwidth=1
            label = r'r(HC$_3$N)'
            tick_font = 14
            label_font = 14
            #sep = 0.05
            cb_axl = [pos.x0 + 0.02, pos.y0 + pos.height + sep,  np.round(pos.width,2)-2*0.02 , width] 
            cb_ax = fig.add_axes(cb_axl)
            cbar = fig.colorbar(sm, orientation=orien, cax=cb_ax)
            cbar.outline.set_linewidth(framewidth)
            cbar.ax.minorticks_off()
            cbar.set_label(label, labelpad=labelpad, fontsize=label_font)
            cbar.ax.tick_params(axis=axis, direction='in')
            cbar.ax.tick_params(labelsize=tick_font)
            cbar.ax.tick_params(length=ticksize, width=tickwidth)
            
            cbticks = [0, 5, 10, 15, 20, 26]
            cbar.set_ticks(cbticks)
            cblabels = [] # Rmax
            rpc_profile.append(1.5)
            for c,tick in enumerate(cbticks):
                tt = tick
                if tick == 26:
                    tt = 25
                
                cblabels.append(f'{rpc_profile[len(rpc_profile)-1-tt]:1.2f}')
            cbar.set_ticklabels(cblabels)
        if len(modelos) == 1:
            for m, mod in enumerate(modelos):
                save_name = modelo = modelos[mod][0]
                fig.savefig(figmod_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        else:
            fig.savefig(figrt_path+'NGC253_'+save_name+'_lines_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
        plt.close()
    # line profiles
    figsize = 20
    naxis = 3
    maxis = 4
    labelsize = 18
    ticksize = 16
    fontsize = 14
    linems = 6
    color_beam_orig = 'k'
    color_beam_345 = 'k'
    facecolor_beam_345 = 'w'
    fig = plt.figure(figsize=(figsize*naxis/maxis*1.15, figsize*0.85))
    gs1 = gridspec.GridSpec(maxis, naxis)#, width_ratios=[1,1,1,0.1], height_ratios=[1])   
    if maxis ==3:
        gs1.update(wspace = 0.15, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 9
        mykeys_flux = ['v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                   'v=0_26_25_SM', 'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.15
        ytextpos = 0.95
    else:
        gs1.update(wspace = 0.22, hspace=0.0, top=0.95, bottom = 0.05, left=0.05, right=0.80)
        xlabel_ind = 8
        mykeys_flux = ['v=0_26_25_SM', 'v7=1_24_1_23_-1_SM', 'v7=1_26_1_25_-1_SM', 'v6=1_24_-1_23_1_SM',
                   'v6=1_26_-1_25_1_SM', 'v7=2_24_0_23_0_SM', 'v7=2_26_0_25_0_SM', 
                   'v5=1_v7=3_26_1_0_25_-1_0_SM', 'v6=v7=1_26_2_25_-2_SM',
                    'v4=1_26_25_SM', 'v6=2_24_0_23_0_SM', 'ratio_v6_v6v7']
        xtextpos = 0.90
        ytextpos = 0.85
    axes=[]
    ytext = 0.95
    ytext2 = 0.95
    save_name = ''
    for l,line in enumerate(mykeys_flux):
        axes.append(fig.add_subplot(gs1[l]))
        if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']:   
            for i,row in hb_df.iterrows():
                if row['dist']<=line_column[line][0]:
                    ysep = (line_column[line][2][1]-line_column[line][2][0])*0.04  
            yminor_locator = AutoMinorLocator(2)
            axes[l].yaxis.set_minor_locator(yminor_locator)
            a = line_column[line][1].split(' ')[0]
            axes[l].set_ylabel(a+r'$\;(\text{mJy}\:\,\text{km}\,\,\text{s}^{-1})$', fontsize=labelsize)
        elif line == 'ratio_v6_v6v7':
                    axes[l].set_ylabel(r'$v_{6}=1/v_{6}=v_{7}=1$', fontsize = labelsize)
                    axes[l].set_ylim([0.5, 4.7]) 
                    yminor_locator = AutoMinorLocator(2)
                    axes[l].yaxis.set_minor_locator(yminor_locator)
        minor_locator = AutoMinorLocator(2)
        axes[l].set_xlim([0.0, 1.51])
        axes[l].tick_params(direction='in')
        axes[l].tick_params(axis="both", which='major', length=8)
        axes[l].tick_params(axis="both", which='minor', length=4)
        axes[l].xaxis.set_tick_params(which='both', top ='on')
        axes[l].yaxis.set_tick_params(which='both', right='on', labelright='off')
        axes[l].tick_params(axis='both', which='major', labelsize=ticksize)
        axes[l].xaxis.set_minor_locator(minor_locator)
        axes[l].tick_params(labelleft=True,
                       labelright=False)
        if l <=xlabel_ind:
            axes[l].tick_params(
                       labelbottom=False)
        else:
            axes[l].set_xlabel(r'Rmax(HC$_3$NC) (pc)', fontsize = labelsize)
    beam_orig = np.pi*0.022*0.020/(4*np.log(2))
    pc2_beam_orig = beam_orig*(distance_pc*np.pi/(180.0*3600.0))**2
    beam_345  = np.pi*0.028*0.034/(4*np.log(2))
    pc2_beam_345  = beam_345*(distance_pc*np.pi/(180.0*3600.0))**2
                            
    flux_dict = {}
    for l,line in enumerate(mykeys_flux):
        if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
                flux_dict[line+'_beam_orig'] = {'flux': [], 'rad': []}
                flux_dict[line+'_beam_345'] =  {'flux': [], 'rad': []}
    for m, mod in enumerate(modelos):
        if mod == '0':
            mzord = 4
        else:
            mzord = 2
        save_name += mod+'_'
        modelo = modelos[mod]
        mod_color = modelo[3]
        factor_model_hc3n = modelo[4][0]
        factor_model_dust = modelo[4][1]
        factor_model_ff   = modelo[4][2]
        if len(modelo)<=5:
            LTE = True
        else:
            LTE = modelo[6]
        mdust, m_molec, m_molec345, rout_model_hc3n_pc = model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE, source_rad)
        obs_df, rpc_profile, td_profile, nh2_profile, nh2_profile_corr, MH2_profile, Mgas_fromnH2_Msun_profile_corr, x_profile, logNHC3N_profile, logNHC3N_profile_corr, sigma, luminosity, logNH2_profile, logNH2_profile_corr, qprof = read_model_input(modelo, my_model_path, results_path, Rcrit)
        rhc3n_ind = [i for i in range(len(x_profile)) if x_profile[i] < 1e-11]
        if len(rhc3n_ind)==0:
            rhc3n = 1.5
        else:
            rhc3n = rpc_profile[rhc3n_ind[0]]
        rpc_profile.append(1.5)
        for l,line in enumerate(mykeys_flux):
            if l == 0:
                if 'model' in mod:
                    modstr = mod.split('l')[0]+'l'+' '+mod.split('l')[1]
                else:
                    modstr = mod
                ytextpos = ytextpos -0.052  
        for l,line in enumerate(mykeys_flux):
            if line not in ['plot_T', 'plot_col', 'plot_dens', 'plot_x', 'ratio_v6_v6v7']: 
                ftotal = 0
                ftotal345 = 0
                for i, row in m_molec.iterrows():
                    if row[0]<1.5: 
                        ftotal += row[line+'_beam_orig']*np.pi*(m_molec.loc[i+1, 0]**2-m_molec.loc[i, 0]**2)/pc2_beam_orig
                for i, row in m_molec345.iterrows():
                    if row[0]<1.5:
                        ftotal345 += row[line+'_beam_345']*np.pi*(m_molec345.loc[i+1, 0]**2-m_molec345.loc[i, 0]**2)/pc2_beam_345
                flux_dict[line+'_beam_orig']['flux'].append(ftotal)
                flux_dict[line+'_beam_orig']['rad'].append(rhc3n)
                flux_dict[line+'_beam_345']['flux'].append(ftotal345)
                flux_dict[line+'_beam_345']['rad'].append(rhc3n)
            elif line in ['ratio_v6_v6v7']:
                mol_ratio = m_molec['v6=1_24_-1_23_1_SM_beam_orig']/m_molec['v6=v7=1_26_2_25_-2_SM_beam_orig']
                axes[l].plot(m_molec[0], mol_ratio, color=mod_color, linestyle= '-', zorder=mzord)
    for l,line in enumerate(mykeys_flux):
        if line not in ['plot_T', 'plot_x', 'plot_col', 'ratio_v6_v6v7']: 
            axes[l].plot(flux_dict[line+'_beam_orig']['rad'], flux_dict[line+'_beam_orig']['flux'], color='k', linestyle= '--', zorder=mzord)
            axes[l].plot(flux_dict[line+'_beam_345']['rad'], flux_dict[line+'_beam_345']['flux'], color='k', linestyle= '-', zorder=mzord)
    colorbar = False      
    if colorbar:
        import matplotlib.cm as cm
        ax = axes[0]
        pos = ax.get_position()
        # Horizontal colorbar
        a = np.array([[0,1]]) 
        sm = plt.cm.ScalarMappable(cmap=cm.gist_rainbow, norm=plt.Normalize(vmin=0, vmax=len(modelos)))
        sm._A = []
        axis = 'x'
        orien = 'horizontal'
        sep=-0.043
        width=0.01
        labelpad = -45
        ticksize=8
        framewidth=1
        tickwidth=1
        label = r'r(HC$_3$N)'
        tick_font = 14
        label_font = 14
        #sep = 0.05
        cb_axl = [pos.x0 + 0.02, pos.y0 + pos.height + sep,  np.round(pos.width,2)-2*0.02 , width] 
        cb_ax = fig.add_axes(cb_axl)
        cbar = fig.colorbar(sm, orientation=orien, cax=cb_ax)
        cbar.outline.set_linewidth(framewidth)
        cbar.ax.minorticks_off()
        cbar.set_label(label, labelpad=labelpad, fontsize=label_font)
        cbar.ax.tick_params(axis=axis, direction='in')
        cbar.ax.tick_params(labelsize=tick_font)
        cbar.ax.tick_params(length=ticksize, width=tickwidth)
        cbticks = [0, 5, 10, 15, 20, 26]
        cbar.set_ticks(cbticks)
        cblabels = [] # Rmax    
        for c,tick in enumerate(cbticks):
            tt = tick
            cblabels.append(f'{rpc_profile[len(rpc_profile)-1-tt]:1.2f}')
        cbar.set_ticklabels(cblabels)
    if len(modelos) == 1:
        for m, mod in enumerate(modelos):
            save_name = modelo = modelos[mod][0]
            fig.savefig(figmod_path+'NGC253_'+save_name+'_flux_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    else:
        fig.savefig(figrt_path+'NGC253_'+save_name+'_flux_SM.pdf', bbox_inches='tight', transparent=True, dpi=400)
    plt.close()
    
def model_convolver(fort_paths, source_rad, model_name, dust_model, my_model_path):
    # Convolve models with beams and sae fort output
    convolve_models = fort_paths+'convolve_new_py'
    convolve_dustmodels = fort_paths+'dustconvolve_new_py'
    if not os.path.exists(my_model_path+model_name+'.30'):
        # Convolving HC3N models
        with open(convolve_models+'.f', 'r') as file:
                convmodel = file.readlines()
        convmodel[50] = convmodel[50].replace('s_rad', f'{source_rad:1.1f}')
        convmodel[54] = convmodel[54].replace('model_name', model_name)
        file.close()
        with open(convolve_models+'n.f', 'w') as file:
            file.writelines(convmodel)
        os.chdir(fort_paths)
        subprocess.call('ifort convolve_new_pyn.f', # Generates fort.30 and fort.305
                                stdout=subprocess.DEVNULL,
                                shell='True')
        subprocess.call('./a.out',
                                stdout=subprocess.DEVNULL,
                                shell='True')
        copyfile('fort.30', my_model_path+model_name+'.30')
        copyfile('fort.305', my_model_path+model_name+'.305')
    
    # Convolving dust models (ALways convolving because rad model can change but is not reflected in dust model name)
    with open(convolve_dustmodels+'.f', 'r') as file:
            convdmodel = file.readlines()
    convdmodel[44] = convdmodel[44].replace('s_rad', f'{source_rad:1.1f}')
    convdmodel[49] = convdmodel[49].replace('dust_model', dust_model)
    file.close()
    with open(convolve_dustmodels+'n.f', 'w') as file:
            file.writelines(convdmodel)
    os.chdir(fort_paths)
    subprocess.call('ifort dustconvolve_new_pyn.f', # Generates fort.31
                            stdout=subprocess.DEVNULL,
                            shell='True')
    subprocess.call('./a.out',
                            stdout=subprocess.DEVNULL,
                            shell='True')
    copyfile('fort.31', my_model_path+dust_model+'.31')

def model_reader(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE=True, read_only=True, source_rad=0):
    """
        Reads the model profiles
        read_only = False requires fortran compilation to convolve model profiles with beam
        read_only = True avoids fortran compilation, but requires the .31 compiled files to exist
    """
    
    model_name = modelo[0]
    dust_model = modelo[1]
    if source_rad == 0:
        source_rad = modelo[2]
    if read_only == False:
        model_convolver(fort_paths, source_rad, model_name, dust_model, my_model_path)
    # Reading dust models
    modelo_dust = my_model_path+dust_model+'.inp'
    print(dust_model)
    print('\t'+modelo[0])
    with open(modelo_dust, 'r') as file:
        modelo_dustlines = file.readlines()
        file.close()
    mdust = pd.read_csv(my_model_path+dust_model+'.31', header=None, delim_whitespace=True) # mdust[3] es free-free
    
    for i, row in mdust.iterrows():
        for column in mdust.columns.values.tolist():
            if mdust[column].dtypes == 'object':
                    if '-' in row[column] and 'E' not in row[column]:
                        mdust.loc[i, column] = row[column].replace('-', 'E-')
    mdust = mdust.apply(pd.to_numeric)

    mdust['F235GHz_mjy_beam345'] = mdust[1]+mdust[3]*factor_model_ff
    mdust['F235GHz_mjy_beam'] = mdust[5]+mdust[6]*factor_model_ff
    mdust['F345GHz_mjy_beam'] = mdust[2]+mdust[4]*factor_model_ff
    
    # Reading models
    modelom = my_model_path+model_name+'.inp'
    with open(modelom, 'r') as file:
        modelom = file.readlines()
        file.close()
    rout_model_hc3n_pc = np.float(list(filter(None, modelom[11].split(' ')))[1])*(1*u.cm).to(u.pc).value
    m_molec = pd.read_csv(my_model_path+model_name+'.30', header=None, delim_whitespace=True) # mdust[3] es free-free
    m_molec345 = pd.read_csv(my_model_path+model_name+'.305', header=None, delim_whitespace=True) # mdust[3] es free-free
    
    for i, row in m_molec.iterrows():
        for column in m_molec.columns.values.tolist():
            if m_molec[column].dtypes == 'object':
                    if '-' in row[column] and 'E' not in row[column]:
                        m_molec.loc[i, column] = row[column].replace('-', 'E-')
    m_molec = m_molec.apply(pd.to_numeric)
    
    for i, row in m_molec345.iterrows():
        for column in m_molec345.columns.values.tolist():
            if m_molec345[column].dtypes == 'object':
                    if '-' in row[column] and 'E' not in row[column]:
                        m_molec345.loc[i, column] = row[column].replace('-', 'E-')
    m_molec345 = m_molec345.apply(pd.to_numeric)
    
    # continuum from models
    m_molec['F235GHz_mjy_beam'] = m_molec[12]*1000*factor_model_hc3n
    m_molec345['F235GHz_mjy_beam345'] = m_molec345[12]*1000*factor_model_hc3n
    
    
    for l,line in enumerate(line_column):
        if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x']:
            # line from models
            if LTE:
                m_molec[line+'_beam_orig'] = m_molec[line_column[line][3]]*1000*factor_model_hc3n
                m_molec345[line+'_beam_345'] = m_molec345[line_column[line][3]]*1000*factor_model_hc3n
            else:
                if line == 'v=0_26_25_SM':
                    mind = 2
                elif line == 'v7=1_24_1_23_-1_SM':
                    mind = 3
                elif line == 'v7=1_26_1_25_-1_SM':
                    mind = 4
                elif line == 'v6=1_24_-1_23_1_SM':
                    mind = 5
                elif line == 'v6=1_26_-1_25_1_SM':
                    mind = 6
                else:
                    mind = 0
                if mind != 0:
                    m_molec[line+'_beam_orig'] = m_molec[mind]*1000*factor_model_hc3n
                    m_molec345[line+'_beam_345'] = m_molec345[mind]*1000*factor_model_hc3n
                else:
                    m_molec[line+'_beam_orig'] = np.nan
                    m_molec345[line+'_beam_345'] = np.nan
    
    return mdust, m_molec, m_molec345, rout_model_hc3n_pc

def model_reader_noconv(modelo, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE=True, read_only=False, source_rad=0):
    """
        Model profiles without convolving by the beam
    """
    D_Mpc = 3.5
    parsec = (1*u.pc).to(u.cm).value
    cubo219_path = '/Users/frico/Documents/data/NGC253_HR/Continuums/MAD_CUB_219GHz_spw25_continuum.I.image.pbcor.fits'
    cubo219 = utiles.Cube_Handler('219', cubo219_path)
    cubo_219_pixlen_pc = u_conversion.lin_size(D_Mpc, cubo219.pylen*3600).to(u.pc).value
    arcsec2_beam=np.pi*cubo219.bmaj*3600*cubo219.bmin*3600/4.0/np.log(2.0)
    sr_beam = (arcsec2_beam*u.arcsec**2).to(u.sr).value
    pc2_beam = arcsec2_beam*(D_Mpc*1e6*np.pi/(180.0*3600.0))**2
    
    cubo345_path = '/Users/frico/Documents/data/NGC253_HR/Continuums/MAD_CUB_350GHz_continuum_mfs.I.manual.image.pbcor.fits'
    cubo345 = utiles.Cube_Handler('345', cubo345_path)
    cubo_345_pixlen_pc = u_conversion.lin_size(D_Mpc, cubo345.pylen*3600).to(u.pc).value
    arcsec2_beam345 =np.pi*cubo345.bmaj*3600*cubo345.bmin*3600/4.0/np.log(2.0)
    sr_beam345 = (arcsec2_beam345*u.arcsec**2).to(u.sr).value
    pc2_beam345 = arcsec2_beam345*(D_Mpc*1e6*np.pi/(180.0*3600.0))**2
    
    flux = 2.2364E+12
    f_pix = flux * ((cubo_219_pixlen_pc/(D_Mpc*1e6))**2)
    f_beam = flux * ((cubo_219_pixlen_pc/(D_Mpc*1e6))**2)/(cubo_219_pixlen_pc**2)*pc2_beam
    
    model_name = modelo[0]
    dust_model = modelo[1]
    if source_rad == 0:
        source_rad = modelo[2]
    
    # Reading dust models
    modelo_dust = my_model_path+dust_model+'.inp'
    print(dust_model)
    print('\t'+modelo[0])
    with open(modelo_dust, 'r') as file:
        modelo_dustlines = file.readlines()
        file.close()
    
    mdust = pd.read_csv(my_model_path+dust_model+'_.par', header=None, delim_whitespace=True)
    
    
    ff_mdust = pd.read_csv(my_model_path+dust_model+'_.ffpar', header=None, delim_whitespace=True)
    
    
    ff_mdust['ff_F235GHz_Jy_sr']=ff_mdust[9]/(2.0*np.pi)*(D_Mpc*1e6)**2/(ff_mdust[0]*source_rad*parsec)*(parsec**2) #Jy/sr
    mdust['F235GHz_Jy_sr']=mdust[9]/(2.0*np.pi)*(D_Mpc*1e6)**2/(mdust[0]*source_rad*parsec)*(parsec**2) #Jy/sr
    ff_mdust['ff_F345GHz_Jy_sr']=ff_mdust[9]/(2.0*np.pi)*(D_Mpc*1e6)**2/(ff_mdust[0]*source_rad*parsec)*(parsec**2) #Jy/sr
    mdust['F345GHz_Jy_sr']=mdust[7]/(2.0*np.pi)*(D_Mpc*1e6)**2/(mdust[0]*source_rad*parsec)*(parsec**2) #Jy/sr
    
    mdust['F235GHz_mjy_beam'] = 1000*sr_beam*(mdust['F235GHz_Jy_sr']+ff_mdust['ff_F235GHz_Jy_sr']*factor_model_ff)
    mdust['F235GHz_mjy_beam345'] = 1000*sr_beam345*(mdust['F235GHz_Jy_sr']+ff_mdust['ff_F235GHz_Jy_sr']*factor_model_ff)
    mdust['F345GHz_mjy_beam'] = 1000*sr_beam345*(mdust['F345GHz_Jy_sr']+ff_mdust['ff_F345GHz_Jy_sr']*factor_model_ff)

    # Reading models
    modelom = my_model_path+model_name+'.inp'
    with open(modelom, 'r') as file:
        modelom = file.readlines()
        file.close()
    rout_model_hc3n_pc = np.float(list(filter(None, modelom[11].split(' ')))[1])*(1*u.cm).to(u.pc).value
    m_molec = pd.read_csv(my_model_path+model_name+'_1rec.par', header=None, delim_whitespace=True) 
    m_molec345 = pd.read_csv(my_model_path+model_name+'_1rec.par', header=None, delim_whitespace=True) 

    for c,column in enumerate(m_molec.columns.values.tolist()):
        if c != 0:
            m_molec[column] = sr_beam*m_molec[column]
        
    for c,column in enumerate(m_molec345.columns.values.tolist()):
        if c !=0:
            m_molec345[column] = sr_beam345*m_molec345[column]
        
    m_cont = pd.read_csv(my_model_path+model_name+'_1rec.parcont', header=None, delim_whitespace=True) 

    
    # continuum from models
    m_molec['F235GHz_mjy_beam'] = sr_beam*m_cont[5]*1000*factor_model_hc3n
    m_molec345['F235GHz_mjy_beam345'] = sr_beam345*m_cont[5]*1000*factor_model_hc3n
    
    
    source_rad = rout_model_hc3n_pc # change to scale to other sizes???
    ff_mdust[0] = ff_mdust[0]*source_rad
    mdust[0] = mdust[0]*source_rad
    
    m_molec[0] = m_molec[0]*source_rad
    print('+++++++++', source_rad)
    print(m_molec[0])
    print('+++++++++')
    m_molec345[0] = m_molec345[0]*source_rad
    for l,line in enumerate(line_column):
        if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x']:
            # line from models
            if LTE:
                m_molec[line+'_beam_orig'] = m_molec[line_column[line][3]]*1000*factor_model_hc3n
                m_molec345[line+'_beam_345'] = m_molec345[line_column[line][3]]*1000*factor_model_hc3n
            else:
                if line == 'v=0_26_25_SM':
                    mind = 2
                elif line == 'v7=1_24_1_23_-1_SM':
                    mind = 3
                elif line == 'v7=1_26_1_25_-1_SM':
                    mind = 4
                elif line == 'v6=1_24_-1_23_1_SM':
                    mind = 5
                elif line == 'v6=1_26_-1_25_1_SM':
                    mind = 6
                else:
                    mind = 0
                if mind != 0:
                    m_molec[line+'_beam_orig'] = m_molec[mind]*1000*factor_model_hc3n
                    m_molec345[line+'_beam_345'] = m_molec345[mind]*1000*factor_model_hc3n
                else:
                    m_molec[line+'_beam_orig'] = np.nan
                    m_molec345[line+'_beam_345'] = np.nan
    
    return mdust, m_molec, m_molec345, rout_model_hc3n_pc

def model_reader_comp(modelo, fort_paths, factor_model_hc3n, factor_model_ff, factor_model_dust, line_column, my_model_path, LTE=True, read_only=False):
    model_name = modelo[0]
    dust_model = modelo[1]
    source_rad = modelo[2]
    if read_only == False:
        model_convolver(fort_paths, source_rad, model_name, dust_model, my_model_path)
    # Reading dust models
    print(dust_model)
    print('\t'+modelo[0])
    mdust = pd.read_csv(my_model_path+dust_model+'.31', header=None, delim_whitespace=True) # mdust[3] es free-free
    mdust['F235GHz_mjy_beam345'] = mdust[1]+mdust[3]*factor_model_ff
    mdust['F235GHz_mjy_beam'] = mdust[5]+mdust[6]*factor_model_ff
    mdust['F345GHz_mjy_beam'] = mdust[2]+mdust[4]*factor_model_ff
    # Reading models
    modelom = my_model_path+model_name+'.inp'
    with open(modelom, 'r') as file:
        modelom = file.readlines()
        file.close()
    rout_model_hc3n_pc = np.float(list(filter(None, modelom[11].split(' ')))[1])*(1*u.cm).to(u.pc).value
    m_molec = pd.read_csv(my_model_path+model_name+'.30', header=None, delim_whitespace=True) # mdust[3] es free-free
    m_molec345 = pd.read_csv(my_model_path+model_name+'.305', header=None, delim_whitespace=True) # mdust[3] es free-free
    if LTE:
        tau_molec = pd.DataFrame()
    else: # Models only return line taus when not thermalised (vel recubrimiento >0)
        tau_molec = pd.read_csv(my_model_path+model_name+'_1rec.tau', header=None, delim_whitespace=True)
        tau_molec[0] = tau_molec[0]*rout_model_hc3n_pc
    # Updating continuum from models and manual factor
    m_molec['F235GHz_mjy_beam'] = m_molec[12]*1000*factor_model_hc3n
    m_molec345['F235GHz_mjy_beam345'] = m_molec345[12]*1000*factor_model_hc3n
    # Continuum tau from dust models
    tau_dust = pd.read_csv(my_model_path+dust_model+'_.taupar', header=None, delim_whitespace=True)
    tau_dust[0] = tau_dust[0]*rout_model_hc3n_pc
    tau_dust['tau_345'] = tau_dust[7]*1.0
    tau_dust['tau_235'] = tau_dust[9]*1.0
    for l,line in enumerate(line_column):
        if line not in ['plot_conts', 'plot_T', 'plot_col', 'plot_dens', 'plot_x']:
            # line from models
            m_molec[line+'_beam_orig'] = m_molec[line_column[line][3]]*1000*factor_model_hc3n
            m_molec345[line+'_beam_345'] = m_molec345[line_column[line][3]]*1000*factor_model_hc3n
            if not LTE:
                tau_molec[line+'_tau'] = tau_molec[line_column[line][3]]
    return mdust, m_molec, m_molec345, rout_model_hc3n_pc, tau_molec, tau_dust
