# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 22:51:05 2022

@author: Fer
"""

from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_plot
from astrothesispy.utiles import u_conversion

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants.si as _si

#==============================================================================
#                    
#                   Utils: Recurrent formulas and functions for ALMA stuff
#
#==============================================================================

def beam_size(bmin, bmaj):
    """
    Returns beam size
    bmin and bmaj are the FWHM of the beam
    """
    beam_size = np.pi * bmin * bmaj / (4. *np.log(2))
    return beam_size

def beam_size_err(bmin, bmaj, bmin_err, bmaj_err):
    """
    Returns beam size error
    bmin and bmaj are the FWHM of the beam
    """
    beam_size_err = np.sqrt((np.pi * bmin *bmaj_err  / (4. *np.log(2)))**2 + (np.pi * bmaj * bmin_err / (4. *np.log(2)))**2)
    return beam_size_err

def HPBW(wave, antenna_meters):
    """
    ALMA primary beam FWHM
    """
    if (wave.unit == 'm') | (wave.unit == 'um'):
        # Wave is in wavelength
        wave = wave.to(u.m)
    else:
        # Wave is frequency
        wave = (wave).to(u.m, equivalencies=u.spectral())
    hpbw_rad = 1.13*wave/antenna_meters
    hpbw_arcsec = (hpbw_rad*u.rad).to(u.arcsec)
    return hpbw_arcsec

def antenna_temperature(wave, source_size, antenna_beam, brightness):
    """
    Goldsmith & Langer 1999
    """
    k_si = _si.k_B
    Ta = ((wave**2)/(2.*k_si))*(source_size/antenna_beam)*brightness
    return Ta

def alma_resol(longest_baseline_in_wavelengths):
    """
    Alma resolution in arcsec 
    https://casaguides.nrao.edu/index.php/Image_Continuum
    """
    res = 206265.0/(longest_baseline_in_wavelengths)
    return res

def FOV_telescope(freq_GHz, antenna_size_in_m):
    FWHM_arcsec = 3600.*(180./np.pi)*(3e8/freq_GHz)/antenna_size_in_m
    return FWHM_arcsec

def alma_cell_size(longest_base_in_wavelengths):
    # From https://casaguides.nrao.edu/index.php/Image_Continuum
    longw = longest_base_in_wavelengths
    resolution_arcsec = 206265.0/longw
    cellsize_arcsec = resolution_arcsec/7
    return cellsize_arcsec

def cellsize_ALMA(obs_freq_GHz, longes_base_inkm):
    max_resol_arcsec = 3600. * (180. / np.pi) * (3e8 / obs_freq_GHz) / longes_base_inkm
    return max_resol_arcsec/5.

def cellsize(resolution):
    """
    Alma cell size in arcsec given a certain resolution in arcsec
    """
    cellsize = resolution/7.
    return cellsize

def Alma_spw_cover(bandwidth_ghz, center_freq_ghz):
    """
    Calculates the frequencies covered
    """
    return [center_freq_ghz-bandwidth_ghz/2., center_freq_ghz+bandwidth_ghz/2.]