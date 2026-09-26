#==============================================================================
#                    
#                   Conversion btw units
#
#==============================================================================

from math import pi
import astropy.units as u
import astropy.constants.si as _si
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from astrothesispy.utiles import utiles

# Cosmology
cosmo = FlatLambdaCDM(H0=69.6, Om0=0.286)

# Luminostiy to Flux
def lum_to_flux(luminosity, z_source, z=True):
    """
    From Luminosity in Lsun to flux in W/m^2
    F = L/(4pid^2)
    if z is false, z_source is dist in Mpc
    """
    if z==True:
        dist_lum = cosmo.luminosity_distance(z_source)
        dist_lum = dist_lum.to(u.m) # dist lum in meters
    else:
        dist_lum = u.Quantity(z_source, u.Mpc)
        dist_lum = dist_lum.to(u.m)
    lum = u.Quantity(luminosity, u.L_sun)
    lum = lsun_to_watt(lum)
    flux = lum/(4.0*pi*dist_lum**2)
    return flux # Flux in W/m^2

def flux_to_lum(flux, z_source, z=True):
    """
    From flux in W/m^2 to Luminosity in Lsun
    F = L/(4pid^2)
    """
    if z==True:
        dist_lum = cosmo.luminosity_distance(z_source)
    else: 
        dist_lum = u.Quantity(z_source, u.Mpc)
    dist_lum = dist_lum.to(u.m) # dist lum in meters
    flux = u.Quantity(flux, u.W / u.m**2)
    lum = (4.0*pi*dist_lum**2) * flux
    lum = lum.to(u.L_sun)
    return lum # Luminosity in Lsun

def fluxwcm2_to_lum(flux, dist_lum):
    """
    From flux in W/cm^2 to Luminosity in Lsun
    F = L/(4pid^2)
    """
    flux = u.Quantity(flux, u.W / u.cm**2)
    flux = flux.to(u.W/u.m**2)
    dist_lum = dist_lum.to(u.m) # dist lum in meters
    lum = (4.0*pi*dist_lum**2) * flux
    lum = lum.to(u.L_sun)
    return lum # Luminosity in Lsun

# Stefan-Boltzmann Law
def stef_boltz(r, T):
    """
    Stefan-Boltzmann law
    r must have astropy units
    T in Kelvins
    returns luminostiy in W
    """
    if r.unit != u.m:
        r_m = r.to(u.m)
    else:
        r_m = r
    s_si = _si.sigma_sb
    T = u.Quantity(T, u.K)
    #r = u.Quantity(r, u.m)
    L = 4*pi*(r_m**2)*s_si*T**4
    return L

# Stefan-Boltzmann Law Error
def stef_boltz_error(r, r_err, T, T_err):
    """
    Stefan-Boltzmann law
    r must have astropy units
    T in Kelvins
    returns luminostiy in W
    """
    if r.unit != u.m:
        r_m = r.to(u.m)
    else:
        r_m = r
    if r_err.unit != u.m:
        r_m_err = r_err.to(u.m)
    else:
        r_m_err = r_err
    s_si = _si.sigma_sb
    T = u.Quantity(T, u.K)
    T_err = u.Quantity(T_err, u.K)
    #r = u.Quantity(r, u.m).value
    L_err = np.sqrt(((4*4*pi*(r_m**2)*s_si*(T**3)*T_err)**2)+((4*pi*2*(r_m)*s_si*(T**4)*r_m_err)**2))
    return L_err

# Planck Function
def planck(freq, T):
    """ 
    Planck spectral radiance B_nu for frequency in Hz and temperature in K.

    Returns W / (m2 Hz sr); inputs can be scalars, arrays or quantities.
    """
    h_si = _si.h
    c_si = _si.c
    k_si = _si.k_B
    T = u.Quantity(T, u.K)
    freq = u.Quantity(freq, u.Hz)
    exponent = (h_si*freq/(k_si*T)).to_value(u.dimensionless_unscaled)
    return (2 * h_si * freq**3 / c_si**2 / np.expm1(exponent) / u.sr).to(
        u.W / (u.m**2 * u.Hz * u.sr)
    )
    
def Energy_cmtoK(energy_cm):
    """
    energy in cm-1 to K
    """
    e_cm = u.Quantity(energy_cm, u.cm**-1)
    faccmtk=1.4388325*u.cm*u.K
    e_k = e_cm*faccmtk
    return e_k.value

def planckwave(wav, T):
    """B_lambda in W / (m2 m sr); bare wavelengths are meters, T is K."""
    wav = u.Quantity(wav, u.m)
    T = u.Quantity(T, u.K)
    exponent = (_si.h * _si.c / (wav * _si.k_B * T)).to_value(u.one)
    return (2 * _si.h * _si.c**2 / wav**5 / np.expm1(exponent) / u.sr).to(
        u.W / (u.m**2 * u.m * u.sr))


def lin_size(D, r, D_err = 0):
    """
    Linear size from angular size
    D is distance in Mpc
    r is radius in arcsec
    returns the linear size in meters
    """
    D = u.Quantity(D, u.Mpc)
    d = D.to(u.m)
    L = r*d/206265.0
    if D_err == 0:
        return L
    else:
        D_err = u.Quantity(D_err, u.Mpc)
        d_err = D_err.to(u.m)
        L_err = np.sqrt(((0*d.value/206265.0)**2)+((d_err.value*r/206265.0)**2))
        L_err = (d_err.value*r/206265.0)
        return L, L_err
    
# Linear Size
def lin_size_pd(D, r, D_err = 0):
    """
    Linear size from angular size
    D is distance in Mpc
    r is radius in arcsec
    returns the linear size in meters
    """
    D = u.Quantity(D, u.Mpc)
    d = D.to(u.m).value
    L = r*d/206265.0
    if D_err == 0:
        return L
    else:
        D_err = u.Quantity(D_err, u.Mpc)
        d_err = D_err.to(u.m)
        L_err = (d_err.value*r/206265.0)
        return L, L_err
    
# Angular Size
def ang_size(D, L):
    """
    Angular size from linear size
    D is distance in Mpc
    L  linear size in meters
    returns the angular size in arcsec
    """
    D = u.Quantity(D, u.Mpc)
    L = u.Quantity(L, u.m)
    d = D.to(u.m)
    r = (u.arcsec)*L*206265.0/d
    return r

# Flux conversions
def wm2_to_jykms(flux, freq_wave_obs, wavelength=False):
    """
    From W/m^2 to Jy km/s
    Intensity (W/m^2)
    freq_wave_obs (GHz or mm)
    c_si (km/s)
    wavelength True --> fobs is lambda_obs
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    
    flux = u.Quantity(flux, u.W/ u.m**2)
    
    spectral_value = u.Quantity(freq_wave_obs, u.mm if wavelength else u.GHz)
    fobs = spectral_value.to(u.Hz, equivalencies=u.spectral())

    dflux = flux * c_si / fobs #  W * km / s * Hz * m^2
    dflux_kms = dflux.to(u.Jy * u.km / u.s) # Jy * km / s
    return dflux_kms # Flux in Jy km/s
    
# Flux conversions
def wm2_to_jy(flux, fwhm, wavelength=False):
    """Mean flux density in Jy from W/m2 and equivalent rectangular bandwidth.
    The legacy fwhm argument is bandwidth in GHz, not Gaussian FWHM.
    wavelength=True interprets it as c/lambda (lambda in mm), not delta-lambda.
    """
    
    flux = u.Quantity(flux, u.W/ u.m**2)
    
    spectral_value = u.Quantity(fwhm, u.mm if wavelength else u.GHz)
    fobs = spectral_value.to(u.Hz, equivalencies=u.spectral())

    dflux = flux / fobs #  W / Hz * m^2
    dflux_k = dflux.to(u.Jy) # Jy
    return dflux_k # Flux in Jy
    
def jy_to_wm2(flux, fwhm, wavelength=False):
    """Integrated flux in W/m2 from Jy and equivalent rectangular bandwidth.
    The legacy fwhm argument is bandwidth in GHz, not Gaussian FWHM.
    wavelength=True interprets it as c/lambda (lambda in mm), not delta-lambda.
    """
    
    flux = u.Quantity(flux, u.Jy)
    
    spectral_value = u.Quantity(fwhm, u.mm if wavelength else u.GHz)
    fobs = spectral_value.to(u.Hz, equivalencies=u.spectral())

    dflux = flux * fobs #  W / Hz * m^2
    dflux_k = dflux.to(u.W/ u.m**2) # Jy
    return dflux_k # Flux in Jy

def jykms_to_wm2(flux, freq_wave_obs, wavelength=False):
    """
    From Jy km/s to W/m^2
    Intensity (W/m^2)
    freq_wave_obs (GHz or mm)
    c_si (km/s)
    wavelength True --> fobs is lambda_obs
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    
    flux = u.Quantity(flux, u.Jy * u.km / u.s)
    
    spectral_value = u.Quantity(freq_wave_obs, u.mm if wavelength else u.GHz)
    fobs = spectral_value.to(u.Hz, equivalencies=u.spectral())

    dflux = flux * fobs / c_si #  Jy * Hz
    dflux_w = dflux.to(u.W / u.m**2) # W/m^2
    return dflux_w # Flux in Jy km/s

def jykms_to_wm2_w(flux, wave_obs):
    """
    From Jy km/s to W/m^2
    Intensity (W/m^2)
    freq_wave_obs (GHz or mm)
    c_si (km/s)
    wavelength True --> fobs is lambda_obs
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    
    flux = u.Quantity(flux, u.Jy * u.km / u.s)
    
    wave_obs = u.Quantity(wave_obs, u.um)
    fobs = wave_obs.to(u.Hz, equivalencies=u.spectral())
    
    dflux = flux * fobs / c_si #  Jy * Hz
    dflux_w = dflux.to(u.W / u.m**2) # W/m^2
    return dflux_w # Flux in Jy km/s

def wm_to_ergscm(flux):
    """
    From W/m^2 to erg/s/cm^2
    """
    flux = u.Quantity(flux, u.W/u.m**2)
    dflux = flux.to(u.erg*u.s**(-1)*u.cm**(-2))
    return dflux # Flux in erg/s/cm^2

def wcm2_to_ergscm2(flux):
    """
    From W/cm^2 to erg/s/cm^2
    """
    flux = u.Quantity(flux, u.W/u.cm**2)
    dflux = flux.to(u.erg*u.s**(-1)*u.cm**(-2))
    return dflux # Flux in erg/s/cm^2

def ergscm_to_wm(flux):
    """
    From erg/s/cm^2 to W/m^2
    """
    flux = u.Quantity(flux, u.erg*u.s**(-1)*u.cm**(-2))
    dflux = flux.to(u.W/u.m**2)
    return dflux # Flux in W/m^2

def ergscmHz_to_jy(flux):
    """
    From erg/s/cm^2 to jy
    """
    flux = u.Quantity(flux, u.erg*u.s**(-1)*u.cm**(-2)*u.Hz**(-1))
    dflux = flux.to(u.Jy)
    return dflux # Flux in Jy


def ergscmum_to_mjy(flux, lambda_microns):
    """
    From erg/s/cm^2/um to mJy
    """
    #flux = u.Quantity(flux, u.erg*u.s**(-1)*u.cm**(-2)*(u.um))
    #lambda_microns = u.Quantity(lambda_microns*(u.um))
    # 1Jy = 1E-23 ergs 
    # (1 * u.mJy).to(u.erg/u.s/u.cm**2/u.Hz) = 1E-26
    dflux = flux*(lambda_microns**2)*1e26/(_si.c.to(u.um/u.s).value)
    return dflux # Flux in mJy

def jy_to_ergscm(flux):
    """
    From erg/s/cm^2 to jy
    """
    flux = u.Quantity(flux, u.Jy)
    dflux = flux.to(u.erg*u.s**(-1)*u.cm**(-2)*u.Hz**(-1))
    return dflux # Flux in erg/s/cm^2

# Luminosities conversions
def lsun_to_watt(luminosity):
    """
    From luminostiy in Lsun to W
    """
    lum = u.Quantity(luminosity, u.L_sun)
    return lum.to(u.W)

def watt_to_lsun(luminosity):
    """
    From luminostiy in W to Lsun
    """
    lum = u.Quantity(luminosity, u.W)
    return lum.to(u.L_sun)

def lsun_to_ergs(luminosity):
    """
    From luminostiy in Lsun to erg/s
    """
    lum = u.Quantity(luminosity, u.L_sun)
    return lum.to(u.erg / u.s)

def ergs_to_lsun(luminosity):
    """
    From luminostiy in erg/s to Lsun
    """
    lum = u.Quantity(luminosity, u.erg / u.s)
    return lum.to(u.L_sun)

def integrated_flux_density_size(S, source_size, beam_size):
    """
     Integrated flux density for a resolved source
     https://iopscience.iop.org/article/10.1086/300337/pdf
    """
    if source_size<beam_size:
        # Unresolved
        I = S*(source_size/beam_size)**(1/2)
    else:
        # Resolved
        I = S*(source_size/beam_size)
    return I


def Flux_ring_average(S, beam_size, r_ring, width_ring):
    """
     Flux averaged over ring area (from Jy/beam to Jy)
     beam_size and r_ring and width_ring must be in same units
    """
    w_ring = width_ring/2
    I = (S/beam_size)*np.pi*((r_ring+w_ring)**2-(r_ring-w_ring)**2)
    return I



def integrated_flux_density_fwhm(S, FWHM_min, bmin):
    """
     Integrated flux density for a resolved source
     https://iopscience.iop.org/article/10.1086/300337/pdf
    """
    if FWHM_min<bmin:
        # Unresolved
        I = S*(FWHM_min/bmin)**(1/2)
    else:
        # Resolved
        I = S*(FWHM_min/bmin)
    return I


# Flux to luminosity
def wm2_to_lsun(flux, dist_lum):
    """
    From luminostiy in wm2 to Lsun
    distance luminosity requiered in Mpc
    """
    d_lum = u.Quantity(dist_lum, u.Mpc)
    f = u.Quantity(flux, u.W / u.m**2)
    lum = 4.*np.pi*(d_lum.to(u.m)**2.)*f
    return lum.to(u.L_sun)

# Jy/beam to Brightness Temperature
def Jybeam_to_T(Jybeam, freq, Bmin, Bmaj):
    # https://science.nrao.edu/facilities/vla/proposing/TBconv
    """
    Flux density in Jy/Beam yo Brightness temperature in Kelvin
    freq in GHz
    Bmin and Bmaj in arcseconds
    """
    T = 1.222E3 * (Jybeam * 1000.) / ((freq**2.) * Bmin *Bmaj)
    return T

def Jybeam_to_Jy_madcuba(Jybeam, npix, bmin, bmaj, xpixscale, ypixscale):
    beamsize = utiles.beam_size(bmin, bmaj)
    pxinbeam = beamsize/(xpixscale*ypixscale)
    if npix/pxinbeam > 1:
        extradiv = npix+npix/pxinbeam
    else:
        extradiv = 1
    Total_flux_Jy = (np.nansum(Jybeam*Jybeam/(1+pxinbeam/npix)) + np.nansum(Jybeam*Jybeam/(pxinbeam/npix)))/(2*extradiv)
    return Total_flux_Jy
    
def Jybeam_to_Jy(Jybeam, bmin, bmaj, xpixscale, ypixscale):
    """
    This gives lower values than madcuba T_T
    Flux density to flux
    Integrated area
    https://www.eaobservatory.org/jcmt/faq/how-can-i-convert-from-mjybeam-to-mjy/
    Total Flux = flux summed over a number of pixels/(number of pixels in a beam)
    [mJy*pixels/beam] / [pixels/beam] = [mJy].
    Beam Area =  2 × π × σ² [arcsec]
    number of pixels in a beam = Beam Area [arcsec] / (pixel length)²
    """
    beams_per_pixel = abs(xpixscale*ypixscale) / (bmin * bmaj * np.pi/(4*np.log(2))) * u.beam
    q = ((Jybeam*u.Jy/u.beam) * beams_per_pixel).to(u.Jy)
    Total_flux_Jy = np.nansum(q).value
    return Total_flux_Jy
    
def Jybeam_to_T_fromGaussian(Jybeam, freq_GHz, x_FWHM, y_FWHM, Jybeam_err=0, x_FWHM_err=0, y_FWHM_err=0):
    """Rayleigh-Jeans brightness temperature and independent 1-sigma error.

    Inputs: Jy/beam, GHz, arcsec (or compatible quantities). Frequency is exact;
    flux and the two Gaussian FWHM uncertainties are assumed uncorrelated.
    Both outputs are quantities in K; arrays broadcast normally.
    """
    flux = u.Quantity(Jybeam, u.Jy)
    flux_err = u.Quantity(Jybeam_err, u.Jy)
    x = u.Quantity(x_FWHM, u.arcsec)
    y = u.Quantity(y_FWHM, u.arcsec)
    dx = u.Quantity(x_FWHM_err, u.arcsec)
    dy = u.Quantity(y_FWHM_err, u.arcsec)
    freq = u.Quantity(freq_GHz, u.GHz)
    if np.any(x <= 0*u.arcsec) or np.any(y <= 0*u.arcsec) or np.any(freq <= 0*u.GHz):
        raise ValueError("Frequency and Gaussian widths must be positive")
    if np.any(flux_err < 0*u.Jy) or np.any(dx < 0*u.arcsec) or np.any(dy < 0*u.arcsec):
        raise ValueError("Uncertainties must be nonnegative")
    omega = np.pi*x*y/(4*np.log(2))
    intensity = flux/omega
    # Derivative form remains finite when flux or a supplied uncertainty is zero.
    intensity_err = np.sqrt((flux_err/omega)**2 +
                            (intensity*dx/x)**2 + (intensity*dy/y)**2)
    equivalency = u.brightness_temperature(freq)
    return intensity.to(u.K, equivalencies=equivalency), intensity_err.to(u.K, equivalencies=equivalency)


def Jybeam_to_T_fromGaussian_v2(Jybeam, freq_GHz, x_FWHM, y_FWHM, Jybeam_err=0, x_FWHM_err=0, y_FWHM_err=0):
    """Compatibility alias for Jybeam_to_T_fromGaussian, including arrays."""
    return Jybeam_to_T_fromGaussian(Jybeam, freq_GHz, x_FWHM, y_FWHM,
                                   Jybeam_err, x_FWHM_err, y_FWHM_err)


def Jybeam_to_Jypx(Jybeam, xpixsize, ypixsize, bmin, bmaj):
    """
    Flux density in Jy/Beam to Jy/px
    Now area integration can be done with Aperture_photometry from photutils
    """
    beam_size = np.pi*bmin*bmaj/(4*np.log(2))
    Jypx = Jybeam*abs(xpixsize*ypixsize)/beam_size
    return Jypx

def Jybeam_to_T_fromGaussian_array(Jybeam, freq_GHz, x_FWHM, y_FWHM, Jybeam_err=0, x_FWHM_err=0, y_FWHM_err=0):
    """Compatibility alias for Jybeam_to_T_fromGaussian, including arrays."""
    return Jybeam_to_T_fromGaussian(Jybeam, freq_GHz, x_FWHM, y_FWHM,
                                   Jybeam_err, x_FWHM_err, y_FWHM_err)


def Jy_to_T(Jy, freq_GHz, Bmin_arcsec, Bmaj_arcsec):
    """
     Conversion from flux density to brightness temperature
     assuming a R-J (hnu<<kT) and a point source
    """
    T = 13.6*(300./freq_GHz)**2 * Jy /(Bmin_arcsec*Bmaj_arcsec)
    return T

def Jybeamkms_to_Tkms(Snu, freq_GHz, Bmin_arcsec, Bmaj_arcsec):
    """
     Conversion from Integrated flux density (Jy/beam km/s) 
     to brightness temperature (K km/s)
     assuming a R-J (hnu<<kT) and a point source
    """
    Tkms = 1.224E6*Snu/((freq_GHz**2.) * (Bmin_arcsec*Bmaj_arcsec))
    return Tkms

def channel_width_freq_to_kms(width_freq, restfreq):
    c_si = _si.c.to(u.km / u.s) # km / s
    #chanv = c_si*(1.-(width_freq/restfreq))
    chanv = c_si*width_freq/restfreq
    return chanv


def vel_to_freq(restfreq, vel):
    """
    velocity in km/s to frequency in Hz conversion 
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    freq = restfreq*(1.-vel/c_si.value)
    return freq

def freq_to_resfreq(freq, vel):
    """
    velocity in km/s to frequency in Hz conversion 
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    restfreq = freq/(1.-vel/c_si.value)
    return restfreq

def freq_to_vel(restfreq, freq):
    """
    velocity in km/s to frequency in Hz conversion 
    """
    c_si = _si.c.to(u.km / u.s) # km / s
    vel = c_si.value*(restfreq-freq)/restfreq
    return vel
