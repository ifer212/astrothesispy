# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 22:28:22 2022

@author: Fer
"""


from astropy.wcs import WCS
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs.utils import skycoord_to_pixel, proj_plane_pixel_scales
from copy import deepcopy
#from legacycontour import _cntr as cntr
from matplotlib.patches import Ellipse

from astrothesispy.utiles import utiles
from astrothesispy.utiles import utiles_plot
from astrothesispy.utiles import utiles_alma
from astrothesispy.utiles import u_conversion

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants.si as _si

#==============================================================================
#                    
#                   Utils: Recurrent formulas and functions for cubes
#
#==============================================================================

class Cube_Handler:    
    'Open fits cubes and access its attributes'
    def __init__(self, name, cube_path, linefreq_center=False, rms=False, vel=0):
        self.path = cube_path
        self.name = name
        fitsfile  = fits.open(self.path)
        fitsdata  = fitsfile[0]
        self.header = fitsdata.header
        if self.header['NAXIS'] == 2:
            # Only 1 spectral dimension (i.e. one plane)
            self.fitsdata   = fitsdata.data
        elif self.header['NAXIS3'] == 1:
            # Also only 1 spectral dimension (i.e. one plane)
            self.fitsdata   = fitsdata.data
        else:
            self.fitsdata   = fitsdata.data[0,:,:,:]
        self.shape       = fitsdata.data.shape
        self.wcs         = WCS(self.header, naxis=2) 
        if vel == 0:
            # No reference velocity given, using cube velref
            self.velref      = self.header['VELREF'] #km/s
        else:
            self.velref      = vel
        self.restfreq    = self.header['RESTFREQ']
        # Computing beamsize as pi*bmin*bmaj/(4log2)
        self.bmaj = self.header['BMAJ']  #DEG
        self.bmin = self.header['BMIN']  #DEG
        self.bpa  = self.header['BPA']   #DEG
        self.beamsize = utiles_alma.beam_size(self.bmin, self.bmaj) #DEG**2
        self.pxlen = self.header['CDELT1'] # DEG
        self.pylen = self.header['CDELT2'] # DEG
        self.pxinbeam = np.abs(self.beamsize/(self.pxlen*self.pylen))
        if rms==False:
            self.std     = np.nanstd(sigma_clip(self.fitsdata, sigma=3, axis=None))
        elif isinstance(rms, list):
            # measuring rms in linefree channels
            self.std     = np.nanstd(self.fitsdata[rms[0]:rms[1],:,:])
        else:
            self.std     = rms
        self.max         = np.nanmax(self.fitsdata)
        self.min         = np.nanmin(self.fitsdata)
        self.cdelt3      = self.header['CDELT3']
        self.totalchans  = self.shape[1]
        # Channel width in km/s
        self.chanwidth   = np.abs(u_conversion.channel_width_freq_to_kms(self.cdelt3, self.restfreq).value)
        # Channel range  
        self.channelrange= np.arange(1, self.shape[1]+1, 1)
        # Observed freq range
        self.obsfreq     = cube_freqrange(self.header)
        # Rest freq range
        self.restfreq    = u_conversion.freq_to_resfreq(self.obsfreq, self.velref)
        # Velocities radio convention at cube reference freq
        self.restvelrange= centering_cube_in_line(self.header, self.restfreq, self.velref)
        # Velocities radio convention at line rest frequency
        if linefreq_center:
            if linefreq_center < 1e3: 
                # llinefreq_center is in GHz we need it in Hz
                linefreq_center = linefreq_center*1e9
            
            self.linevelrange = centering_cube_in_line(self.header, linefreq_center, self.velref)
    def centerinline(self, linefreq):
        # Modifying velocity range
        self.linevelrange = centering_cube_in_line(self.header, linefreq, self.velref)
        
    def Moments(self, velrange, sigma_level, rangetype='velocity'):
        """
        Moment 0 (i.e. integrated intensity between 2 velocities)
        Moment 1 (i.e. velocity map)
        Moment 2 (i.e. dispersion map)
        """
        if rangetype=='channels':
            # velocities are channels instead
            self.linevelrange = self.channelrange
            velrange.sort()
            chan_i = np.where(self.channelrange == velrange[0])[0][0]
            chan_f = np.where(self.channelrange == velrange[1])[0][0]
            
        else:
            if rangetype=='velocity':
                # Finding channels of that velocity (i.e. min velocity channel respect to that given)
                variable = self.linevelrange
                velrange.sort()
                select = [0,1]
            elif rangetype=='frequency':
                # Finding channels of that rest frequency (i.e. min freq channel respect to that given)
                variable = self.restfreq
                velrange.sort(reverse=True)
                select = [1,0]
            chan_i = np.where((variable <= velrange[0]+self.chanwidth) & (variable >= velrange[0]-self.chanwidth))[0][select[0]]
            chan_f = np.where((variable <= velrange[1]+self.chanwidth) & (variable >= velrange[1]-self.chanwidth))[0][select[1]]
        # Line max between channels
        self.linemax     = np.where(self.fitsdata == np.nanmax(self.fitsdata[chan_i:chan_f+1,:,:]))
        # Moment 0
        self.M0 = self.chanwidth*np.nansum(self.fitsdata[chan_i:chan_f+1,:,:], axis=0) #Jy km/s/beam
        # Integrated sigma
        self.integstd = utiles.integratedsigma(1., (chan_f-chan_i)*self.chanwidth, self.std, self.chanwidth) #Jy km/s/beam
        # Px and Py of max
        self.M0max = np.where(self.M0 == np.nanmax(self.M0))
        # Points above rms (for other moments)
        above_rms_mask = np.ma.masked_where(self.fitsdata[chan_i:chan_f+1,:,:]<=(sigma_level*self.std), self.fitsdata[chan_i:chan_f+1,:,:])
        self.mask= self.M0 >= sigma_level*self.integstd
        self.maskM0 = np.ma.masked_where(self.M0 < sigma_level*self.integstd, self.M0).filled(np.nan)
        
        # Fill with nans
        self.above_rms = above_rms_mask.filled(np.nan)
        self.above_rmsM0 = self.chanwidth*np.nansum(self.above_rms, axis=0)
        # addingdims to velrange
        v_extradims = np.expand_dims(np.expand_dims(self.linevelrange[chan_i:chan_f+1], axis=1), axis=1)
        # Moment 1
        # np.nansum() we only sum over channels with intensity above 3rms 
        self.M1 = (np.nansum(v_extradims*self.above_rms[:,:,:], axis=0)/np.nansum(self.above_rms[:,:,:], axis=0))
        #self.M1 = (np.nansum(v_extradims*self.above_rms[:,:,:], axis=0)/self.M0) # Other definition for M1
        self.M1[~self.mask] = np.nan
        # Moment 2
        # we only sum over channels with intensity above 3rms 
        self.M2 = np.sqrt(np.nansum(((v_extradims-self.M1)**2)*self.above_rms[:,:,:], axis=0)/np.nansum(self.above_rms[:,:,:], axis=0))
        self.M2[~self.mask] = np.nan
        
class Spectra_Handler:    
    'Open 1-dim fits file and access its attributes'
    def __init__(self, name, cube_path, linefreq_center=False, rms=False):
        self.path = cube_path
        self.name = name
        fitsfile  = fits.open(self.path)
        fitsdata  = fitsfile[0]
        self.header = fitsdata.header
        #if self.header['NAXIS3'] == 1:
        #    # Only 1 spectral dimension (i.e. one plane)
        self.fitsdata   = fitsdata.data
        self.shape       = fitsdata.data.shape
        self.wcs         = WCS(self.header, naxis=2) 
        self.velref      = self.header['VELREF'] #km/s
        self.restfreq    = self.header['RESTFREQ']
        # Computing beamsize as pi*bmin*bmaj/(4log2)
        self.bmaj = self.header['BMAJ']  #DEG
        self.bmin = self.header['BMIN']  #DEG
        self.bpa  = self.header['BPA']   #DEG
        self.beamsize = utiles_alma.beam_size(self.bmin, self.bmaj) #DEG**2
        self.pxlen = self.header['CDELT1'] # DEG # pixscale in x
        self.pylen = self.header['CDELT2'] # DEG # pixscale in y
        self.pxinbeam = np.abs(self.beamsize/(self.pxlen*self.pylen))
        if rms==False:
            self.std     = np.nanstd(self.fitsdata)
        elif isinstance(rms, list):
            # measuring rms in linefree channels
            self.std     = np.nanstd(self.fitsdata[rms[0]:rms[1],:,:])
        else:
            self.std     = rms
        self.max         = np.nanmax(self.fitsdata)
        self.min         = np.nanmin(self.fitsdata)
        self.cdelt3      = self.header['CDELT3']
        self.totalchans  = self.shape[1]
        # Channel width in km/s
        self.chanwidth   = np.abs(u_conversion.channel_width_freq_to_kms(self.cdelt3, self.restfreq).value)
        # Channel range  
        self.channelrange= np.arange(1, self.shape[1]+1, 1)
        # Observed freq range
        self.obsfreq     = cube_freqrange(self.header)
        # Rest freq range
        self.restfreq    = u_conversion.freq_to_resfreq(self.obsfreq, self.velref)
        # Velocities radio convention at cube reference freq
        self.restvelrange= centering_cube_in_line(self.header, self.restfreq, self.velref)
        # Velocities radio convention at line rest frequency
        if linefreq_center:
            self.linevelrange = centering_cube_in_line(self.header, linefreq_center, self.velref)
    def centerinline(self, linefreq):
        # Modifying velocity range
        self.linevelrange = centering_cube_in_line(self.header, linefreq, self.velref)
        

def ang_distance_btw_2points(ra1, dec1, ra2, dec2):
    '''
    Angular distance between two points
    '''
    dec1 = np.deg2rad(dec1)
    dec2 = np.deg2rad(dec2)
    ra1 = np.deg2rad(ra1)
    ra2 = np.deg2rad(ra2)
    num = np.sqrt(((np.cos(dec2))**2)*((np.sin(ra2-ra1))**2) + (np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2)
    den = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
    dist = (180./np.pi)*np.arctan(num/den)
    return dist

def ang_distance_btw_2points_v2(ra1, dec1, ra2, dec2):
    '''
    Angular distance between two points
    '''
    dec1 = np.deg2rad(dec1)
    dec2 = np.deg2rad(dec2)
    ra1 = np.deg2rad(ra1)
    ra2 = np.deg2rad(ra2)
    cosdist = np.cos(90-dec1)*np.cos(90-dec2)+np.sin(90-dec1)*np.sin(90-dec2)*np.cos(ra1-ra2)
    dist = np.arccos(cosdist)
    return dist

def centering_cube_in_line(head, line_freq, gal_vel):
    """
    centers cube velocities around a given freq, taking into account the galaxy velocity in km/s
    """
    c = c = _si.c.to(u.km / u.s).value # speed of light in km/s
    freq_range = np.arange(head['CRVAL3'],head['NAXIS3']*head['CDELT3']+head['CRVAL3'],head['CDELT3'])
    F0=line_freq # Line rest frequency in Hz
    vradio_range=c*(F0-freq_range)/F0-gal_vel # definicion radio
    return vradio_range

def cube_freqrange(header):
    freq_range = np.arange(header['CRVAL3'],header['NAXIS3']*header['CDELT3']+header['CRVAL3'],header['CDELT3'])
    return freq_range

def moment_cube_maker(cube, spw, freq_str, line_str, galaxy, location, save_path, molec,
                                      line_df, blend_line = '', plot_moments=False, save_moments=False, sigma_level=3):
    """
      Cube Moment maker for specific line given rms, line freq and channel range
      Return cube with its moments
         if line_df is not a df, line_df = [rms, line_freq_center in GHz, [chans_ini, chans_fin], vel]
    """
    print(f'\nCreating moments map for: {cube} with spw {spw}')
    if isinstance(line_df, pd.DataFrame):
        if molec == 'HC3N':
            cuant = line_str.split(molec+'_')[-1]
        else: 
            cuant = line_str
        print(line_df)
        rms = line_df['rms_spw_'+spw+'_'+freq_str+'GHz'][0]
        linefreq_center = line_df['freq_'+cuant][0]
        velrange = [line_df['chans_ini_'+line_str][0], line_df['chans_fin_'+line_str][0]]
        ref_vel = line_df['HC3N_v0_VLSR_kms'][0]
    elif isinstance(line_df, list):
        rms = line_df[0]
        linefreq_center = line_df[1]
        velrange = line_df[2]
        ref_vel = line_df[4]
    else:
        raise ValueError('Input line_df is not a valid type')
    if np.isnan(velrange).any():
        print(f'Skipping moment for line {line_str}, channels not defined')
        cubo = np.array([])
    else:
        print(f'\tMaking {line_str} moments')
        cubo = Cube_Handler(line_str, cube, rms=rms, linefreq_center = linefreq_center, vel=ref_vel)
        cubo.Moments(velrange=velrange, sigma_level=sigma_level, rangetype='channels')
        if save_moments:
            outname = galaxy+'_'+location+'_'+line_str+blend_line
            # Saving moments as fits
            for m,mom in enumerate([cubo.M0, cubo.M1, cubo.M2]):
                data = mom[None, None, ...]
                hdu = fits.PrimaryHDU(data, cubo.header)
                if m==0:
                    hdu.header['BUNIT'] = 'Jy/beam km/s'
                    hdu.header['INTSIGMA'] = cubo.integstd
                else:
                    hdu.header['BUNIT'] = 'km/s'
                hdul = fits.HDUList([hdu])
                hdul.writeto(save_path+outname+'_M'+str(m)+'_aboveM0.fits',overwrite='True')
            if plot_moments:
                utiles_plot.plot_cube_moments(cubo, save_path=save_path+'figures', outname=outname+'_'+str(m), cbar0=False, cbar1=False)
    return cubo

def cube_moment_ratio_maker(cube_1_str, cube_2_str, moments_path, line_df, ratios_path, suff=''):
            """
            Ratio between cube_1 and cube 2 (cube1/cube2)

            """
            if 'aboveM0' in cube_1_str:
                cube_name1 = cube_1_str.split('HC3N_')[-1].split('_M0_aboveM0.fits')[0]
            else:
                cube_name1 = cube_1_str.split('HC3N_')[-1].split('_M0.fits')[0]
            out_str = ''
            if 'v5' in cube_name1:
                cube_name1 = cube_name1
            elif cube_name1.count('=')>1:
                cube_name1 = cube_name1.split('_v')[0]
                out_str = '_v6_blended'
            cube_1 = Cube_Handler(cube_name1, moments_path+cube_1_str, linefreq_center = line_df['freq_'+cube_name1][0])
            print(cube_1_str)
            print(cube_2_str)
            if 'aboveM0' in cube_2_str:
                cube_name2 = cube_2_str.split('HC3N_')[-1].split('_M0_aboveM0.fits')[0]
            else:
                cube_name2 = cube_2_str.split('HC3N_')[-1].split('_M0.fits')[0]
            if 'v5' in cube_name2:
                cube_name2 = cube_name2
            elif cube_name2.count('=')>1:
                cube_name2 = cube_name2.split('_v')[0]
                out_str = '_v6_blended'
            cube_2 = Cube_Handler(cube_name2, moments_path+cube_2_str, linefreq_center = line_df['freq_'+cube_name2][0])
            # Looping through cube with more pixels # i.e. 39-38
            ratio_ndarray = deepcopy(cube_1.fitsdata)
            for py in range(cube_1.shape[2]):
                for px in range(cube_1.shape[3]):
                    # Coordinates from pixels in cube 1
                    RA, Dec =  RADec_position(px, py, cube_1.wcs, origin=1)
                    # Pixels in cube 2 from coordinates corresponding to px in cube 1
                    px_2, py_2 =  px_position(RA, Dec, cube_2.wcs, origin=1)
                    px_i2 = int(np.round(px_2, 0))
                    py_i2 = int(np.round(py_2, 0))
                    # Extracting value from each px and cube
                    value_1 = cube_1.fitsdata[:, :, py, px][0][0]
                    if px_i2<=0 or py_i2<=0:
                        # out of bounds
                        value_2 = 0
                    elif py_i2 >= cube_2.shape[2] or px_i2 >= cube_2.shape[3]:
                        # out of bounds
                        value_2 = 0
                    else:
                        value_2 = cube_2.fitsdata[:, :, py_i2, px_i2][0][0]
                    # Only adding px values to pixels with 39-38 & 24-23 > 3 sigma
                    if (value_2 >= 3*cube_2.header['INTSIGMA']) and (value_1 >= 3*cube_1.header['INTSIGMA']):
                        ratio_ndarray[:, :, py, px][0][0] = value_1/value_2
                    else:
                        ratio_ndarray[:, :, py, px][0][0] = 0
                    # Saving spectral_index_fits and continuum
            hdu = fits.PrimaryHDU(ratio_ndarray, cube_1.header)
            hdul = fits.HDUList([hdu])
            hdul.writeto(ratios_path+'NGC253_'+cube_name1+'_'+cube_name2+out_str+'_ratio'+suff+'.fits', overwrite=True)

def cube_moment_ratio_maker_nolineinfo(cube_1_str, cube_2_str, moments_path, ratios_path, suff=''):
    """
    Ratio between cube_1 and cube 2 (cube1/cube2) without giving info of the molecular lines
    
    """
    if 'aboveM0' in cube_1_str:
        cube_name1 = cube_1_str.split('HC3N_')[-1].split('_M0_aboveM0.fits')[0]
    else:
        cube_name1 = cube_1_str.split('HC3N_')[-1].split('_M0.fits')[0]
    out_str = ''
    if 'v5' in cube_name1:
        cube_name1 = cube_name1
    elif cube_name1.count('=')>1:
        cube_name1 = cube_name1.split('_v')[0]
        out_str = '_v6_blended'
    cube_1 = Cube_Handler(cube_name1, moments_path+cube_1_str)
    print(cube_1_str)
    print(cube_2_str)
    if 'aboveM0' in cube_2_str:
        cube_name2 = cube_2_str.split('HC3N_')[-1].split('_M0_aboveM0.fits')[0]
    else:
        cube_name2 = cube_2_str.split('HC3N_')[-1].split('_M0.fits')[0]
    if 'v5' in cube_name2:
        cube_name2 = cube_name2
    elif cube_name2.count('=')>1:
        cube_name2 = cube_name2.split('_v')[0]
        out_str = '_v6_blended'
    cube_2 = Cube_Handler(cube_name2, moments_path+cube_2_str)
    # Looping through cube with more pixels # i.e. 39-38
    ratio_ndarray = deepcopy(cube_1.fitsdata)
    for py in range(cube_1.shape[2]):
        for px in range(cube_1.shape[3]):
            # Coordinates from pixels in cube 1
            RA, Dec =  RADec_position(px, py, cube_1.wcs, origin=1)
            # Pixels in cube 2 from coordinates corresponding to px in cube 1
            px_2, py_2 =  px_position(RA, Dec, cube_2.wcs, origin=1)
            px_i2 = int(np.round(px_2, 0))
            py_i2 = int(np.round(py_2, 0))
            # Extracting value from each px and cube
            value_1 = cube_1.fitsdata[:, :, py, px][0][0]
            if px_i2<=0 or py_i2<=0:
                # out of bounds
                value_2 = 0
            elif py_i2 >= cube_2.shape[2] or px_i2 >= cube_2.shape[3]:
                # out of bounds
                value_2 = 0
            else:
                value_2 = cube_2.fitsdata[:, :, py_i2, px_i2][0][0]
            # Only adding px values to pixels with 39-38 & 24-23 > 3 sigma
            if (value_2 >= 3*cube_2.header['INTSIGMA']) and (value_1 >= 3*cube_1.header['INTSIGMA']):
                ratio_ndarray[:, :, py, px][0][0] = value_1/value_2
            else:
                ratio_ndarray[:, :, py, px][0][0] = 0
            # Saving spectral_index_fits and continuum
    hdu = fits.PrimaryHDU(ratio_ndarray, cube_1.header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(ratios_path+'NGC253_'+cube_name1+'_'+cube_name2+out_str+'_ratio'+suff+'.fits', overwrite=True)
    return 'NGC253_'+cube_name1+'_'+cube_name2+out_str+'_ratio'+suff+'.fits'
            
def sigma_mask(cube_fitsdata, sigma, sigma_level):
    """
    Creates a mask given a sigma threshold
    """
    cube_sigmamask = np.ma.masked_where(cube_fitsdata <= sigma_level*sigma, cube_fitsdata, copy=True)
    return cube_sigmamask

# Deprecated since legacycontour is no longer supported!!!
# =============================================================================
# def above_sigma_points(data, sigma, sigma_level):
#     """
#     Function to get all points inside a contour without plotting it
#     returns a boolean mask, to plot it:
#         all_mask_values = np.ma.masked_where(~all_mask, data)
#         plt.imshow(all_mask_values)
#     """
#     x, y = np.meshgrid(np.arange(0, data.shape[0], 1), np.arange(0, data.shape[1], 1))
#     # Reshape mesh to Nx2
#     points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
#     # Generate contours without plotting
#     c = cntr.Cntr(x,y,data)
#     # Trace a contour at z == sigma_level*sigma
#     res = c.trace(sigma_level*sigma)
#     nseg = len(res) // 2
#     # List of arrays of vertices and path codes
#     segments,  = res[:nseg], res[nseg:]
#     all_mask = np.zeros(data.shape)
#     indiv_masks = []
#     for segment in segments:
#         # Generate a path from polygon vertices
#         path = Path(segment)
#         # Points inside path
#         m_inside = path.contains_points(points)
#         # Reshaping mask to data shape
#         m_inside.shape = x.shape
#         # Indivudual masks
#         indiv_masks.append(m_inside)
#         # Boolean logic to all masks (sum of Trues of all paths)
#         all_mask = np.logical_or(m_inside,all_mask)
#     return all_mask, m_inside
# =============================================================================

def integrated_area(data, above_sigma_mask, pxinbeam=1):
    """
    Sum spectra of a defined area
    """
    above_extradims = np.expand_dims(above_sigma_mask, axis=0)
    # Expanding dimensions to same as spectral axis
    above_extradims_ok = above_extradims*np.full((data.shape[0], 1,1),True)
    all_mask_value = np.ma.masked_where(~above_extradims_ok, data)
    all_mask_value_s = np.nansum(all_mask_value, axis=(1,2)) # Madcuba luego divide por el nº de px integrados
    all_mask_value_s_beam = all_mask_value_s/pxinbeam
    return all_mask_value_s, all_mask_value_s_beam

def Beam_mask(cube, px, py):
    """
    Mask inside beam
    pixel size
    position angle (deg)
    bmaj ellipse_major (arcsec)
    bmin ellipse_minor (arcsec)
    """
    bmin=cube.bmin
    bmaj=cube.bmaj
    pa  =cube.bpa
    pixsize = np.abs(cube.header['CDELT1'])
    # Beam
    if pa < 0:
        angle = 90 - pa
    else:
        angle = 90 +pa
    ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                    angle=angle, edgecolor=None, facecolor=None, linewidth=None)
    x, y = np.meshgrid(np.arange(0, cube.M0.shape[0], 1), np.arange(0, cube.M0.shape[1], 1))
    points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
    mask_inside = ellipse.contains_points(points)
    mask_inside.shape = x.shape
    return mask_inside

def Ellipse_mask(cube, RA, Dec, bmin_arcsec, bmaj_arcsec, angle_deg, wcs):
    """
    RA, Dec position of ellipse center
    """
    if angle_deg < 0:
        angle = 90. - angle_deg
    else:
        angle = 90. + angle_deg
    from matplotlib.patches import Ellipse
    # Ra Dec to px and py 
    px, py = px_position(RA, Dec, wcs)
    # Pixel size from cube header
    pixsize = np.abs(cube.header['CDELT1'])
    if cube.header['CUNIT1']=='deg':
            pixsize = np.abs(cube.header['CDELT1'])*3600.
    ellipse = Ellipse(xy=(px, py), width=bmaj_arcsec/pixsize, height=bmin_arcsec/pixsize,
                    angle=angle, edgecolor=None, facecolor=None, linewidth=None)
    x, y = np.meshgrid(np.arange(0, cube.header['NAXIS1'], 1), np.arange(0, cube.header['NAXIS2'], 1))
    points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
    mask_inside = ellipse.contains_points(points)
    mask_inside.shape = x.shape
    return mask_inside

def Ellipse_mask_fromfit(cube, x0, y0, bmin_arcsec, bmaj_arcsec, angle_deg, wcs):
    """
    RA, Dec position of ellipse center
    """
    if angle_deg < 0:
        angle = 90. - angle_deg
    else:
        angle = 90. + angle_deg
    from matplotlib.patches import Ellipse
    # Ra Dec to px and py 
    px, py = x0, y0
    # Pixel size from cube header
    pixsize = np.abs(cube.header['CDELT1'])
    if cube.header['CUNIT1']=='deg':
            pixsize = np.abs(cube.header['CDELT1'])*3600.
    ellipse = Ellipse(xy=(px, py), width=bmaj_arcsec/pixsize, height=bmin_arcsec/pixsize,
                    angle=angle, edgecolor=None, facecolor=None, linewidth=None)
    x, y = np.meshgrid(np.arange(0, cube.header['NAXIS1'], 1), np.arange(0, cube.header['NAXIS2'], 1))
    points = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))
    mask_inside = ellipse.contains_points(points)
    mask_inside.shape = x.shape
    return mask_inside

def createAnnularMask(dimy, dimx, center, big_radius, small_radius):
    """
    Anular selection of big_radius - small_radius width
    """
    Y, X = np.ogrid[:dimy, :dimx]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = (small_radius < distance_from_center) &  (distance_from_center <= big_radius)
    return mask, distance_from_center

def createAnnularMask_1px(dimy, dimx, center, radius):
    """
    Anular selection of 1px width
    """
    Y, X = np.ogrid[:dimy, :dimx]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = ((radius-1) < distance_from_center) &  (distance_from_center <= radius)
    return mask, distance_from_center

def px_position(RA, Dec, wcs, origin = 0):
    """
    Returns the px and py position of J2000 RA Dec  coordiantes
    RA = '02_42_40.83423'
    Dec = '-00_00_33.11504'
    """
    # Checking if its given in str format
    if all(isinstance(i, str) for i in [RA, Dec]):
        if '_' in RA:
            pos = utiles.HMS2deg(ra=RA.replace('_', ' '), dec=Dec.replace('_', ' '))
        elif ':' in RA:
            pos = utiles.HMS2deg(ra=RA.replace(':', ' '), dec=Dec.replace(':', ' '))
        else:
            pos = (RA, Dec)
    else:
        pos=(RA, Dec)
    # There is 1px difference due to the origin of index between numpy and fits. 
    # I have to especify this in the origin
    px, py = wcs.wcs_world2pix(float(pos[0]), float(pos[1]), origin)
    return px, py

def RADec_position(px, py, wcs, origin = 1):
    """
    Returns the RA and Dec position in J2000 coordiantes from px and py
    """
    RA, Dec = wcs.wcs_pix2world(px, py, origin)
    return RA, Dec

def pxspectra_from_RADec(RA, Dec, cube):
    """
    Returns the pixel spectra for a given RA and Dec
    Should return only one value if cube has only one plane (i.e. continuum)
    """
    px, py = px_position(RA, Dec, cube.wcs)
    spectra = cube.fitsdata[:, :, int(np.round(px)), int(np.round(py))]
    return spectra

def findmax_in_cube(cube):
    """
    Returns the maximum and its pixel position given a cube treated with cube_handler
    
    """
    max_px_positions = np.unravel_index(np.argmax(cube.fitsdata, axis=None), cube.fitsdata.shape)
    max_px = max_px_positions[3]
    max_py = max_px_positions[2]
    max_value = np.nanmax(cube.fitsdata)
    return max_px, max_py, max_value

def linear_offset_coords(wcs, center):
    """
    Returns a locally linear offset coordinate system.
    
    Given a 2-d celestial WCS object and a central coordinate, return a WCS
    that describes an 'offset' coordinate system, assuming that the
    coordinates are locally linear (that is, the grid lines of this offset
    coordinate system are always aligned with the pixel coordinates, and
    distortions from spherical projections and distortion terms are not taken
    into account)
    
    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The original WCS, which should be a 2-d celestial WCS
    center : `~astropy.coordinates.SkyCoord`
        The coordinates on which the offset coordinate system should be
        centered.
    """
    # Convert center to pixel coordinates
    xp, yp = skycoord_to_pixel(center, wcs)
    # Set up new WCS
    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = xp + 1, yp + 1
    new_wcs.wcs.crval = 0., 0.
    new_wcs.wcs.cdelt = proj_plane_pixel_scales(wcs)*3600.
    new_wcs.wcs.ctype = 'XOFFSET', 'YOFFSET'
    new_wcs.wcs.cunit = 'arcsec', 'arcsec'
    return new_wcs

def rotate2(degs, header):
    """Return a rotation matrix for counterclockwise rotation by ``deg`` degrees."""
    rads = np.radians(degs)
    s = np.sin(rads)
    c = np.cos(rads)
    return np.array([[c*header['CDELT1'], -s*header['CDELT2']],
                  [s*header['CDELT1'], c*header['CDELT2']]])