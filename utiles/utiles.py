#==============================================================================
#                    
#                   Utils: Recurrent formulas and functions
#
#==============================================================================

import numpy as np
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.modeling import models, fitting


def iterate(iterable):
    iterator = iter(iterable)
    item = next(iterator)
    for next_item in iterator:
        yield item, next_item
        item = next_item
    yield item, 0
            
def get_attributes(clase):
    # Get attributes of a class
    return list(clase.__dict__.keys())

def format_e(n):
    """
     To print numbers in scientific form
    """
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def dif_btw_listelemnt(lista):
    return [j-i for i, j in zip(lista[:-1], lista[1:])]

def ratio_btw_listelemnt(lista): 
    # inverse ratio (i+1)/(i)
    return [j/i for i, j in zip(lista[:-1], lista[1:])]

def ratio_btw_listelemnt_asc(lista): 
    # ratio (i)/(i+1)
    return [i/j for i, j in zip(lista[:-1], lista[1:])]

def str_btw_listelemnt_asc(lista): 
    # ratio (i)/(i+1)
    return [i+'_'+j for i, j in zip(lista[:-1], lista[1:])]

def mean_btw_listelemnt(lista):
    return [(j+i)/2. for i, j in zip(lista[:-1], lista[1:])]

def checkEqual(iterator):
    '''
    Checks if all elements in a list are the same
    '''
    return len(set(iterator)) <= 1

def find_nearest(array, value):
    """
        Find nearest to value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# 
def find_between( s, first, last ):
    """
        Find string between characters
    """
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def rounding_exp(a, decimals):
    b = np.log10(a)
    exp = int(b)
    num = 10.**(b-int(b))
    r_num = np.round(num, decimals)
    r_final = r_num*10**exp
    return r_final

def round_to_multiple(number, base):
    nearest_multiple = base * round(number/base)
    return nearest_multiple
        
def binnings(lista_x, lista_y, binsize):
    # el rango lo hago -zbinsize y +2*zbinsize para que al pitnarlo con step no se me corte la anhura del "bin"
    bins = np.arange(min(lista_x)-binsize, max(lista_x)+binsize+binsize, binsize)
    binned = [] # binned list
    for i, val in enumerate(bins):
        binned.append([])
        for j, lob in enumerate(lista_x):
            if val-(binsize/2.0) <= lob <= val+(binsize/2.0):
                binned[i].append(binned[j])
    return (bins, binned)

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def merge_ranges(ranges):
    """
    Merge overlapping and adjacent ranges and yield the merged ranges
    in order. The argument must be an iterable of pairs (start, stop).

    >>> list(merge_ranges([(5,7), (3,5), (-1,3)]))
    [(-1, 7)]
    >>> list(merge_ranges([(5,6), (3,4), (1,2)]))
    [(1, 2), (3, 4), (5, 6)]
    >>> list(merge_ranges([]))
    []
    """
    ranges = iter(sorted(ranges))
    current_start, current_stop = next(ranges)
    for start, stop in ranges:
        if start > current_stop:
            # Gap between segments: output current segment and start a new one.
            yield current_start, current_stop
            current_start, current_stop = start, stop
        else:
            # Segments adjacent or overlapping: merge.
            current_stop = max(current_stop, stop)
    yield current_start, current_stop

def val_is_outrange(lista_x, lista_y, range_min, range_max):
    inrange_x = []
    inrange_y = []
    for i, val in enumerate(lista_x):
        if val <= range_min or val >= range_max:
            inrange_x.append(lista_x[i])
            inrange_y.append(lista_y[i])
    return (inrange_x, inrange_y)
    
def val_is_inrange(lista_x, lista_y, range_min, range_max):
    inrange_x = []
    inrange_y = []
    for i, val in enumerate(lista_x):
        if val >= range_min and val <= range_max:
            inrange_x.append(lista_x[i])
            inrange_y.append(lista_y[i])
    return (inrange_x, inrange_y)

def get_numbers_from_filename(filename):
    '''
    Return numbers in filename
    '''
    return re.search(r'\d+', filename).group(0)

def HMS2deg(ra='', dec=''):
  RA, DEC, rs, ds = '', '', 1, 1
  if dec:
    D, M, S = [float(i) for i in dec.split()]
    if str(D)[0] == '-':
      ds, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    DEC = '{0}'.format(deg*ds)
  
  if ra:
    H, M, S = [float(i) for i in ra.split()]
    if str(H)[0] == '-':
      rs, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    RA = '{0}'.format(deg*rs)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

def deg2HMS(ra='', dec='', round=False):
  RA, DEC, rs, ds = '', '', '', ''
  if dec:
    if str(dec)[0] == '-':
      ds, dec = '-', abs(float(dec))
    deg = int(float(dec))
    decM = abs(int((dec-deg)*60))
    if round:
      decS = int((abs((float(dec)-deg)*60)-decM)*60)
    else:
      decS = (abs((float(dec)-deg)*60)-decM)*60
    DEC = '{0}{1} {2} {3}'.format(ds, deg, decM, decS)
  
  if ra:
    if str(ra)[0] == '-':
      rs, ra = '-', abs(float(ra[1::]))
    raH = int(float(ra)/15)
    raM = int(((float(ra)/15)-raH)*60)
    if round:
      raS = int(((((float(ra)/15)-raH)*60)-raM)*60)
    else:
      raS = ((((float(ra)/15)-raH)*60)-raM)*60
    RA = '{0}{1} {2} {3}'.format(rs, raH, raM, raS)
  
  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0, niter=100):
    """
    Bootstrapping method
    
    p0 matrix with inital guess parameters
    datax and datay input data
    function: fun = lambda p, x : p[0]*x+p[1] for example, p are the parameters
    yerr_systematic = error for each point
    """
    errfunc = lambda p, x, y: function(p,x) - y
    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)

    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # niter random data sets are generated and fitted
    ps = []
    for i in range(niter):
        randomDelta = np.random.normal(0., np.abs(sigma_err_total), len(datay))
        randomdataY = datay + randomDelta
        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)
        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap

def weight_mean_err(values, weights):
    # https://stats.stackexchange.com/questions/252162/se-of-weighted-mean
    var = np.var(values)*np.nansum(weights**2)/(np.nansum(weights)**2)
    return np.sqrt(var)

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def gaussian_fit(datax, datay, mean_0, stddev_0):
    """
    Fitting to a Gaussian
    datax and datay must be np.arrays
    mean_0      -> mean initial guess
    stddev_0    -> stddev initial guess
    Returns [amplitude, mean, sigma, fwhm], [amplitude_err, mean_err, sigma_err, fwhm_err]
    """
    # Checking if data type is np.array or pd.Series
    if type(datax) not in (np.ndarray, np.array, pd.Series) or type(datay) not in (np.ndarray, np.array, pd.Series):
        raise ValueError('Input is not np.ndarray or pd.Series')
    # Defining initial gaussian and fitting
    g_init = models.Gaussian1D(amplitude=datay.max(), mean = mean_0, stddev=stddev_0)
    #g_init.stddev.bounds = 1.e-100, None
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, datax, datay)
    # Parameters and errors
    g_params = g.parameters
    cov_matrix = fit_g.fit_info['param_cov']
    if cov_matrix is None:
        g_errors = [np.nan, np.nan, np.nan]
    else:
        g_errors = np.sqrt(np.diag(cov_matrix))
    # Full width at half maximun
    g_fwhm = 2. * np.sqrt(2* np.log(2)) * g_params[2]
    g_fwhm_err = 2. * np.sqrt(2* np.log(2)) * g_errors[2]
    g_params = np.append(g_params, g_fwhm)
    g_errors = np.append(g_errors, g_fwhm_err)
    return g_params, g_errors, cov_matrix

def deconv_sizes(obs_size, beamsize):
    # Simple size deconvolution
    real_size = np.sqrt(obs_size**2-beamsize**2)
    return real_size

def stdev_to_fwhm(std, std_err):
    g_fwhm = 2. * np.sqrt(2* np.log(2)) * std
    g_fwhm_err = 2. * np.sqrt(2* np.log(2)) * std_err
    return g_fwhm, g_fwhm_err

def fwhm_to_stdev(fwhm, fwhm_err):
    g_std = fwhm/ (2. * np.sqrt(2* np.log(2)))
    g_std_err = fwhm_err / (2. * np.sqrt(2* np.log(2)))
    return g_std, g_std_err

def gaussian_area(amplitude, stddev, amplitude_err, stddev_err):
    """
    Calculates de area unde a Gaussian and its error using:
        Area = Ampl * stddev * sqrt(2 * pi)
    """
    g_area = amplitude * stddev * np.sqrt(2*np.pi)
    g_area_err = np.sqrt(((stddev * np.sqrt(2*np.pi) * amplitude_err)**2.) +
                ((amplitude * np.sqrt(2*np.pi) * stddev_err)**2.))
    return g_area, g_area_err

def add_jitter(df, xvar, yvar, x_step=0.5, y_step=0.5):
    # xcenter
    x_centre = df[xvar].median(skipna=True)
    # sort the dataset in descending order
    df.sort_values(by=[yvar],ascending=False,inplace=True)
    # add an extra column to record the x values
    df['x'] = pd.Series(np.zeros(len(df)), index=df.index)
    # determine the minimum and maximum number of y_steps that encapsulates the data
    ymin=int(df[yvar].min()/y_step)
    ymax=int(df[yvar].max()/y_step)+2
    
    # now step through the bands of data
    for iy in range(ymin,ymax):
        # create an array of Booleans identifying which points lie in the current range
        points_in_range=((df[yvar] > (iy*y_step)) & (df[yvar] <= (iy+1)*y_step))
        # count the number of points in the current range
        num_points = np.sum(points_in_range)
        if num_points>1:
            if (num_points % 2)==0:
                # if there are an even number create the positive side (which here is [1,2])
                a=np.arange(1,(num_points/2)+1,1)
            else:
                # otherwise if there are an odd number create the positive side (which here is [0,1,2])
                a=np.arange(0,int(num_points/2.)+1,1)
            # then the negative side which is [-1,-2]
            b=np.arange(-1,int(num_points/-2.)-1,-1)
            # now create a new array that can hold both,
            c=np.empty((a.size + b.size,), dtype=a.dtype)
            # ..and interweave them
            c[0::2] = a
            c[1::2] = b
            df.loc[points_in_range,'x']=(c*x_step)+x_centre
        else:
            df.loc[points_in_range,'x']=x_centre
    return df
    
def fit_g_bootstrap(datax, datay, data_rms, g_params, g_errors, nboost, seed):
    """
    For bootstrapping with fitted gaussian parameters
    datax       -> x original data
    datay       -> y original data
    g_params    -> amplitude, mean, stddev, fwhm (gaussian parameters)
    g_err       -> amplitude_err, mean_err, stddev_err, fwhm_err (gaussian parameters errors)
    Mirar como definir atributos
    """
    # Setting seed
    np.random.seed(seed)
    boot_param = []
    boot_error = []
    boot_amp = []
    boot_mean = []
    boot_std = []
    boot_w = []
    print('\t\tStarting Bootsrap')
    for i in range(nboost):
        resample = np.random.choice(range(len(datay)), len(datay), replace=True)
        x_boots = [datax[i] for i in resample]
        y_boots = [datay[i] for i in resample]
        # Montecarlo
        #randomDelta = np.random.normal(0., data_rms, len(y_boots))
        #y_resampled = y_boots + randomDelta
        y_resampled = np.random.normal(y_boots, data_rms)
        #print '\tSimul: ' + str(i)
        gg_params, gg_errors, g_cov_mat = gaussian_fit(np.array(x_boots), np.array(y_resampled), np.mean(x_boots), np.std(x_boots))
        boot_param.append(gg_params)
        boot_error.append(gg_errors)
        if not np.any(pd.isnull(gg_params)):
            boot_amp.append(gg_params[0])
            boot_mean.append(gg_params[1])
            boot_std.append(gg_params[2])
            boot_w.append(gg_params[3])
        else:
            continue
    print('\t\tBootsrap finished')
    bins_num = int(np.round(np.sqrt(nboost/2.), decimals=0))
    # Width of the Amplitud distribution
    (n_amp, bins_amp) = np.histogram(boot_amp, bins=bins_num)
    amp_params, amp_errors, amp_cov_mat = gaussian_fit(np.array(mean_btw_listelemnt(bins_amp)), n_amp, g_params[0], g_errors[0])
    # Width of the Mean distribution
    (n_mean, bins_mean) = np.histogram(boot_mean, bins=bins_num) 
    mean_params, mean_errors, mean_cov_mat = gaussian_fit(np.array(mean_btw_listelemnt(bins_mean)), n_mean, g_params[1], g_errors[1])
    #Width of the Stddev distribution
    (n_std, bins_std) = np.histogram(boot_std, bins=bins_num)
    std_params, std_errors, std_cov_mat = gaussian_fit(np.array(mean_btw_listelemnt(bins_std)), n_std, g_params[2], g_errors[2])
    #Width of the FWHM distribution
    (n_w, bins_w) = np.histogram(boot_w, bins=bins_num)
    w_params, w_errors, w_cov_mat = gaussian_fit(np.array(mean_btw_listelemnt(bins_w)), n_w, g_params[3], g_errors[3])
    # Parameter Results
    param_boots = [amp_params, mean_params, std_params, w_params]
    param_boots_err = [amp_errors, mean_errors, std_errors, w_errors]
    # Parameter Distributions
    amp_dist = [mean_btw_listelemnt(bins_amp), n_amp]
    mean_dist = [mean_btw_listelemnt(bins_mean), n_mean]
    std_dist = [mean_btw_listelemnt(bins_std), n_std]
    w_dist = [mean_btw_listelemnt(bins_w), n_w]
    param_dist = [amp_dist, mean_dist, std_dist, w_dist]
    param_boot_vals = [boot_amp, boot_mean, boot_std, boot_w]
    return param_boots, param_boots_err, param_dist, param_boot_vals
    
def fit_g_bootstrap_plotter(g_params, g_errors, param_boots, param_boots_err, param_dist, param_boot_vals, out_fig_dir, galaxy, line):
    # Disables displaying plots
    param_names = ['Amp', 'Mean', 'Std', 'FWHM']
    for i, par in enumerate(param_boots):
        fig = plt.figure()
        ax = fig.add_subplot((111))
        ax.plot(param_dist[i][0], param_dist[i][1], linewidth=0.5, drawstyle='steps-mid', color='k', label='') 
        b_gauss_amp = models.Gaussian1D(amplitude=param_boots[i][0], mean=param_boots[i][1], stddev=param_boots[i][2])
        ax.plot(param_dist[i][0], b_gauss_amp(param_dist[i][0]), linewidth=0.8, color='r', label='', linestyle='--' )
        ax.errorbar(g_params[i], 0.3*param_boots[i][0], xerr=g_errors[i], marker='|', markersize=3, color='b', elinewidth=0.5, capsize=1., capthick=0.6, label='No Boots')
        ax.errorbar(param_boots[i][1], 0.25*param_boots[i][0], xerr=param_boots[i][2], marker='|', markersize=3, color='r', elinewidth=0.5, capsize=1., capthick=0.6,label='Boots')
        ax.set_xlabel(param_names[i])
        plt.legend(loc='best', fontsize='xx-small', facecolor=None, frameon=False)
        plt.savefig(out_fig_dir+'/'+galaxy+'_'+line+'_boots_'+param_names[i]+'.png', bbox_inches='tight', transparent=True, dpi=600)
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot((111))
        ax.scatter(np.arange(1, len(param_boot_vals[i])+1.), param_boot_vals[i], label='', facecolors='none', edgecolors='k', s=3, linewidth=0.5)
        ax.errorbar(0.4*len(param_boot_vals[i]), g_params[i], yerr=g_errors[i], marker='.', markersize=3, color='b', elinewidth=0.5, capsize=1., capthick=0.6, label='No Boots')
        ax.errorbar(0.6*len(param_boot_vals[i]), param_boots[i][1], yerr=param_boots[i][2], marker='.', markersize=3, color='r', elinewidth=0.5, capsize=1., capthick=0.6,label='Boots')
        ax.set_ylabel(param_names[i])
        plt.legend(loc='best', fontsize='xx-small', facecolor=None, frameon=False)
        plt.savefig(out_fig_dir+'/'+galaxy+'_'+line+'_boots_simval_'+param_names[i]+'.png', bbox_inches='tight', transparent=True, dpi=600)
        plt.close()
        plt.close('all')

def function_fit(func, xdata, ydata, ini_pars):
    popt, pcov = optimize.curve_fit(func, xdata, ydata)
    # Errors 
    popt_errors = np.sqrt(np.diag(pcov))
    return popt, popt_errors


def numformatter(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


class data:
    """
    To create a dictionary
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def integratedsigma(sigma_level, fwhm, rms, channel_res_kms):
    """
    Calculate the integrated sigma level
    """
    xsigma = sigma_level*rms*fwhm/np.sqrt(fwhm/channel_res_kms)
    return xsigma

def Beam_plotter(px, py, bmin, bmaj, pixsize, pa, axis, color, wcs, linewidth=0.4, rectangle=True):
    """
    Plotting beam
    pixel size
    position angle (deg)
    bmaj ellipse_major (arcsec)
    bmin ellipse_minor (arcsec)
    """
    from matplotlib.patches import Ellipse
    from matplotlib.patches import Rectangle
    # Beam
    if pa < 0:
        angle = 90 - pa
    else:
        angle = 90 +pa
    ellipse = Ellipse(xy=(px, py), width=bmaj/pixsize, height=bmin/pixsize,
                    angle=angle, edgecolor=color, facecolor=color, linewidth=0.1,
                    transform=axis.get_transform(wcs))
    axis.add_patch(ellipse)
    if rectangle:        
        # Rectangle Border
        r_ax1 = bmaj*1.4
        r_ax2 = bmaj*1.4
        r = Rectangle(xy=(px-r_ax2/pixsize/2, py-r_ax2/pixsize/2), width=r_ax1/pixsize,
                      height=r_ax2/pixsize, edgecolor=color, facecolor='none', linewidth=linewidth,
                  transform=axis.get_transform(wcs))
        axis.add_patch(r)

def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.full(reference.shape, False)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def linfit(x,m,b):
        """
        m = -1/(T*np.log(10))
        b = np.log10(Ntot/Z)
        x = Eu/k
        y = np.log10(Nu/gu)
        """
        return m*x+b

def weighted_avg_and_stdv2(values, values_err):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    
    """
    masked_data = np.ma.masked_array(values, np.isnan(values))
    masked_errs = np.ma.masked_array(values_err, np.isnan(values_err))
    weights = 1/(values_err)
    
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def latex_float(f):
    """
        Returns string with latex exponent format
    """
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return rf'{np.float(base):1.1f}\times10^{{{int(exponent):1.0f}}}'
    else:
        return float_str