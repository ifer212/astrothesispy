import os
import numpy as np

from astrothesispy.utiles import Madcuba_plotter as MP
from astrothesispy.utiles import utiles_plot as plot_utiles


source = 'SHC_13'
data_path = f'data/NGC253_HR/Spectra/{source}/rings/'

class ring:
    def __init__(self, source, dist, xlimits, ylimits, data_path, save_path = './',
                 rms = 0, vel = 250, move_lines = {}, rows = 5, columns = 1, figformat = '.pdf'
                ):
        self.source = source
        self.dist = dist
        self.distname =  str(self.dist).replace('.', 'p')
        self.xlimits = xlimits
        self.ylimits = ylimits
        self.move_lines = move_lines
        self.figname = f'{source}_d{self.distname}'
        self.rows = rows
        self.columns = columns
        self.rms = [rms]*self.rows*self.columns
        self.figformat = figformat
        self.vel = vel
        self.molec_info = data_path
        self.data_path = f'{data_path}distpc_{self.distname}/'
        self.save_path = save_path
        
    def ring_plot(self):
        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')
        # Default values for ring figures
        x_label = r'Rest. Freq (GHz)'
        y_label = r'mJy beam$^{-1}$'
        unit = 'mJy'
        spec_color = 'k'
        fill_color = '0.9'
        linewidth = 1.2
        linestyle = '-'
        drawstyle = 'histogram'
        xsep = 5
        panel_naming = False
        # Fit lines
        fit_color = [plot_utiles.redpink, 'b']
        fit_linewidth = 1.4
        fit_linestyle = '-'
        only_sumfit = False
        # Molecule labels
        label_color = 'k'
        highlight_lines = {r'HC3N': 'b'}
        txtypercen = 4/100 # % of panel height to consider labels are close
        txtxpercen = 4/100 # % of panel width to consider labels are close  
        new_labelling = {'CCS': r'C2S', 'CH3OH,vt=0-2': r'CH3OH', 
                 'OCS,v=0': r'OCS', 'SO2,v=0': r'SO2',
                 'CH3CN,v=0': r'CH3CN', 'C2H3CN,v=0': r'C2H3CN',
                 'HC3N_vib': r'HC3N', 'SiO-18,v=0-5':r'SiO-18',
                 'SO,v=0': r'SO', 'SiC2,v=0': r'SiC2'}
        # Fonts
        labelfont = 'Arial'
        labelsize=18
        panelfont = 'courier new'
        panfontsize=14
        anotcolor = '0.2'
        moleculefont = 'courier new'
        molfontize=15
        ticklabelsize = 12
        vel_shift = 257 - self.vel    
        # Calling plot function
        MP.MADCUBA_plot(self.rows, self.columns, self.data_path, self.molec_info, self.save_path, self.figname, self.figformat,
                        self.xlimits, self.ylimits, x_label, y_label,
                        drawstyle, spec_color, linewidth, linestyle, fit_color, fit_linewidth,
                        fit_linestyle, only_sumfit, unit, self.rms, xsep, vel_shift,
                        highlight_lines, label_color, fill_color, labelsize, labelfont,
                        molfontize, moleculefont, anotcolor, panelfont, panfontsize,
                        ticklabelsize, txtypercen, txtxpercen, panel_naming=panel_naming,
                        new_labelling = new_labelling)
        
def ring_create_and_plot(source, data_path, save_path, size = 1.5, step = 0.1):
    """
        Creating and plotting rings
    """
    # Same x axis limits for all rings
    x_limits    = {0: [[218.26,219.28], 0], # spec 0
               1: [[219.34,220.20], 0], # spec 0
               2: [[220.30,221.30], 1], # spec 1
               3: [[236.12,237.18], 2], # spec 2
               4: [[237.20,238.08], 2]  # spec 2
               }
    
    ylimits  = {}
    ylimits['ring1'] = {0: [-0.35/1000, 3.25/1000], # spec 0
                        1: [-0.35/1000, 2.20/1000], # spec 0
                        2: [-0.35/1000, 2.50/1000], # spec 1
                        3: [-0.35/1000, 2.50/1000], # spec 2
                        4: [-0.35/1000, 3.50/1000]  # spec 2
                        }    
    ylimits['ring2'] =  {0: [-0.35/1000, 3.25/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }         
    ylimits['ring3'] =  {0: [-0.35/1000, 3.25/1000], # spec 0   
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }         
    ylimits['ring4'] =  {0: [-0.35/1000, 3.25/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }         
    ylimits['ring5'] =  {0: [-0.35/1000, 3.25/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }         
    ylimits['ring4'] =  {0: [-0.35/1000, 3.25/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }        
    ylimits['ring4'] =  {0: [-0.35/1000, 3.00/1000], # spec 0
                         1: [-0.35/1000, 2.00/1000], # spec 0
                         2: [-0.35/1000, 2.00/1000], # spec 1
                         3: [-0.35/1000, 2.00/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         }         
    ylimits['ring4'] =  {0: [-0.35/1000, 2.25/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.50/1000], # spec 1
                         3: [-0.35/1000, 2.50/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
                         } 
    ylimits['ring4'] =  {0: [-0.35/1000, 2.30/1000], # spec 0
                         1: [-0.35/1000, 2.20/1000], # spec 0
                         2: [-0.35/1000, 2.20/1000], # spec 1
                         3: [-0.35/1000, 2.20/1000], # spec 2
                         4: [-0.35/1000, 2.50/1000]  # spec 2
                         }         
    ylimits['ring4'] =  {0: [-0.35/1000, 2.0/1000], # spec 0
                         1: [-0.35/1000, 2.0/1000], # spec 0
                         2: [-0.35/1000, 2.0/1000], # spec 1
                         3: [-0.35/1000, 2.0/1000], # spec 2
                         4: [-0.35/1000, 2.3/1000]  # spec 2
               }         
    ylimits['ring4'] =  {0: [-0.35/1000, 1.0/1000], # spec 0
                         1: [-0.35/1000, 1.20/1000], # spec 0
                         2: [-0.35/1000, 1.50/1000], # spec 1
                         3: [-0.35/1000, 1.50/1000], # spec 2
                         4: [-0.35/1000, 1.70/1000]  # spec 2
                         }         
    ylimits['ring4'] =  {0: [-0.35/1000, 1.00/1000], # spec 0
                         1: [-0.35/1000, 1.00/1000], # spec 0
                         2: [-0.35/1000, 1.00/1000], # spec 1
                         3: [-0.35/1000, 1.00/1000], # spec 2
                         4: [-0.35/1000, 1.50/1000]  # spec 2
                         }
    ylimits['ring4'] =  {0: [-0.35/1000, 1.00/1000], # spec 0
                         1: [-0.35/1000, 1.00/1000], # spec 0
                         2: [-0.35/1000, 1.00/1000], # spec 1
                         3: [-0.35/1000, 1.00/1000], # spec 2
                         4: [-0.35/1000, 1.50/1000]  # spec 2
                         }
    ylimits['ring4'] =  {0: [-0.35/1000, 1.00/1000], # spec 0
                         1: [-0.35/1000, 1.00/1000], # spec 0
                         2: [-0.35/1000, 1.00/1000], # spec 1
                         3: [-0.35/1000, 1.00/1000], # spec 2
                         4: [-0.35/1000, 1.50/1000]  # spec 2
                         }
    ylimits['ring4'] =  {0: [-0.35/1000, 1.00/1000], # spec 0
                         1: [-0.35/1000, 1.00/1000], # spec 0
                         2: [-0.35/1000, 1.00/1000], # spec 1
                         3: [-0.35/1000, 1.00/1000], # spec 2
                         4: [-0.35/1000, 3.50/1000]  # spec 2
               }
    ylimits['ring15'] = {0: [-0.35/1000, 3.25/1000], # spec 0
                        1: [-0.35/1000, 2.20/1000], # spec 0
                        2: [-0.35/1000, 2.50/1000], # spec 1
                        3: [-0.35/1000, 2.50/1000], # spec 2
                        4: [-0.35/1000, 3.50/1000]  # spec 2
                        }    
    
    
    start_pc = 0
    end_pc = size
    distances_pc = np.arange(start_pc, end_pc+step, step)
    for dist in distances_pc:
        



y_limits    = {0: [-0.35/1000, 1.00/1000], # spec 0
               1: [-0.35/1000, 1.00/1000], # spec 0
               2: [-0.35/1000, 1.00/1000], # spec 1
               3: [-0.35/1000, 1.00/1000], # spec 2
               4: [-0.35/1000, 3.50/1000]  # spec 2
               }         # 3.50

ring_1 = ring(source, dist, xlimits, ylimits, data_path)

