from astrothesispy.radiative_transfer import NGC253HR_nLTE_modelresults

presen_figs = False
paper_figs = True

# For presen images
if presen_figs:
    line_column = {'plot_conts': [],
                'v=0_26_25_SM': [15, r'$v=0$  26 - 25', [-10, 111], 1], 
                'v7=1_24_1_23_-1_SM': [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 111], 6],
                'v7=1_26_1_25_-1_SM': [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 111], 2],
                'v6=1_24_-1_23_1_SM': [15, r'$v_{6}=1$  24,-1 - 23,1', [-4, 54], 10],
                'v6=1_26_-1_25_1_SM': [15, r'$v_{6}=1$  26,-1 - 25,1', [-4, 54], 3],
                'v7=2_24_0_23_0_SM':  [15, r'$v_{7}=2$  24,0 - 23,0', [-4, 54], 11],
                'v7=2_26_0_25_0_SM':  [15, r'$v_{7}=2$  26,0 - 25,0', [-4, 54], 4],
                'v5=1_v7=3_26_1_0_25_-1_0_SM':  [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-4, 34], 7],
                'v6=v7=1_26_2_25_-2_SM': [10, r'$v_{6}=v_{7}=1$  26,2,2 - 25,-2,2', [-4, 34],  8],
                'v4=1_26_25_SM': [10, r'$v_{4}=1$  26 - 25', [-3, 23], 5],
                'v6=2_24_0_23_0_SM':  [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 23], 9],
                }

# Lines for paper
if paper_figs:
    line_column = {'plot_conts': [],
                'v=0_26_25_SM': [15, r'$v=0$  26 - 25', [-10, 150], 1], 
                'v7=1_24_1_23_-1_SM': [15, r'$v_{7}=1$  24,1 - 23,-1', [-10, 150], 6],
                'v7=1_26_1_25_-1_SM': [15, r'$v_{7}=1$  26,1 - 25,-1', [-10, 150], 2],
                'v6=1_24_-1_23_1_SM': [15, r'$v_{6}=1$  24,-1 - 23,1', [-10, 74], 10],
                'v6=1_26_-1_25_1_SM': [15, r'$v_{6}=1$  26,-1 - 25,1', [-10, 74], 3],
                'v7=2_24_0_23_0_SM':  [15, r'$v_{7}=2$  24,0 - 23,0', [-5, 74], 11],
                'v7=2_26_0_25_0_SM':  [15, r'$v_{7}=2$  26,0 - 25,0', [-5, 74], 4],
                'v5=1_v7=3_26_1_0_25_-1_0_SM':  [10, r'$v_{5}=1/v_7=3$  26,1,0 - 25,-1,0', [-3, 39], 7],
                'v6=v7=1_26_2_25_-2_SM': [10, r'$v_{6}=v_{7}=1$  26,2,2 - 25,-2,2', [-3, 39],  8],
                'v4=1_26_25_SM': [10, r'$v_{4}=1$  26 - 25', [-3, 29], 5],
                'v6=2_24_0_23_0_SM':  [10, r'$v_{6}=2$  24,0 - 23,0', [-3, 29], 9],
                }

for l,line in enumerate(line_column):
    if line != 'plot_conts':
        # Cont error already included in line error!  
        new_hb_df[line+'_mJy_kms_beam_orig_errcont'] = new_hb_df[line+'_mJy_kms_beam_orig_err']#np.sqrt(new_hb_df[line+'_mJy_kms_beam_orig_err']**2 + new_hb_df['cont_'+line+'_beam_orig_err']**2)
        new_hb_df[line+'_mJy_kms_beam_345_errcont'] = new_hb_df[line+'_mJy_kms_beam_345_err']#np.sqrt(new_hb_df[line+'_mJy_kms_beam_345_err']**2 + new_hb_df['cont_'+line+'_beam_345_err']**2)
        