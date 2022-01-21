
import os, sys
from re import I
from tqdm import tqdm
import numpy as np
import fitsio as fio

# Figure 2; Median measured FWHM for the PSF stars
def fwhm_PSF():

    # Input: PIFF files. 

    import ngmix
    from matplotlib import pyplot as plt

    fwhm_r = {}
    fwhm_i = {}
    fwhm_z = {}
    for t in tqdm(tilenames):
        if os.path.exists(os.path.join(work_piff, 'r_band/'+t+'_piff_model.fits')):
            star_r_table = fio.read(os.path.join(work_piff, 'r_band/'+t+'_piff_model.fits'))
            star_i_table = fio.read(os.path.join(work_piff, 'i_band/'+t+'_piff_model.fits'))
            star_z_table = fio.read(os.path.join(work_piff, 'z_band/'+t+'_piff_model.fits'))
        
            star_r_table = star_r_table[star_r_table['STAR_FLAG']==0]
            star_i_table = star_i_table[star_i_table['STAR_FLAG']==0]
            star_z_table = star_z_table[star_z_table['STAR_FLAG']==0]

            for i,expnum in enumerate(star_r_table['EXPNUM']):
                if expnum not in fwhm_r:
                    fw = ngmix.moments.T_to_fwhm(star_r_table[i]['STAR_T'])
                    fwhm_r.update({str(expnum):fw})
                else:
                    fw = ngmix.moments.T_to_fwhm(star_r_table[i]['STAR_T'])
                    np.append(fwhm_r[str(expnum)], fw)
            
            for i,expnum in enumerate(star_i_table['EXPNUM']):
                if expnum not in fwhm_i:
                    fw = ngmix.moments.T_to_fwhm(star_i_table[i]['STAR_T'])
                    fwhm_i.update({str(expnum):fw})
                else:
                    fw = ngmix.moments.T_to_fwhm(star_i_table[i]['STAR_T'])
                    np.append(fwhm_i[str(expnum)], fw)

            for i,expnum in enumerate(star_z_table['EXPNUM']):
                if expnum not in fwhm_z:
                    fw = ngmix.moments.T_to_fwhm(star_z_table[i]['STAR_T'])
                    fwhm_z.update({str(expnum):fw})
                else:
                    fw = ngmix.moments.T_to_fwhm(star_z_table[i]['STAR_T'])
                    np.append(fwhm_z[str(expnum)], fw)
    median_r = [np.median(fwhm_r[fwhm]) for fwhm in list(fwhm_r.keys())]
    median_i = [np.median(fwhm_i[fwhm]) for fwhm in list(fwhm_i.keys())]
    median_z = [np.median(fwhm_z[fwhm]) for fwhm in list(fwhm_z.keys())]
    
    fig,ax=plt.subplots(figsize=(8,6))
    ax.hist(median_r, bins=300, histtype='step', label='r')
    ax.hist(median_i, bins=50, histtype='step', label='i')
    ax.hist(median_z, bins=50, histtype='step', label='z')
    ax.set_xlim(0.6,1.7)
    ax.set_xlabel('Seeing FWHM (arcsec)', fontsize=13)
    ax.set_ylabel('Number of exposures', fontsize=13)
    plt.legend(loc='upper right')
    plt.savefig('seeing_fwhm.png')

# Figure 6; PSF residuals as a function of star magnitudes (BFE)
def brighter_fatter_effect(gold_cat, data, band): #piff_cat_i, piff_cat_z):

    from matplotlib import pyplot as plt

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)

    def match_ra_dec(band, data):

        import smatch
        gold = fio.read(gold_cat)
        nside = 4096
        maxmatch = 1
        radius = 0.263/3600 # degrees

        if len(data[data['RA']>360]) != 0:
            print('Rounding off RA>360')
            mask = (data['RA']>360)
            data['RA'][mask] -= 360
        
        matches = smatch.match(data['RA'], data['DEC'], radius, gold['RA'], gold['DEC'], nside=nside, maxmatch=maxmatch)

        matched_data = data[matches['i1']]
        matched_data_mag = gold[matches['i2']]['BDF_MAG_'+band.capitalize()]
        print(len(matched_data))

        return matched_data, matched_data_mag

    def compute_res(d):

        prefix = 'MODEL'
        de1 =  d['STAR_E1']-d[prefix+'_E1']
        de2 =  d['STAR_E2']-d[prefix+'_E2']
        dt =  (d['STAR_T']-d[prefix+'_T'])
        dtfrac = dt/d['STAR_T']
        print('mean de = ',np.mean(de1),np.mean(de2))
        print('mean dt = ',np.mean(dt))

        return dtfrac, dt,  de1, de2

    def bin_by_mag(m, dT, dTfrac, de1, de2, min_mused):
        min_mag = 15 
        max_mag = 21 
        
        mag_bins = np.linspace(min_mag,max_mag,31)

        index = np.digitize(m, mag_bins)
        bin_de1 = [de1[index == i].mean() for i in range(1, len(mag_bins))]
        bin_de2 = [de2[index == i].mean() for i in range(1, len(mag_bins))]
        bin_dT = [dT[index == i].mean() for i in range(1, len(mag_bins))]
        bin_dTfrac = [dTfrac[index == i].mean() for i in range(1, len(mag_bins))]
        bin_de1_err = [ np.sqrt(de1[index == i].var() / len(de1[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_de2_err = [ np.sqrt(de2[index == i].var() / len(de2[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_dT_err = [ np.sqrt(dT[index == i].var() / len(dT[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_dTfrac_err = [ np.sqrt(dTfrac[index == i].var() / len(dTfrac[index == i]))
                        for i in range(1, len(mag_bins)) ]
        
        # Fix up nans
        for i in range(1,len(mag_bins)):
            if i not in index:
                bin_de1[i-1] = 0.
                bin_de2[i-1] = 0.
                bin_dT[i-1] = 0.
                bin_dTfrac[i-1] = 0.
                bin_de1_err[i-1] = 0.
                bin_de2_err[i-1] = 0.
                bin_dT_err[i-1] = 0.
                bin_dTfrac_err[i-1] = 0.
        
        fit = np.polyfit(mag_bins[:-1], bin_dT, 3)
        fit_fn = np.poly1d(fit) 
        # fit_fn is now a function which takes in x and returns an estimate for y
        return mag_bins, bin_dT, bin_dT_err
        fig, axes = plt.subplots(3,1, sharex=True)
        min_mag=15
        max_mag=21
        min_mused=16.5

        ax = axes[0]
        ax.set_ylim(-0.003,0.005)
        ax.plot([min_mag,max_mag], [0,0], color='black')
        ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        t_line = ax.errorbar(mag_bins[:-1], bin_dT, yerr=bin_dT_err, color='blue', fmt='o',label="all")
        ax.set_ylabel(r'$(T_{\rm *} - T_{\rm model}) \quad({\rm arcsec}^2)$', fontsize='xx-large')

        ax = axes[1]
        ax.set_ylim(-0.007,0.008)
        ax.plot([min_mag,max_mag], [0,0], color='black')
        ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        t_line = ax.errorbar(mag_bins[:-1], bin_dTfrac, yerr=bin_dTfrac_err, color='blue', fmt='o')
        ax.set_ylabel(r'$(T_{\rm *} - T_{\rm model})/ T_{\rm *}$', fontsize='xx-large')

        ax = axes[2]
        ax.set_ylim(-6.e-4,6.e-4)
        ax.plot([min_mag,max_mag], [0,0], color='black')
        ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        e1_line = ax.errorbar(mag_bins[:-1], bin_de1, yerr=bin_de1_err,  color='red', fmt='o', mfc='white',label=r"$\rm{Y6 \quad e_1}$")
        e2_line = ax.errorbar(mag_bins[:-1], bin_de2, yerr=bin_de2_err, color='blue', fmt='o',label=r"$\rm{Y6 \quad e_2}$")
        ax.legend(loc='lower left')
        ax.set_ylabel(r'$e_{\rm *} - e_{\rm model}$', fontsize='xx-large')
        ax.set_xlim(min_mag,max_mag)
        ax.set_xlabel(r'$\rm{Magnitude}$', fontsize='xx-large')
        

        #plt.hist(meddt["Ks"], 30,range=(-0.004,0.002),color="purple",label="Ks",histtype='step')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['mathtext.fontset'] = 'stix'

        fig.set_size_inches(7.0,12.0)
        plt.tight_layout()
        plt.savefig('dpsf_bfe_newcuts_test_v2.png')
        plt.show()

    # data_r = piff_cat_r # fio.read(piff_cat_r)
    # data_i = piff_cat_i # fio.read(piff_cat_i)
    # data_z = piff_cat_z # fio.read(piff_cat_z)
    # data_piff = []
    # data_mag = []
    # for band, data in zip(['r', 'i', 'z'], [data_r, data_i, data_z]):
    #     matched_piff_data, matched_mag_data = match_ra_dec(band, data)
    #     data_piff.append(matched_piff_data)
    #     data_mag.append(matched_mag_data)
    # data_piff = np.concatenate(data_piff, axis=0)
    # data_mag = np.concatenate(data_mag, axis=0)
    data_piff, data_mag = match_ra_dec(band, data)
    
    fracsizeres, sizeres, e1res, e2res = compute_res(data_piff)
    # data_mag = flux2mag(data['FLUX'])
    # bin_by_mag(data_mag, sizeres, fracsizeres, e1res, e2res, 16.5)
    mag_bins, bin_dT, bin_dT_err = bin_by_mag(data_mag, sizeres, fracsizeres, e1res, e2res, 16.5)
    return mag_bins, bin_dT, bin_dT_err

# Figure 7; PSF residuals as a function of color (monochrmoatic PSF)
def psf_monochromaticity(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z):

    from matplotlib import pyplot as plt

    def match_ra_dec_(data1, data2):

        mask1_ra = np.in1d(data1['RA'], data2['RA'])
        mask1_dec = np.in1d(data1['DEC'], data2['DEC'])
        mask1 = mask1_ra * mask1_dec

        mask2_ra = np.in1d(data2['RA'], data1['RA'])
        mask2_dec = np.in1d(data2['DEC'], data1['DEC'])
        mask2 = mask2_ra * mask2_dec

        data1_masked = data1[mask1]
        data2_masked = data2[mask2]

        data1_masked_sorted = data1_masked[np.argsort(data1_masked, order=('RA', 'DEC'))]
        data2_masked_sorted = data2_masked[np.argsort(data2_masked, order=('RA', 'DEC'))]

        return data1_masked_sorted, data2_masked_sorted

    def match_ra_dec_color(band, data):

        import smatch
        gold = fio.read(gold_cat)
        nside = 4096
        maxmatch = 1
        radius = 0.263/3600 # degrees

        if len(data[data['RA']>360]) != 0:
            print('Rounding off RA>360')
            mask = (data['RA']>360)
            data['RA'][mask] -= 360

        matches = smatch.match(data['RA'], data['DEC'], radius, gold['RA'], gold['DEC'], nside=nside, maxmatch=maxmatch)

        matched_data = data[matches['i1']]
        matched_data_mag = gold[matches['i2']]['BDF_MAG_R'] - gold[matches['i2']]['BDF_MAG_Z']

        return matched_data, matched_data_mag

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)

    def compute_res(d):

        prefix = 'MODEL'
        de1 =  d['STAR_E1']-d[prefix+'_E1']
        de2 =  d['STAR_E2']-d[prefix+'_E2']
        dt =  (d['STAR_T']-d[prefix+'_T'])
        dtfrac = dt/d['STAR_T']
        print('mean de = ',np.mean(de1),np.mean(de2))
        print('mean dt = ',np.mean(dt))

        return dtfrac, dt,  de1, de2

    def bin_by_col2(m, T, dTfrac, de1, de2, min_mused):
        min_mag = -0.8 #min(m)
        max_mag = 2.5 #max(m)   
        mag_bins = np.linspace(min_mag, max_mag,31)

        index = np.digitize(m, mag_bins)
        bin_de1 = [de1[index == i].mean() for i in range(1, len(mag_bins))]
        bin_de2 = [de2[index == i].mean() for i in range(1, len(mag_bins))]
        bin_T = [T[index == i].mean() for i in range(1, len(mag_bins))]
        bin_dTfrac = [dTfrac[index == i].mean() for i in range(1, len(mag_bins))]
        bin_de1_err = [ np.sqrt(de1[index == i].var() / len(de1[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_de2_err = [ np.sqrt(de2[index == i].var() / len(de2[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_T_err = [ np.sqrt(T[index == i].var() / len(T[index == i]))
                        for i in range(1, len(mag_bins)) ]
        bin_dTfrac_err = [ np.sqrt(dTfrac[index == i].var() / len(dTfrac[index == i]))
                        for i in range(1, len(mag_bins)) ]
        print(bin_T_err)
        print(bin_T)
        print(len(bin_T_err))
        print(len(bin_T))

        # Fix up nans
        for i in range(1,len(mag_bins)):
            if i not in index:
                bin_de1[i-1] = 0.
                bin_de2[i-1] = 0.
                bin_dTfrac[i-1] = 0.
                bin_de1_err[i-1] = 0.
                bin_de2_err[i-1] = 0.
                bin_dTfrac_err[i-1] = 0.
                
                
        fit = np.polyfit(mag_bins[:-1], bin_dTfrac, 3)
        fit_fn = np.poly1d(fit) 
        # fit_fn is now a function which takes in x and returns an estimate for y
        
        med=np.median(m)
        p20=np.percentile(m,20)
        p80=np.percentile(m,80)
        fig, axes = plt.subplots(1,3)
        
        ax = axes[0]
        #ax.set_ylim(-0.002,0.002)
        #ax.plot([min_mag-0.1,max_mag], [0,0], color='black')
        #ax.plot([min_mused,min_mused],[-1,1], color='Grey')
        #ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        t_line = ax.errorbar(mag_bins[:-1], bin_T, yerr=bin_T_err, color='blue', fmt='o')
        ax.axvline(x=med, linewidth=2, color='grey',label=r"$\rm{median}$")
        ax.axvline(x=p20, linewidth=2, color='grey',linestyle='--',label=r'$\rm{20th \, percentile}$')
        ax.axvline(x=p80, linewidth=2, color='grey',linestyle=':',label=r'$\rm{80th \, percentile}$')
        #ax.legend([t_line], [r'$\delta T$'])
        ax.set_ylabel(r'$T_{\rm *}\quad({\rm arcsec}^2)$', fontsize='xx-large')
        ax.set_xlim(min_mag,max_mag)
        # ax.set_ylim(0.35, 0.41)
        #ax.set_title(r'$riz-{\rm band \, PSF}$', fontsize='x-large' )
        ax.legend(prop={'size': 16},loc='upper right')
        ax.set_xlabel(r'$r-z$', fontsize='xx-large' )

        ax = axes[1]
        #ax.set_ylim(-0.002,0.002)
        ax.plot([min_mag-0.1,max_mag], [0,0], color='black')
        #ax.plot([min_mused,min_mused],[-1,1], color='Grey')
        #ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        t_line = ax.errorbar(mag_bins[:-1], bin_dTfrac, yerr=bin_dTfrac_err, color='blue', fmt='o')
        #ax.plot(mag_bins[:-1], fit_fn(mag_bins[:-1]), '--k')
        #ax.legend([t_line], [r'$\delta T$'])
        ax.axvline(x=med, linewidth=2, color='grey')
        ax.axvline(x=p20, linewidth=2, color='grey',linestyle='--')
        ax.axvline(x=p80, linewidth=2, color='grey',linestyle=':')
        ax.set_ylabel(r'$(T_{\rm *} - T_{\rm model})/ T_{\rm *}$', fontsize='xx-large')
        ax.set_xlim(min_mag,max_mag)
        # ax.set_ylim(-0.007,0.0125)
        ax.set_xlabel(r'$r-z$', fontsize='xx-large' )
        #ax.set_xlim(min_mag, max_mag)

        ax = axes[2]
        #ax.set_ylim(-3.e-4,4.e-4)
        ax.plot([min_mag-0.1,max_mag+0.1], [0,0], color='black')
        #ax.plot([-0.5,min_mused],[-1,1], color='Grey')
        #ax.fill( [min_mag,min_mag,mi1_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
        e1_line = ax.errorbar(mag_bins[:-1], bin_de1, yerr=bin_de1_err,  color='red', fmt='o', mfc='white')
        e2_line = ax.errorbar(mag_bins[:-1], bin_de2, yerr=bin_de2_err, color='blue', fmt='o')
        #ax.axhline(y=0.0002, linewidth=4, color='grey')
        #ax.axhline(y=-0.0002, linewidth=4, color='grey')
        ax.axvline(x=med, linewidth=2, color='grey')  #0.78
        ax.axvline(x=p20, linewidth=2, color='grey',linestyle='--')  #-0.66
        ax.axvline(x=p80, linewidth=2, color='grey',linestyle=':')  #2.24
        ax.legend([e1_line, e2_line], [r'$e_1$', r'$e_2$'], loc='upper right', fontsize='xx-large')
        ax.set_ylabel(r'$e_{\rm *} - e_{\rm model}$', fontsize='xx-large')
        # ax.set_ylim(-0.00075,0.001)
        ax.set_xlim(min_mag,max_mag)
        ax.set_xlabel(r'$r-z$', fontsize='xx-large' )
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams["font.family"] = "Times New Roman"
        
        plt.rcParams['mathtext.fontset'] = 'stix'

        fig.set_size_inches(16.0,4.0)
        plt.tight_layout()
        plt.savefig('dpsf_colour_r-z_test_v2.png')
        plt.show()
        
    data_r = fio.read(piff_cat_r)
    data_i = fio.read(piff_cat_i)
    data_z = fio.read(piff_cat_z)
    data_piff = []
    data_rzmag = []
    for band, data in zip(['r', 'i', 'z'], [data_r, data_i, data_z]):
        matched_piff_data, matched_mag_data = match_ra_dec_color(band, data)
        data_piff.append(matched_piff_data)
        data_rzmag.append(matched_mag_data)
    data_piff = np.concatenate(data_piff, axis=0)
    data_rzmag = np.concatenate(data_rzmag, axis=0)
    fracsizeres, sizeres, e1res, e2res = compute_res(data_piff)
    bin_by_col2(data_rzmag, data_piff['STAR_T'], fracsizeres, e1res, e2res, 15)

def piff_model_qa_cuts_hist(basic_piff_models):

    from matplotlib import pyplot as plt
    basic = fio.read(basic_piff_models)
    fig, axes = plt.subplots(1,4,figsize=(30,7))

    ax = axes[0]
    ax.hist(basic['FWHM_CEN'], bins=5000, histtype='step')
    ax.vlines(3.6, 0, 5000, linestyles='dashed')
    ax.set_xlabel('fwhm_cen', fontsize='xx-large')
    ax.set_ylim(0,600)

    ax = axes[1]
    ax.hist(basic['STAR_T_STD']/basic['STAR_T_MEAN'], bins=1000, range=[0., 0.5], histtype='step')
    ax.vlines(0.3, 0, 5000, linestyles='dashed')
    ax.set_xlabel('star_T_std/star_T_mean', fontsize='xx-large')
    ax.set_ylim(0,5000)

    ax = axes[2]
    ax.hist(basic['NSTAR'], bins=200, histtype='step')
    ax.vlines(25, 0, 20000, linestyles='dashed')
    ax.set_xlabel('nstar', fontsize='xx-large')
    ax.set_ylim(0,13000)

    ax = axes[3]
    ax.hist(abs(basic['STAR_T_MEAN'] - basic['EXP_STAR_T_MEAN'])/basic['EXP_STAR_T_STD'], bins=2000, range=[0., 5.], histtype='step')
    ax.vlines(0.4, 0, 1000, linestyles='dashed')
    ax.set_xlabel('abs(star_T_mean-exp_star_T_mean)/exp_star_T_std', fontsize='xx-large')
    ax.set_ylim(0,800)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.savefig('piff_model_qa_cuts.png')
    plt.show()

def main(argv):

    # f = open('/home/s1/masaya/des-y6-analysis/tiles.txt', 'r')
    # tilenames = f.read().split('\n')[:-1]
    # tilename_delete_list = ['DES0031+0001']
    # tilenames.remove('DES0031+0001')
    work_mdet = os.path.join('/data/des70.a/data/masaya/', 'metadetect/v2')
    work_pizza = os.path.join('/data/des70.a/data/masaya/', 'pizza-slice')
    work_piff = os.path.join('/data/des70.a/data/masaya/', 'piff_models')
    work_gold = os.path.join('/data/des70.a/data/masaya/', 'gold')

    mdet_cat = os.path.join(work_mdet, 'mdet_test_all_v2.fits')
    gold_cat = os.path.join(work_gold, 'y6_gold_2_0_magnitudes.fits')
    good_piff_models = os.path.join(work_piff, 'good_piffs_newcuts_query_test_v2.fits')
    basic_piff_models = os.path.join(work_piff, 'basic_piffs_query_test_v2.fits')
    piff_cat_r = os.path.join(work_piff, 'r_band/master_r_piff_models_newcuts_test_v2.fits')
    piff_cat_i = os.path.join(work_piff, 'i_band/master_i_piff_models_newcuts_test_v2.fits')
    piff_cat_z = os.path.join(work_piff, 'z_band/master_z_piff_models_newcuts_test_v2.fits')
    piff_all_cat = os.path.join(work_piff, 'master_all_piff_models.fits')

    # combine_piff(['r', 'i', 'z'], work_piff, tilenames)
    # combine_gold(32, work_gold)
    # brighter_fatter_effect(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z)
    # psf_monochromaticity(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z)

    ##########################################################
    ## WHEN PLOTTING EXTRA STUFFS LIKE BANDPASS DEPENDENCE. ##
    ##########################################################
    # from matplotlib import pyplot as plt
    # # gold_cat2 = [os.path.join(work_gold, 'y6_gold_2_0_magnitudes_T1.fits'), os.path.join(work_gold, 'y6_gold_2_0_magnitudes_T2.fits'), os.path.join(work_gold, 'y6_gold_2_0_magnitudes_T3.fits'), os.path.join(work_gold, 'y6_gold_2_0_magnitudes_T4.fits')]
    # fig, axes = plt.subplots(3,4, figsize=(30,22), sharex=True)
    # min_mag=15
    # max_mag=21
    # min_mused=16.5
    # bands = ['r', 'i', 'z']
    # piff_cat = [piff_cat_r, piff_cat_i, piff_cat_z]
    # for j,band in enumerate(bands):
    #     for i in range(4):

    #         q, quartiles_exp = bin_exposures_quartile(good_piff_models, 'EXP_STAR_T_MEAN')
    #         data_r = fio.read(piff_cat[j])
    #         data_r = data_r[np.isin(data_r['EXPNUM'], quartiles_exp[i])]
    #         # data_i = fio.read(piff_cat_i)
    #         # data_i = data_i[np.isin(data_i['EXPNUM'], quartiles_exp[i])]
    #         # data_z = fio.read(piff_cat_z)
    #         # data_z = data_z[np.isin(data_z['EXPNUM'], quartiles_exp[i])]
    #         mag_bins, bin_dT, bin_dT_err = figure6(gold_cat, data_r, band) # data_i, data_z)
    #         # mag_bins, bin_dT, bin_dT_err = figure6(gold_cat, piff_cat[i], bands[i])

    #         ax = axes[j,i]
    #         ax.set_ylim(-0.003,0.005)
    #         ax.plot([min_mag,max_mag], [0,0], color='black')
    #         ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
    #         t_line = ax.errorbar(mag_bins[:-1], bin_dT, yerr=bin_dT_err, color='blue', fmt='o',label=str("{:2.2f}".format(q[i]))+' < <T> < '+str("{:2.2f}".format(q[i+1])))
    #         ax.set_ylabel(r'$(T_{\rm *} - T_{\rm model}) \quad({\rm arcsec}^2)$', fontsize='xx-large')
    #         ax.set_xlabel(r'$\rm{Magnitude}$', fontsize='xx-large')
    #         ax.legend(fontsize='x-large')

    #         #plt.hist(meddt["Ks"], 30,range=(-0.004,0.002),color="purple",label="Ks",histtype='step')
    #         plt.rcParams['font.family'] = 'serif'
    #         plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    #         plt.rcParams["font.family"] = "Times New Roman"
    #         plt.rcParams['mathtext.fontset'] = 'stix'

    # # fig.set_size_inches(7.0,12.0)
    # plt.tight_layout()
    # plt.savefig('dpsf_bfe_riz_expTquartiles_newcuts_test_v2.png')


if __name__ == "__main__":
    main(sys.argv)