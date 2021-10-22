

#######################
#  y6_shearcat_tests  #
#######################

import os, sys
from re import I
from tqdm import tqdm
import numpy as np
import fitsio as fio

# Combine PIFF tables.
def combine_piff(bands, work_piff, tilenames, all=False):
    
    if not all:
        for band in bands:
            model = []
            for t in tqdm(tilenames):
                if not os.path.exists(os.path.join(work_piff, band+'_band/'+t+'_piff_model.fits')):
                    continue
                model.append(fio.read(os.path.join(work_piff, band+'_band/'+t+'_piff_model.fits')))
            
            fio.write(os.path.join(work_piff, band+'_band/master_'+band+'_piff_models.fits'), np.concatenate(model, axis=0))
    else:
        r = fio.read(os.path.join(work_piff, 'r_band/master_r_piff_models.fits'))
        i = fio.read(os.path.join(work_piff, 'i_band/master_i_piff_models.fits'))
        z = fio.read(os.path.join(work_piff, 'z_band/master_z_piff_models.fits'))
        fio.write(os.path.join(work_piff, 'master_all_piff_models.fits'), np.concatenate([r, i, z], axis=0))

# combine gold catalogs split by healpix ids.
def combine_gold(test_region, work_gold):

    import healpy as hp

    gold_mag = []
    # npix = hp.nside2npix(nside)
    for i in range(test_region):
        for split in range(5):
            if os.path.exists(os.path.join(work_gold, 'gold_2_0_magnitudes_'+str(i)+'_'+str(split).zfill(6)+'.fits')):
                gold_mag.append(fio.read(os.path.join(work_gold, 'gold_2_0_magnitudes_'+str(i)+'_'+str(split).zfill(6)+'.fits')))
    gold_all = np.concatenate(gold_mag, axis=0)
    fio.write(os.path.join(work_gold, 'y6_gold_2_0_magnitudes.fits'), gold_all)


# Figure 2; Median measured FWHM for the PSF stars
def figure2():
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

# Figure 4; galaxy count, shear response, variance of e, shear weight as a function of S/N and size ratio.
def figure4(steps, mdet_cat):

    import os
    np.random.seed(1738)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from math import log10
    import pylab as mplot
    import matplotlib.ticker as ticker

    font = {'size'   : 13}
    mplot.rc('font', **font)
    mplot.rc('text', usetex=False)
    mplot.rc('font', family='serif')

    def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
        # return x and y indices of data (x,y) on a log-spaced grid that runs from [xy]min to [xy]max in [xy]steps
        
        logstepx = log10(xmax/xmin)/xsteps
        logstepy = log10(ymax/ymin)/ysteps
        
        indexx = (np.log10(x/xmin)/logstepx).astype(int)
        indexy = (np.log10(y/ymin)/logstepy).astype(int)
        
        indexx = np.maximum(indexx,0)
        indexx = np.minimum(indexx, xsteps-1)
        indexy = np.maximum(indexy,0)
        indexy = np.minimum(indexy, ysteps-1)
        
        return indexx,indexy

    def apply_loggrid(x, y, grid, xmin=0, xmax=0, xsteps=0, ymin=0, ymax=0, ysteps=0):
        indexx,indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
        res = np.zeros(len(x))
        res = grid[indexx,indexy]
        return res

    def logmeshplot(data, xedges, yedges, label="quantity"):
        fig=plt.figure(figsize=(6,6))
        ax = plt.subplot(111)
        X, Y = np.meshgrid(yedges, xedges)
        plt.pcolormesh(X, Y, data)
        plt.xscale('log')
        plt.yscale('log')
        plt.colorbar(label=label)
        plt.ylabel("mcal snr")
        plt.xlabel("mcal size/psf_size")

        plt.minorticks_off() 
        ax.set_xticks(np.array([0.6,0.7,0.8,0.9,1.,2.,3.,4.,]))
        ax.set_xticklabels(np.array([r'6 x $10^{-1}$','','','',r'$10^{0}$',r'2 x $10^{0}$','',r'4 x $10^{0}$']))

    def mesh_average(quantity,indexx,indexy,steps,count):
        m = np.zeros((steps,steps)) # revised version, was -1 before
        np.add.at(m,(indexx,indexy),quantity)
        m /= count
        return m

    def find_assign_grid(d, mdet_step, snmin, snmax, steps, sizemin, sizemax):

        mask = ((d['flags'] == 0) & (d['mdet_s2n'] > 10) & (d['mdet_T_ratio'] > 1.2) & (d['mfrac'] < 0.1) & (d['mdet_step'] == mdet_step))
        mastercat_snr = d[mask]['mdet_s2n']
        mastercat_Tr = d[mask]['mdet_T_ratio']
        new_indexx,new_indexy = assign_loggrid(mastercat_snr, mastercat_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
        
        return new_indexx, new_indexy, mask

    def find_bincount_2d(indexx, indexy, steps):

        from collections import Counter
        index_tup = [(i,j) for i,j in zip(indexx, indexy)]
        count = Counter(index_tup)
        all_count = np.zeros((steps, steps))
        for i in range(steps):
            for j in range(steps):
                if count[(i,j)] != 0:
                    all_count[i,j] = count[(i,j)]
        
        return all_count


    def mesh_response(d, snmin, snmax, steps, sizemin, sizemax):
        
        g1p_indexx, g1p_indexy, mask_1p = find_assign_grid(d, '1p', snmin, snmax, steps, sizemin, sizemax)
        g1p_count = find_bincount_2d(g1p_indexx, g1p_indexy, steps)
        g1m_indexx, g1m_indexy, mask_1m = find_assign_grid(d, '1m', snmin, snmax, steps, sizemin, sizemax)
        g1m_count = find_bincount_2d(g1m_indexx, g1m_indexy, steps)
        g2p_indexx, g2p_indexy, mask_2p = find_assign_grid(d, '2p', snmin, snmax, steps, sizemin, sizemax)
        g2p_count = find_bincount_2d(g2p_indexx, g2p_indexy, steps)
        g2m_indexx, g2m_indexy, mask_2m = find_assign_grid(d, '2m', snmin, snmax, steps, sizemin, sizemax)
        g2m_count = find_bincount_2d(g2m_indexx, g2m_indexy, steps)

        g_1p = np.zeros((steps, steps))
        g_1m = np.zeros((steps, steps))
        g_2p = np.zeros((steps, steps))
        g_2m = np.zeros((steps, steps))
        np.add.at(g_1p, (g1p_indexx, g1p_indexy), d[mask_1p]['mdet_g'][:,0])
        np.add.at(g_1m, (g1m_indexx, g1m_indexy), d[mask_1m]['mdet_g'][:,0])
        np.add.at(g_2p, (g2p_indexx, g2p_indexy), d[mask_2p]['mdet_g'][:,1])
        np.add.at(g_2m, (g2m_indexx, g2m_indexy), d[mask_2m]['mdet_g'][:,1])
        g_1p /= g1p_count
        g_1m /= g1m_count
        g_2p /= g2p_count
        g_2m /= g2m_count

        R11 = (g_1p - g_1m)/0.02
        R22 = (g_2p - g_2m)/0.02
        new_response = (R11+R22)/2

        return new_response
    
    d = fio.read(mdet_cat)
    mask_noshear = ((d['flags'] == 0) & (d['mdet_s2n'] > 10) & (d['mdet_T_ratio'] > 1.2) & (d['mfrac'] < 0.1) & (d['mdet_step'] == 'noshear'))
    mastercat_noshear_snr = d[mask_noshear]['mdet_s2n']
    mastercat_noshear_Tr = d[mask_noshear]['mdet_T_ratio']
    new_e1 = d[mask_noshear]['mdet_g'][:,0]
    new_e2 = d[mask_noshear]['mdet_g'][:,1]

    snmin=10
    snmax=300
    sizemin=1.2
    sizemax=3
    steps=steps
    new_indexx,new_indexy = assign_loggrid(mastercat_noshear_snr, mastercat_noshear_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
    new_count = np.zeros((steps,steps))
    np.add.at(new_count,(new_indexx,new_indexy), 1)
    H, xedges, yedges = np.histogram2d(mastercat_noshear_snr, mastercat_noshear_Tr, bins=[np.logspace(log10(snmin),log10(snmax),steps+1), np.logspace(log10(sizemin),log10(sizemax),steps+1)])
    new_response = mesh_response(d, snmin, snmax, steps, sizemin, sizemax)
    new_meanes = mesh_average(np.sqrt((new_e1**2+new_e2**2)/2),new_indexx,new_indexy,steps,new_count)
    new_shearweight = (new_response/new_meanes)**2

    # count
    fig=plt.figure(figsize=(6,15))
    ax = plt.subplot(411)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_count/1.e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    #im.colorbar(label="count")
    im.axes.get_xaxis().set_visible(False)
    fig.colorbar(im, ax=ax, label=r"count [$10^5$]")

    # Eq 9. 
    ax = plt.subplot(412)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_meanes)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    im.axes.get_xaxis().set_visible(False)
    fig.colorbar(im, ax=ax, label=r"$\sqrt{\langle e_{1,2}^2\rangle}$")

    ax = plt.subplot(413)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_response, vmin=0.1, vmax=1.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    im.axes.get_xaxis().set_visible(False)
    fig.colorbar(im, ax=ax, label=r"$\langle$R$\rangle$")

    ax = plt.subplot(414)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_shearweight, vmin=0, vmax=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    plt.xlabel("galaxy size / PSF size")
    im.axes.get_xaxis().get_ticklabels()[3].set_visible(False)
    fig.colorbar(im, ax=ax, label="shear weight")

    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.minorticks_off() 
    ax.set_xticks(np.array([1.2,1.3,1.4,1.5,1.6,1.7,2.,3.]))
    ax.set_xticklabels(np.array([r'1.2 x $10^{0}$','','','','','',r'2 x $10^{0}$',r'3 x $10^{0}$']))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    plt.savefig('count_response_ellip_SNR_Tr_v2.png')


def bin_exposures_quartile(good_piff_models, par):

    good = fio.read(good_piff_models)
    expnumq = np.unique(good['EXPNUM'])
    quantity=[]
    for u in expnumq:
        seeing=np.unique(good[good['EXPNUM']==u][par])[0]
        quantity.append(seeing)
    quartiles = np.quantile(quantity, [0., 0.25, 0.50, 0.75, 1.0])
    quartiles_exp = []
    for q in range(len(quartiles)-1):
        mask = ((good[par] >= quartiles[q]) & (good[par] < quartiles[q+1]))
        quartiles_exp.append(good[mask]['EXPNUM'])

    return quartiles, quartiles_exp

# Figure 6; PSF residuals as a function of star magnitudes (BFE)
def figure6(gold_cat, data, band): #piff_cat_i, piff_cat_z):

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
def figure7(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z):

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


# Figure 11; tangential and cross-component shear around bright and faint stars. 
def figure11(mdet_cat, piff_all_cat):

    import treecorr
    from matplotlib import pyplot as plt

    def flux2mag(flux, zero_pt=30):
        return zero_pt - 2.5 * np.log10(flux)
    
    def _make_cuts(d, shear):

        msk = (
            (d['flags'] == 0)
            & (d['mdet_s2n'] > 10)
            & (d['mdet_T_ratio'] > 1.2)
            & (d['mfrac'] < 0.1)
            & (d['mdet_step'] == shear)
        )
        
        return np.mean(d['mdet_g'][msk, :], axis=0), msk

    def calculate_response(d):

        g_noshear, mask_noshear = _make_cuts(d, 'noshear')
        g_1p, mask_1p = _make_cuts(d, '1p')
        g_1m, mask_1m = _make_cuts(d, '1m')
        g_2p, mask_2p = _make_cuts(d, '2p')
        g_2m, mask_2m = _make_cuts(d, '2m')

        R11 = (g_1p[0] - g_1m[0])/0.02
        R22 = (g_2p[1] - g_2m[1])/0.02
        R = [R11, R22]

        return R, mask_noshear

    shear_catalog = fio.read(mdet_cat)
    full_response, mask_noshear = calculate_response(shear_catalog)
    ra = shear_catalog[mask_noshear]['ra']
    dec = shear_catalog[mask_noshear]['dec']
    g1 = shear_catalog[mask_noshear]['mdet_g'][:,0]/full_response[0]
    g2 = shear_catalog[mask_noshear]['mdet_g'][:,1]/full_response[1]

    # data_r = fio.read(os.path.join(work_piff, 'r_band/master_r_piff_models.fits'))
    # data_i = fio.read(os.path.join(work_piff, 'i_band/master_i_piff_models.fits'))
    # data_z = fio.read(os.path.join(work_piff, 'z_band/master_z_piff_models.fits'))
    data_piff = fio.read(piff_all_cat)

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 1.0,
        max_sep = 250,
        nbins = 20,

        var_method = 'jackknife'
    )

    ########################
    # bright stars; m<16.5 #
    ########################
    data_mag = flux2mag(data_piff['FLUX'])
    mask_bright = (data_mag<16.5)
    star_catalog_bright = data_piff[mask_bright]
    ra_piff = star_catalog_bright['RA']
    dec_piff = star_catalog_bright['DEC']

    cat1 = treecorr.Catalog(ra=ra_piff, dec=dec_piff, ra_units='deg', dec_units='deg', npatch=10)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, npatch=10)
    ng_bright = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_bright.process(cat1, cat2)

    #######################
    # faint stars; m>16.5 #
    #######################
    mask_faint = (data_mag>16.5)
    star_catalog_faint = data_piff[mask_faint]
    ra_piff = star_catalog_faint['RA']
    dec_piff = star_catalog_faint['DEC']

    cat1 = treecorr.Catalog(ra=ra_piff, dec=dec_piff, ra_units='deg', dec_units='deg', npatch=10)
    ng_faint = treecorr.NGCorrelation(bin_config, verbose=2)
    ng_faint.process(cat1, cat2)

    fig, axes = plt.subplots(1,2, figsize=(15,7))
        
    ax = axes[0]
    ax.errorbar(ng_bright.meanr, ng_bright.xi, yerr=np.sqrt(ng_bright.varxi), fmt='o', label=r'$\gamma_{\rm t}$')
    ax.errorbar(ng_bright.meanr, ng_bright.xi_im, yerr=np.sqrt(ng_bright.varxi), fmt='x', label=r'$\gamma_{\rm x}$')
    ax.set_ylabel(r'$\gamma_{\rm t}(\theta), \gamma_{\rm x}(\theta)$', fontsize='xx-large')
    # ax.set_xlim(min_mag,max_mag)
    ax.set_title('Bright stars [m<16.5]', fontsize='x-large' )
    ax.legend(prop={'size': 16},loc='lower right')
    ax.set_xlabel(r'$\theta [arcmin]$', fontsize='xx-large' )
    ax.set_xscale('log')

    ax = axes[1]
    #ax.set_ylim(-0.002,0.002)
    ax.errorbar(ng_faint.meanr, ng_faint.xi, yerr=np.sqrt(ng_faint.varxi), fmt='o', label=r'$\gamma_{\rm t}$')
    ax.errorbar(ng_faint.meanr, ng_faint.xi_im, yerr=np.sqrt(ng_faint.varxi), fmt='x', label=r'$\gamma_{\rm x}$')
    #ax.plot(mag_bins[:-1], fit_fn(mag_bins[:-1]), '--k')
    ax.set_ylabel(r'$\gamma_{\rm t}(\theta), \gamma_{\rm x}(\theta)$', fontsize='xx-large')
    # ax.set_xlim(min_mag,max_mag)
    ax.set_title('Faint stars [m>16.5]', fontsize='x-large' )
    # ax.set_ylim(-0.007,0.0125)
    ax.set_xlabel(r'$\theta [arcmin]$', fontsize='xx-large' )
    ax.set_xscale('log')

    fig.suptitle('Tangential shear around stars', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('tangential_shear_around_stars.png')


# Figure 14; Tangential shear around field center
def figure14():

    import treecorr
    from matplotlib import pyplot as plt
    import tqdm
    import esutil as eu
    import json
    import pickle
    from tqdm import tqdm

    CCD28 = np.array([12289, 14336, 12289, 16384])
    CCD35 = np.array([14337, 16384, 12289, 16384])
    
    def _make_cuts(d, shear):

        msk = (
            (d['flags'] == 0)
            & (d['mdet_s2n'] > 10)
            & (d['mdet_T_ratio'] > 1.2)
            & (d['mfrac'] < 0.1)
            & (d['mdet_step'] == shear)
        )
        
        return np.mean(d['mdet_g'][msk, :], axis=0), msk

    def calculate_response(d):

        g_noshear, mask_noshear = _make_cuts(d, 'noshear')
        g_1p, mask_1p = _make_cuts(d, '1p')
        g_1m, mask_1m = _make_cuts(d, '1m')
        g_2p, mask_2p = _make_cuts(d, '2p')
        g_2m, mask_2m = _make_cuts(d, '2m')

        R11 = (g_1p[0] - g_1m[0])/0.02
        R22 = (g_2p[1] - g_2m[1])/0.02
        R = [R11, R22]

        return R, mask_noshear
    
    def find_field_centres(work_mdet, tilenames, centre_x, centre_y, bands=['r','i','z'], save=False):

        ################################################################################################################
        ## THIS IS BEING LEFT UNDONE.                                                                                 ##
        ## TASK:  MODIFY THE CODE TO READ IN EXPOSURE NUMBER FILE AND CENTER RA, DEC FILE MADE IN QUERY_GOOD_PIFF.PY. ##
        ################################################################################################################
        if save:
            output = {}
            en_list = []
            for tilename in tqdm(tilenames):
                f_mdet = os.path.join(work_mdet, tilename+'_metadetect-v3_mdetcat_part0000.fits')

                coadd_band_exist = {band: None for band in bands}
                for band in bands:
                    f = os.path.join(work_pizza, band+'_band/'+tilename+'_r5227p01_'+band+'_pizza-cutter-slices.fits.fz')
                    if os.path.exists(f):
                        coadd_band_exist[band] = f
                    else:
                        f = os.path.join(work_pizza, band+'_band/'+tilename+'_r5227p03_'+band+'_pizza-cutter-slices.fits.fz')
                        if os.path.exists(f):
                            coadd_band_exist[band] = f
                
                if (not os.path.exists(f_mdet)) or np.all(np.where(list(coadd_band_exist.values()))):
                    print('this tilename ', tilename, ' does not have either mdet cat or coadd info.')
                    continue

                mdet_cat = fio.read(f_mdet)
                for obj in range(len(mdet_cat)):
                    for band in bands:
                        if coadd_band_exist[band] is None:
                            continue
                        coadd = fio.FITS(coadd_band_exist[band])
                        epochs = coadd['epochs_info'].read()
                        image_info = coadd['image_info'].read()
                        slice_id = mdet_cat['slice_id'][obj]
                        single_epochs = epochs[epochs['id']==slice_id]
                        file_id = single_epochs[single_epochs['flags']==0]['file_id']

                        for f in enumerate(file_id):
                            en = int(image_info['image_path'][f][5:13].lstrip("0"))
                            if en not in en_list:
                                en_list.append(en)
                                # wcs = eu.wcsutil.WCS(json.loads(image_info['wcs'][f]))
                                # position_offset = image_info['position_offset'][f]
                                # output[en] = {'wcs': wcs, 'offset': position_offset}
                # for e in en:
                    # get ra, dec in ccd28 and 35 for each exposure and average them. 
                
                with open(os.path.join(work_pizza, 'expnum_wcs_riz.pickle'), 'wb') as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(work_pizza, 'expnum_wcs_riz.pickle'), 'rb') as handle:
                output = pickle.load(handle)

        radec_centres = np.zeros(len(list(output.keys())), dtype=[('ra', float), ('dec', float)])
        for i, expnum in enumerate(list(output.keys())):
            wcs = output[expnum]['wcs']
            position_offset = output[expnum]['offset']
            ra, dec = wcs.image2sky(centre_x+position_offset, centre_y+position_offset)
            radec_centres['ra'][i] = ra
            radec_centres['dec'][i] = dec

        return radec_centres
            

    shear_catalog = fio.read(os.path.join(work_mdet, 'mdet_test_all.fits'))
    full_response, mask_noshear = calculate_response(shear_catalog)
    ra = shear_catalog[mask_noshear]['ra']
    dec = shear_catalog[mask_noshear]['dec']
    g1 = shear_catalog[mask_noshear]['mdet_g'][:,0]/full_response[0]
    g2 = shear_catalog[mask_noshear]['mdet_g'][:,1]/full_response[1]
    centre_x = (CCD28[0] + CCD35[1])/2.
    centre_y = (CCD28[2] + CCD35[3])/2.

    if not os.path.exists(os.path.join(work_pizza, 'expnum_wcs_riz.pickle')):
        ra_centres = find_field_centres(work_mdet, tilenames, centre_x, centre_y, save=True)['ra']
        dec_centres = find_field_centres(work_mdet, tilenames, centre_x, centre_y, save=True)['dec']
    else:
        ra_centres = find_field_centres(work_mdet, tilenames, centre_x, centre_y)['ra']
        dec_centres = find_field_centres(work_mdet, tilenames, centre_x, centre_y)['dec']

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 1.0,
        max_sep = 250,
        nbins = 20,

        var_method = 'jackknife'
    )

    cat1 = treecorr.Catalog(ra=ra_centres, dec=dec_centres, ra_units='deg', dec_units='deg', npatch=10)
    cat2 = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=g1, g2=g2, npatch=10)
    ng = treecorr.NGCorrelation(bin_config, verbose=2)
    ng.process(cat1, cat2)

    fig, axes = plt.subplots(figsize=(15,7))
        
    ax = axes[0]
    ax.errorbar(ng, ng*ng.xi, yerr=np.sqrt(ng.varxi), fmt='o')
    ax.set_ylabel(r'$\theta\gamma_{\rm t}(\theta)$', fontsize='xx-large')
    ax.legend(prop={'size': 16},loc='lower right')
    ax.set_xlabel(r'$\theta [arcmin]$', fontsize='xx-large' )
    ax.set_xscale('log')

    # fig.suptitle('Tangential shear around stars', fontsize='x-large')
    plt.tight_layout()
    plt.savefig('tangential_shear_around_field_centres.png')

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
    figure4(20, mdet_cat)
    # figure6(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z)
    # figure7(gold_cat, piff_cat_r, piff_cat_i, piff_cat_z)
    # figure11(mdet_cat, piff_all_cat)
    # figure14() -> modify the input and output. 

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