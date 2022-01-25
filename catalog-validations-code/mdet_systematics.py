

####################################################################
#  y6 shear catalog tests (This does not include PSF diagnostics)  #
####################################################################

import os, sys
from re import I
from tqdm import tqdm
import numpy as np
import fitsio as fio

# Figure 4; galaxy count, shear response, variance of e, shear weight as a function of S/N and size ratio.
def inverse_variance_weight(steps, mdet_cat, more_cuts=None):

    # Input: 
    #   steps = how many bins in each axis.
    #   mdet_cat = metadetection catalog.

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

        mask = ((d['FLAGS'] == 0) & (d['MDET_S2N'] > 10) & (d['MDET_T_RATIO'] > 1.2) & (d['MFRAC'] < 0.1) & (d['MASK_FLAGS'] == 0) & (d['MDET_STEP'] == mdet_step))
        mastercat_snr = d[mask]['MDET_S2N']
        mastercat_Tr = d[mask]['MDET_T_RATIO']
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
        np.add.at(g_1p, (g1p_indexx, g1p_indexy), d[mask_1p]['MDET_G_1'])
        np.add.at(g_1m, (g1m_indexx, g1m_indexy), d[mask_1m]['MDET_G_1'])
        np.add.at(g_2p, (g2p_indexx, g2p_indexy), d[mask_2p]['MDET_G_2'])
        np.add.at(g_2m, (g2m_indexx, g2m_indexy), d[mask_2m]['MDET_G_2'])
        g_1p /= g1p_count
        g_1m /= g1m_count
        g_2p /= g2p_count
        g_2m /= g2m_count

        R11 = (g_1p - g_1m)/0.02
        R22 = (g_2p - g_2m)/0.02
        new_response = (R11+R22)/2

        return new_response
    
    d = fio.read(mdet_cat)
    if more_cuts is None:
        msk = ((d['FLAGS'] == 0) & (d['MDET_S2N'] > 10) & (d['MDET_T_RATIO'] > 1.2) & (d['MFRAC'] < 0.1) & (d['MASK_FLAGS'] == 0))
    else:
        msk_default = ((d['FLAGS'] == 0) & (d['MDET_S2N'] > 10) & (d['MDET_T_RATIO'] > 1.2) & (d['MFRAC'] < 0.1) & (d['MASK_FLAGS'] == 0))
        msk = (more_cuts & msk_default)
    mask_noshear = (msk & (d['MDET_STEP'] == 'noshear'))
    mastercat_noshear_snr = d[mask_noshear]['MDET_S2N']
    mastercat_noshear_Tr = d[mask_noshear]['MDET_T_RATIO']
    new_e1 = d[mask_noshear]['MDET_G_1']
    new_e2 = d[mask_noshear]['MDET_G_2']

    snmin=10
    snmax=1000
    sizemin=1.2
    sizemax=3
    steps=steps
    new_indexx,new_indexy = assign_loggrid(mastercat_noshear_snr, mastercat_noshear_Tr, snmin, snmax, steps, sizemin, sizemax, steps)
    new_count = np.zeros((steps,steps))
    np.add.at(new_count,(new_indexx,new_indexy), 1)
    H, xedges, yedges = np.histogram2d(mastercat_noshear_snr, mastercat_noshear_Tr, bins=[np.logspace(log10(snmin),log10(snmax),steps+1), np.logspace(log10(sizemin),log10(sizemax),steps+1)])
    new_response = mesh_response(d[msk], snmin, snmax, steps, sizemin, sizemax)
    new_meanes = mesh_average(np.sqrt((new_e1**2+new_e2**2)/2),new_indexx,new_indexy,steps,new_count)
    new_shearweight = (new_response/new_meanes)**2

    # count
    fig=plt.figure(figsize=(12,10))
    ax = plt.subplot(221)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_count/1.e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    plt.xlabel("galaxy size / PSF size")
    #im.colorbar(label="count")
    # im.axes.get_xaxis().set_visible(True)
    fig.colorbar(im, ax=ax, label=r"count [$10^5$]")

    # Eq 9. 
    ax = plt.subplot(222)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_meanes)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    plt.xlabel("galaxy size / PSF size")
    # im.axes.get_xaxis().set_visible(True)
    fig.colorbar(im, ax=ax, label=r"$\sqrt{\langle e_{1,2}^2\rangle}$")

    ax = plt.subplot(223)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_response, vmin=0.1, vmax=1.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    plt.xlabel("galaxy size / PSF size")
    # im.axes.get_xaxis().set_visible(True)
    fig.colorbar(im, ax=ax, label=r"$\langle$R$\rangle$")

    ax = plt.subplot(224)
    X, Y = np.meshgrid(yedges, xedges)
    im = ax.pcolormesh(X, Y, new_shearweight, vmin=0, vmax=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("S/N")
    plt.xlabel("galaxy size / PSF size")
    # im.axes.get_xaxis().get_ticklabels()[3].set_visible(False)
    fig.colorbar(im, ax=ax, label="shear weight")

    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.minorticks_off() 
    ax.set_xticks(np.array([1.2,1.3,1.4,1.5,1.6,1.7,2.,3.]))
    ax.set_xticklabels(np.array([r'1.2 x $10^{0}$','','','','','',r'2 x $10^{0}$',r'3 x $10^{0}$']))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    plt.savefig('count_response_ellip_SNR_Tr_v6_cutsv1.pdf', bbox_inches='tight')


# Figure 11; tangential and cross-component shear around bright and faint stars. 
def shear_stellar_contamination(mdet_cat, piff_all_cat):

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
def tangential_shear_field_center():

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
    inverse_variance_weight(20, mdet_cat, more_cuts=None)
    # shear_stellar_contamination(mdet_cat, piff_all_cat)
    # tangential_shear_field_center() -> modify the input and output. 

if __name__ == "__main__":
    main(sys.argv)