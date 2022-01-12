import json
import numpy as np
import os
from matplotlib import pyplot as plt

def measure_rho(data, max_sep, max_mag, tag=None, use_xy=False, prefix='piff',
                alt_tt=False, opt=None, subtract_mean=False, do_rho0=False):
    """Compute the rho statistics
    """
    import treecorr

    e1 = data['obs_e1']
    e2 = data['obs_e2']
    T = data['obs_T']
    p_e1 = data[prefix+'_e1']
    p_e2 = data[prefix+'_e2']
    p_T = data[prefix+'_T']
    m = data['mag']

    if max_mag > 0:
        e1 = e1[m<max_mag]
        e2 = e2[m<max_mag]
        T = T[m<max_mag]
        p_e1 = p_e1[m<max_mag]
        p_e2 = p_e2[m<max_mag]
        p_T = p_T[m<max_mag]

    q1 = e1-p_e1
    q2 = e2-p_e2
    dt = (T-p_T)/T
    w1 = e1 * dt
    w2 = e2 * dt
    print('mean e = ',np.mean(e1),np.mean(e2))
    print('std e = ',np.std(e1),np.std(e2))
    print('mean T = ',np.mean(T))
    print('std T = ',np.std(T))
    print('mean de = ',np.mean(q1),np.mean(q2))
    print('std de = ',np.std(q1),np.std(q2))
    print('mean dT = ',np.mean(T-p_T))
    print('std dT = ',np.std(T-p_T))
    print('mean dT/T = ',np.mean(dt))
    print('std dT/T = ',np.std(dt))
    if subtract_mean:
        e1 -= np.mean(e1)
        e2 -= np.mean(e2)
        q1 -= np.mean(q1)
        q2 -= np.mean(q2)
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
        dt -= np.mean(dt)

    if use_xy:
        x = data['fov_x']
        y = data['fov_y']
        if max_mag > 0:
            x = x[m<max_mag]
            y = y[m<max_mag]
        print('x = ',x)
        print('y = ',y)

        ecat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=e1, g2=e2)
        qcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=q1, g2=q2)
        wcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=w1, g2=w2, k=dt)
    else:
        ra = data['ra']
        dec = data['dec']
        if max_mag > 0:
            ra = ra[m<max_mag]
            dec = dec[m<max_mag]
        print('ra = ',ra)
        print('dec = ',dec)

        ecat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2)
        qcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=q1, g2=q2)
        wcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=w1, g2=w2, k=dt)

    ecat.name = 'ecat'
    qcat.name = 'qcat'
    wcat.name = 'wcat'
    if tag is not None:
        for cat in [ ecat, qcat, wcat ]:
            cat.name = tag + ":"  + cat.name

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = max_sep,
        bin_size = 0.2,
    )

    if opt == 'lucas':
        bin_config['min_sep'] = 2.5
        bin_config['max_sep'] = 250.
        bin_config['nbins'] = 20
        del bin_config['bin_size']

    if opt == 'fine_bin':
        bin_config['min_sep'] = 0.1
        bin_config['max_sep'] = 2000.
        bin_config['bin_size'] = 0.01


    pairs = [ (qcat, qcat),
              (ecat, qcat),
              (wcat, wcat),
              (qcat, wcat),
              (ecat, wcat) ]
    if do_rho0:
        pairs.append( (ecat, ecat) )
    results = []
    for (cat1, cat2) in pairs:
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        rho = treecorr.GGCorrelation(bin_config, verbose=2)

        if cat1 is cat2:
            rho.process(cat1)
        else:
            rho.process(cat1, cat2)
        print('mean xi+ = ',rho.xip.mean())
        print('mean xi- = ',rho.xim.mean())
        results.append(rho)

    if alt_tt:
        print('Doing alt correlation of %s vs %s'%(dtcat.name, dtcat.name))

        rho = treecorr.KKCorrelation(bin_config, verbose=2)
        rho.process(dtcat)
        results.append(rho)

    return results


def write_stats(stat_file, rho1, rho2, rho3, rho4, rho5, rho0=None, tao0=None, tao2=None, tao5=None, corr_tt=None):

    stats = [
        rho1.meanlogr.tolist(),
        rho1.xip.tolist(),
        rho1.xip_im.tolist(),
        rho1.xim.tolist(),
        rho1.xim_im.tolist(),
        rho1.varxip.tolist(),
        rho1.varxim.tolist(),
        rho2.xip.tolist(),
        rho2.xip_im.tolist(),
        rho2.xim.tolist(),
        rho2.xim_im.tolist(),
        rho2.varxip.tolist(),
        rho2.varxim.tolist(),
        rho3.xip.tolist(),
        rho3.xip_im.tolist(),
        rho3.xim.tolist(),
        rho3.xim_im.tolist(),
        rho3.varxip.tolist(),
        rho3.varxim.tolist(),
        rho4.xip.tolist(),
        rho4.xip_im.tolist(),
        rho4.xim.tolist(),
        rho4.xim_im.tolist(),
        rho4.varxip.tolist(),
        rho4.varxim.tolist(),
        rho5.xip.tolist(),
        rho5.xip_im.tolist(),
        rho5.xim.tolist(),
        rho5.xim_im.tolist(),
        rho5.varxip.tolist(),
        rho5.varxim.tolist(),
    ]
    if rho0 is not None:
        stats.extend([
            rho0.xip.tolist(),
            rho0.xip_im.tolist(),
            rho0.xim.tolist(),
            rho0.xim_im.tolist(),
            rho0.varxip.tolist(),
            rho0.varxim.tolist(),
        ])
    if tao0 is not None:
        stats.extend([
            tao0.xip.tolist(),
            tao0.xip_im.tolist(),
            tao0.xim.tolist(),
            tao0.xim_im.tolist(),
            tao0.varxip.tolist(),
            tao0.varxim.tolist(),
        ])
    if tao2 is not None:
        stats.extend([
            tao2.xip.tolist(),
            tao2.xip_im.tolist(),
            tao2.xim.tolist(),
            tao2.xim_im.tolist(),
            tao2.varxip.tolist(),
            tao2.varxim.tolist(),
        ])
    if tao5 is not None:
        stats.extend([
            tao5.xip.tolist(),
            tao5.xip_im.tolist(),
            tao5.xim.tolist(),
            tao5.xim_im.tolist(),
            tao5.varxip.tolist(),
            tao5.varxim.tolist(),
        ])
    if corr_tt is not None:
        stats.extend([
            corr_tt.xi.tolist(),
            corr_tt.varxi.tolist()
        ])
    #print('stats = ',stats)
    print('stat_file = ',stat_file)
    with open(stat_file,'w') as fp:
        json.dump([stats], fp)
    print('Done writing ',stat_file)

def measure_tao(data1, data2, max_sep, max_mag, tag=None, use_xy=False, prefix='piff',
                alt_tt=False, opt=None, subtract_mean=False):
    """Compute the rho statistics
    """
    import treecorr

    e1 = data1['obs_e1']
    e2 = data1['obs_e2']
    T = data1['obs_T']
    p_e1 = data1[prefix+'_e1']
    p_e2 = data1[prefix+'_e2']
    p_T = data1[prefix+'_T']
    m = data1['mag']

    g_e1 = data2['gal_e1']
    g_e2 = data2['gal_e2']
    g_T = data2['gal_T']

    if max_mag > 0:
        e1 = e1[m<max_mag]
        e2 = e2[m<max_mag]
        T = T[m<max_mag]
        p_e1 = p_e1[m<max_mag]
        p_e2 = p_e2[m<max_mag]
        p_T = p_T[m<max_mag]
        g_e1 = g_e1[m<max_mag]
        g_e2 = g_e2[m<max_mag]
        g_T = g_T[m<max_mag]

    q1 = e1-p_e1
    q2 = e2-p_e2
    dt = (T-p_T)/T
    w1 = e1 * dt
    w2 = e2 * dt
    print('mean e = ',np.mean(e1),np.mean(e2))
    print('std e = ',np.std(e1),np.std(e2))
    print('mean T = ',np.mean(T))
    print('std T = ',np.std(T))
    print('mean de = ',np.mean(q1),np.mean(q2))
    print('std de = ',np.std(q1),np.std(q2))
    print('mean dT = ',np.mean(T-p_T))
    print('std dT = ',np.std(T-p_T))
    print('mean dT/T = ',np.mean(dt))
    print('std dT/T = ',np.std(dt))
    if subtract_mean:
        e1 -= np.mean(e1)
        e2 -= np.mean(e2)
        q1 -= np.mean(q1)
        q2 -= np.mean(q2)
        w1 -= np.mean(w1)
        w2 -= np.mean(w2)
        dt -= np.mean(dt)

    if use_xy:
        x = data1['fov_x']
        y = data1['fov_y']
        if max_mag > 0:
            x = x[m<max_mag]
            y = y[m<max_mag]
        print('x = ',x)
        print('y = ',y)

        ecat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=e1, g2=e2)
        qcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=q1, g2=q2)
        wcat = treecorr.Catalog(x=x, y=y, x_units='arcsec', y_units='arcsec', g1=w1, g2=w2, k=dt)
    else:
        ra = data1['ra']
        dec = data1['dec']
        ra_gal = data2['ra']
        dec_gal = data2['dec']
        if max_mag > 0:
            ra = ra[m<max_mag]
            dec = dec[m<max_mag]
        print('ra = ',ra)
        print('dec = ',dec)

        ecat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=e1, g2=e2)
        qcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=q1, g2=q2)
        wcat = treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg', g1=w1, g2=w2, k=dt)
        gcat = treecorr.Catalog(ra=ra_gal, dec=dec_gal, ra_units='deg', dec_units='deg', g1=g_e1, g2=g_e2)

    ecat.name = 'ecat'
    qcat.name = 'qcat'
    wcat.name = 'wcat'
    gcat.name = 'gcat'
    if tag is not None:
        for cat in [ ecat, qcat, wcat, gcat ]:
            cat.name = tag + ":"  + cat.name

    bin_config = dict(
        sep_units = 'arcmin',
        bin_slop = 0.1,

        min_sep = 0.5,
        max_sep = max_sep,
        bin_size = 0.2,
    )

    if opt == 'lucas':
        bin_config['min_sep'] = 2.5
        bin_config['max_sep'] = 250.
        bin_config['nbins'] = 20
        del bin_config['bin_size']

    if opt == 'fine_bin':
        bin_config['min_sep'] = 0.1
        bin_config['max_sep'] = 2000.
        bin_config['bin_size'] = 0.01


    pairs = [ (gcat, ecat),
              (gcat, qcat),
              (gcat, wcat) ]
    
    results = []
    for (cat1, cat2) in pairs:
        print('Doing correlation of %s vs %s'%(cat1.name, cat2.name))

        rho = treecorr.GGCorrelation(bin_config, verbose=2)

        if cat1 is cat2:
            rho.process(cat1)
        else:
            rho.process(cat1, cat2)
        print('mean xi+ = ',rho.xip.mean())
        print('mean xi- = ',rho.xim.mean())
        results.append(rho)

    if alt_tt:
        print('Doing alt correlation of %s vs %s'%(dtcat.name, dtcat.name))

        rho = treecorr.KKCorrelation(bin_config, verbose=2)
        rho.process(dtcat)
        results.append(rho)

    return results

def write_stats_tao(stat_file, tao0, tao2, tao5):

    stats = [
        tao0.meanlogr.tolist(),
        tao0.xip.tolist(),
        tao0.xip_im.tolist(),
        tao0.xim.tolist(),
        tao0.xim_im.tolist(),
        tao0.varxip.tolist(),
        tao0.varxim.tolist(),
        tao2.xip.tolist(),
        tao2.xip_im.tolist(),
        tao2.xim.tolist(),
        tao2.xim_im.tolist(),
        tao2.varxip.tolist(),
        tao2.varxim.tolist(),
        tao5.xip.tolist(),
        tao5.xip_im.tolist(),
        tao5.xim.tolist(),
        tao5.xim_im.tolist(),
        tao5.varxip.tolist(),
        tao5.varxim.tolist(),
    ]
    
    #print('stats = ',stats)
    print('stat_file = ',stat_file)
    with open(stat_file,'w') as fp:
        json.dump([stats], fp)
    print('Done writing ',stat_file)

def pretty_rho1(meanr, rho, sig, sqrtn, rho3=None, sig3=None, rho4=None, sig4=None, gband=False):
    import matplotlib.patches as mp
    if False:
        # This is all handwavy arguments about what the requirements are.  
        # I'm taking Cathering's CFHTLS xi+ values of 1.e-4 at 1 arcmin, 2e-6 at 40 arcmin.
        # Then I'm saying our requirements on rho need to be about 0.16 times this for SV (S/N=6),
        # but more like 0.03 times this for Y5.
        t1 = 0.5
        t2 = 300.
        xa = numpy.log(1.0)
        xb = numpy.log(40.0)
        ya = numpy.log(1.e-4)
        yb = numpy.log(2.e-6)
        x1 = numpy.log(t1)
        x2 = numpy.log(t2)
        y1 = (yb * (x1-xa) + ya * (xb-x1)) / (xb-xa)
        y2 = (yb * (x2-xa) + ya * (xb-x2)) / (xb-xa)
        xi1 = numpy.exp(y1)
        xi2 = numpy.exp(y2)

        sv_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.16, xi2 * 0.16, 0.], 
                           color = '#FFFF82')
        y5_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.03, xi2 * 0.03, 0.], 
                           color = '#BAFFA4')
    #else:
    elif False:
        # Use Alexandre's xi file as a requirement.
        theta = [0.5, 1.49530470404, 2.91321328166, 5.67019971545, 11.0321639144,
                 21.4548924111, 41.6931936543, 80.8508152859, 156.285886576,
                 297.92139021, 300.]

        dxi = [1.4e-5, 3.82239654447e-06, 2.36185315415e-06, 1.26849547074e-06, 6.3282672138e-07,
               3.25623661098e-07, 1.747852053e-07, 8.75326181278e-08, 3.60247306537e-08,
               1.13521735321e-08, 1.125e-8]
        req = [ x / 5.9 for x in dxi ]
        sv_req = plt.fill( [theta[0]] + theta + [theta[-1]], [0.] + req + [0.],
                           color = '#FFFF82')
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho1_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if rho3 is not None:
        plt.plot(meanr*1.03, rho3, color='green')
        plt.plot(meanr*1.03, -rho3, color='green', ls=':')
        plt.errorbar(meanr[rho3>0]*1.03, rho3[rho3>0], yerr=sig3[rho3>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[rho3<0]*1.03, -rho3[rho3<0], yerr=sig3[rho3<0]/sqrtn, color='green', ls='', marker='s')
        rho3_line = plt.errorbar(-meanr, rho3, yerr=sig3, color='green', marker='s')
    if rho4 is not None:
        plt.plot(meanr*1.06, rho4, color='red')
        plt.plot(meanr*1.06, -rho4, color='red', ls=':')
        plt.errorbar(meanr[rho4>0]*1.06, rho4[rho4>0], yerr=sig4[rho4>0]/sqrtn, color='red', ls='', marker='^')
        plt.errorbar(meanr[rho4<0]*1.06, -rho4[rho4<0], yerr=sig4[rho4<0]/sqrtn, color='red', ls='', marker='^')
        rho4_line = plt.errorbar(-meanr, rho4, yerr=sig4, color='red', marker='^')
    #sv_req = mp.Patch(color='#FFFF82')
    if rho3 is not None and rho4 is not None:
        if gband:
            loc = 'upper right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 24
        plt.legend([rho1_line, rho3_line, rho4_line],
                   [r'$\rho_1(\theta)$', r'$\rho_3(\theta)$', r'$\rho_4(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-9, 5.e-6] )
        #plt.ylim( [1.e-9, 2.e-5] )
        plt.ylim( [1.e-10, 1.e-5] )
    elif True:
        plt.legend([rho1_line, sv_req],
                   [r'$\rho_1(\theta)$', r'Requirement'],
                   loc='upper right')
        plt.ylim( [1.e-9, 5.e-6] )
    else: # For talk
        plt.legend([rho1_line, sv_req],
                   [r'$\rho_1(\theta)$',
                    r'Requirements for $d\sigma_8/\sigma_8 < 0.03$'],
                   loc='upper right')
        plt.ylim( [1.e-9, 3.e-6] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\rho(\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def pretty_rho2(meanr, rho, sig, sqrtn, rho5=None, sig5=None, gband=False):
    import matplotlib.patches as mp
    # The requirements on rho2 are less stringent.  They are larger by a factor 1/alpha.
    # Let's use alpha = 0.03.
    alpha = 0.03
    if False:
        t1 = 0.5
        t2 = 300.
        xa = numpy.log(1.0)
        xb = numpy.log(40.0)
        ya = numpy.log(1.e-4)
        yb = numpy.log(2.e-6)
        x1 = numpy.log(t1)
        x2 = numpy.log(t2)
        y1 = (yb * (x1-xa) + ya * (xb-x1)) / (xb-xa)
        y2 = (yb * (x2-xa) + ya * (xb-x2)) / (xb-xa)
        xi1 = numpy.exp(y1)
        xi2 = numpy.exp(y2)

        sv_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.16 / alpha, xi2 * 0.16 / alpha, 0.], 
                           color = '#FFFF82')
        y5_req = plt.fill( [t1, t1, t2, t2], [0., xi1 * 0.03 / alpha, xi2 * 0.03 / alpha, 0.], 
                           color = '#BAFFA4')

    #else:
    elif False:
        # Use Alexandre's xi file as a requirement.
        theta = [0.5, 1.49530470404, 2.91321328166, 5.67019971545, 11.0321639144,
                 21.4548924111, 41.6931936543, 80.8508152859, 156.285886576,
                 297.92139021, 300.]

        dxi = [1.4e-5, 3.82239654447e-06, 2.36185315415e-06, 1.26849547074e-06, 6.3282672138e-07,
               3.25623661098e-07, 1.747852053e-07, 8.75326181278e-08, 3.60247306537e-08,
               1.13521735321e-08, 1.125e-8]
        req = [ x / (2.4 * alpha) for x in dxi ]
        sv_req = plt.fill( [theta[0]] + theta + [theta[-1]], [0.] + req + [0.],
                           color = '#FFFF82')
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho2_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if rho5 is not None:
        plt.plot(meanr*1.03, rho5, color='green')
        plt.plot(meanr*1.03, -rho5, color='green', ls=':')
        plt.errorbar(meanr[rho5>0]*1.03, rho5[rho5>0], yerr=sig5[rho5>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[rho5<0]*1.03, -rho5[rho5<0], yerr=sig5[rho5<0]/sqrtn, color='green', ls='', marker='s')
        rho5_line = plt.errorbar(-meanr, rho5, yerr=sig5, color='green', marker='s')
    #sv_req = mp.Patch(color='#FFFF82')
    if rho5 is not None:
        if gband:
            loc = 'lower right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 24
        plt.legend([rho2_line, rho5_line],
                   [r'$\rho_2(\theta)$', r'$\rho_5(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-7, 5.e-4] )
        plt.ylim( [1.e-8, 1.e-5] )
    elif True: # For paper
        plt.legend([rho2_line, sv_req],
                   [r'$\rho_2(\theta)$', r'Requirement'],
                   loc='upper right')
        plt.ylim( [1.e-7, 5.e-4] )
    else:
        plt.legend([rho2_line, sv_req],
                   [r'$\rho_2(\theta)$',
                    r'Requirements for $d\sigma_8/\sigma_8 < 0.03$'],
                   loc='upper right')
        plt.ylim( [1.e-7, 3.e-4] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\rho(\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()


def pretty_rho0(meanr, rho, sig, sqrtn):
    import matplotlib.patches as mp
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    rho0_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if True:
        plt.legend([rho0_line],
                   [r'$\rho_0(\theta)$'],
                   loc='upper right', fontsize=24)
        plt.ylim( [1.e-6, 1.e-3] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\rho(\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def pretty_tao(meanr, rho, sig, sqrtn, tao2=None, sig2=None, tao5=None, sig5=None, gband=False):
    import matplotlib.patches as mp
    plt.plot(meanr, rho, color='blue')
    plt.plot(meanr, -rho, color='blue', ls=':')
    plt.errorbar(meanr[rho>0], rho[rho>0], yerr=sig[rho>0]/sqrtn, color='blue', ls='', marker='o')
    plt.errorbar(meanr[rho<0], -rho[rho<0], yerr=sig[rho<0]/sqrtn, color='blue', ls='', marker='o')
    tao0_line = plt.errorbar(-meanr, rho, yerr=sig, color='blue', marker='o')
    if tao2 is not None:
        plt.plot(meanr*1.03, tao2, color='green')
        plt.plot(meanr*1.03, -tao2, color='green', ls=':')
        plt.errorbar(meanr[tao2>0]*1.03, tao2[tao2>0], yerr=sig2[tao2>0]/sqrtn, color='green', ls='', marker='s')
        plt.errorbar(meanr[tao2<0]*1.03, -tao2[tao2<0], yerr=sig2[tao2<0]/sqrtn, color='green', ls='', marker='s')
        tao2_line = plt.errorbar(-meanr, tao2, yerr=sig2, color='green', marker='s')
    if tao5 is not None:
        plt.plot(meanr*1.06, tao5, color='red')
        plt.plot(meanr*1.06, -tao5, color='red', ls=':')
        plt.errorbar(meanr[tao5>0]*1.06, tao5[tao5>0], yerr=sig5[tao5>0]/sqrtn, color='red', ls='', marker='^')
        plt.errorbar(meanr[tao5<0]*1.06, -tao5[tao5<0], yerr=sig5[tao5<0]/sqrtn, color='red', ls='', marker='^')
        tao5_line = plt.errorbar(-meanr, tao5, yerr=sig5, color='red', marker='^')
    
    if tao2 is not None and tao5 is not None:
        if gband:
            loc = 'upper right'
            fontsize = 18
        else:
            loc = 'upper right'
            fontsize = 24
        plt.legend([tao0_line, tao2_line, tao5_line],
                   [r'$\tau_0(\theta)$', r'$\tau_2(\theta)$', r'$\tau_5(\theta)$'],
                   loc=loc, fontsize=fontsize)
        #plt.ylim( [1.e-9, 5.e-6] )
        #plt.ylim( [1.e-9, 2.e-5] )
        plt.ylim( [1.e-10, 1.e-4] )
    elif True:
        plt.legend([tao0_line],
                   [r'$\tau_0(\theta)$'],
                   loc='upper right', fontsize=24)
        plt.ylim( [1.e-6, 1.e-3] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlim( [0.5,300.] )
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\tau (\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()


def plot_corr_tt(meanr, corr, sig):
    plt.plot(meanr, corr, color='blue')
    plt.plot(meanr, -corr, color='blue', ls=':')
    plt.errorbar(meanr[corr>0], corr[corr>0], yerr=sig[corr>0], color='blue', ls='', marker='o')
    plt.errorbar(meanr[corr<0], -corr[corr<0], yerr=sig[corr<0], color='blue', ls='', marker='o')
    plt.ylim( [1.e-7, 5.e-5] )
    plt.xlim( [0.5,300.] )
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.xlabel(r'$\theta$ (arcmin)', fontsize=24)
    plt.ylabel(r'$\langle (dT_p/T_p) (dT_p/T_p) \rangle (\theta)$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log', nonposy='clip')
    plt.tight_layout()

def plot_overall_rho(work, name):

    base_keys = ['r', 'i', 'z']
    keys = [ name+'_' + k for k in base_keys ]

    for key in keys:
        stat_file = os.path.join(work, "rho_" + key + ".json")
        if not os.path.isfile(stat_file):
            print('File not found: ',stat_file,' (skipping)')
            continue

        # Read the json file 
        print('Read %s'%stat_file)
        with open(stat_file,'r') as f:
            stats = json.load(f)

        #print' stats = ',stats
        if len(stats) == 1:  # I used to save a list of length 1 that in turn was a list
            stats = stats[0]

        print('len(stats) = ',len(stats))
        ( meanlogr,
          rho1p,
          rho1p_im,
          rho1m,
          rho1m_im,
          var1p,
          var1m,
          rho2p,
          rho2p_im,
          rho2m,
          rho2m_im,
          var2p,
          var2m,
          rho3p,
          rho3p_im,
          rho3m,
          rho3m_im,
          var3p,
          var3m,
          rho4p,
          rho4p_im,
          rho4m,
          rho4m_im,
          var4p,
          var4m,
          rho5p,
          rho5p_im,
          rho5m,
          rho5m_im,
          var5p,
          var5m,
        ) = stats[:31]

        meanr = np.exp(meanlogr)
        rho1p = np.array(rho1p)
        rho1m = np.array(rho1m)
        rho2p = np.array(rho2p)
        rho2m = np.array(rho2m)
        rho3p = np.array(rho3p)
        rho3m = np.array(rho3m)
        rho4p = np.array(rho4p)
        rho4m = np.array(rho4m)
        rho5p = np.array(rho5p)
        rho5m = np.array(rho5m)
        sig_rho1 = np.sqrt(var1p)
        sig_rho2 = np.sqrt(var2p)
        sig_rho3 = np.sqrt(var3p)
        sig_rho4 = np.sqrt(var4p)
        sig_rho5 = np.sqrt(var5p)
        sqrtn = 1

        cols = [meanr,
                rho1p, rho1m, sig_rho1,
                rho2p, rho2m, sig_rho2,
                rho3p, rho3m, sig_rho3,
                rho4p, rho4m, sig_rho4,
                rho5p, rho5m, sig_rho5]
        header = ('meanr  '+
                  'rho1  rho1_xim  sig_rho1  '+
                  'rho2  rho2_xim  sig_rho2  '+
                  'rho3  rho3_xim  sig_rho3  '+
                  'rho4  rho4_xim  sig_rho4  '+
                  'rho5  rho5_xim  sig_rho5  ')

        if len(stats) > 31:
            ( rho0p,
              rho0p_im,
              rho0m,
              rho0m_im,
              var0p,
              var0m,
            ) = stats[31:37]
            rho0p = np.array(rho0p)
            rho0m = np.array(rho0m)
            sig_rho0 = np.sqrt(var0p)
            cols += [rho0p, rho0m, sig_rho0]
            header += 'rho0  rho0_xim  sig_rho0  '
        if len(stats) > 37:
            ( tao0p,
              tao0p_im,
              tao0m,
              tao0m_im,
              vart0p,
              vart0m,
              tao2p,
              tao2p_im,
              tao2m,
              tao2m_im,
              vart2p,
              vart2m,
              tao5p,
              tao5p_im,
              tao5m,
              tao5m_im,
              vart5p,
              vart5m,
            ) = stats[37:55]
            tao0p = np.array(tao0p)
            tao0m = np.array(tao0m)
            tao2p = np.array(tao2p)
            tao2m = np.array(tao2m)
            tao5p = np.array(tao5p)
            tao5m = np.array(tao5m)
            sig_tao0 = np.sqrt(vart0p)
            sig_tao2 = np.sqrt(vart2p)
            sig_tao5 = np.sqrt(vart5p)
            cols += [tao0p, tao0m, sig_tao0, 
                     tao2p, tao2m, sig_tao2, 
                     tao5p, tao5m, sig_tao5,]
            header += ('tao0  tao0_xim  sig_tao0  '+
                       'tao2  tao2_xim  sig_tao2  '+
                       'tao5  tao5_xim  sig_tao5  ')

        outfile = 'rho_' + key + '.dat'
        np.savetxt(outfile, np.array(cols), fmt='%.6e', header=header)
        print('wrote',outfile)
 
        plt.clf()
        pretty_rho1(meanr, rho1p, sig_rho1, sqrtn, rho3p, sig_rho3, rho4p, sig_rho4,
                    gband=(key.endswith('g')))
        plt.savefig('rho1_' + key + '.pdf')

        plt.clf()
        pretty_rho2(meanr, rho2p, sig_rho2, sqrtn, rho5p, sig_rho5,
                    gband=(key.endswith('g')))
        plt.savefig('rho2_' + key + '.pdf')

        if len(stats) > 31:
            plt.clf()
            pretty_rho0(meanr, rho0p, sig_rho0, sqrtn)
            plt.savefig('rho0_' + key + '.pdf')
        
        if len(stats) > 37:
            plt.clf()
            pretty_tao(meanr, tao0p, sig_rho0, sqrtn, tao2p, sig_tao2, tao5p, sig_tao5)
            plt.savefig('tao_' + key + '.pdf')

        if len(stats) > 55:
            corr_tt, var_corr_tt = stats[37:39]
            corr_tt = np.array(corr_tt)
            sig_corr_tt = np.sqrt(var_corr_tt)

            plt.clf()
            plot_corr_tt(meanr, corr_tt, sig_corr_tt)
            plt.savefig('corrtt_' + key + '.pdf')

def plot_overall_tao(work, name):

    base_keys = ['r', 'i', 'z']
    keys = [ name+'_' + k for k in base_keys ]

    for key in keys:
        stat_file = os.path.join(work, "tao_" + key + ".json")
        if not os.path.isfile(stat_file):
            print('File not found: ',stat_file,' (skipping)')
            continue

        # Read the json file 
        print('Read %s'%stat_file)
        with open(stat_file,'r') as f:
            stats = json.load(f)

        #print' stats = ',stats
        if len(stats) == 1:  # I used to save a list of length 1 that in turn was a list
            stats = stats[0]

        print('len(stats) = ',len(stats))
        ( meanlogr,
          tao0p,
          tao0p_im,
          tao0m,
          tao0m_im,
          vart0p,
          vart0m,
          tao2p,
          tao2p_im,
          tao2m,
          tao2m_im,
          vart2p,
          vart2m,
          tao5p,
          tao5p_im,
          tao5m,
          tao5m_im,
          vart5p,
          vart5m,
        ) = stats[:19]
        meanr = np.exp(meanlogr)
        tao0p = np.array(tao0p)
        tao0m = np.array(tao0m)
        tao2p = np.array(tao2p)
        tao2m = np.array(tao2m)
        tao5p = np.array(tao5p)
        tao5m = np.array(tao5m)
        sig_tao0 = np.sqrt(vart0p)
        sig_tao2 = np.sqrt(vart2p)
        sig_tao5 = np.sqrt(vart5p)
        sqrtn = 1
        cols = [meanr, 
                tao0p, tao0m, sig_tao0, 
                tao2p, tao2m, sig_tao2, 
                tao5p, tao5m, sig_tao5]
        header = ('meanr  '+
                  'tao0  tao0_xim  sig_tao0  '+
                  'tao2  tao2_xim  sig_tao2  '+
                  'tao5  tao5_xim  sig_tao5  ')

        outfile = 'tao_' + key + '.dat'
        np.savetxt(outfile, np.array(cols), fmt='%.6e', header=header)
        print('wrote',outfile)
        
        plt.clf()
        pretty_tao(meanr, tao0p, sig_tao0, sqrtn, tao2p, sig_tao2, tao5p, sig_tao5)
        plt.savefig('tao_' + key + '.pdf')