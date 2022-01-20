
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt
from esutil import stat
import pickle

# This catalog only contains 500 tiles. 
d = fio.read('/data/des70.a/data/masaya/metadetect/v2/mdet_test_all_v2.fits')

# histogram
def _get_hist_with_upper_cuts(d, qa_upper_cuts, qa_upper_cuts_value):

    plt.clf()
    fig,ax=plt.subplots(1,3,figsize=(19,7))
    label = 'QA Cuts: '
    total = len(d)
    for i in range(len(qa_upper_cuts)):
        msk = (d[qa_upper_cuts[i]] < qa_upper_cuts_value[i])
        d = d[msk]
        label += qa_upper_cuts[i]+'<'+str(qa_upper_cuts_value[i])+' '
    label2 = 'Remaining: '+str("{:2.2f}".format(len(d)*100/total))+"%"
    
    ax[0].hist(d['MDET_S2N'], bins=100000, histtype='step')
    ax[0].set_xlabel('S/N', fontsize=20)
    ax[0].tick_params(labelsize=15)

    ax[1].hist(d['MDET_T'], bins=100000, histtype='step')
    ax[1].set_xlabel('T', fontsize=20)
    ax[1].tick_params(labelsize=15)

    ax[2].hist(d['MDET_T_RATIO'], bins=100000, histtype='step')
    ax[2].set_xlabel(r'$T_{ratio}$', fontsize=20)
    ax[2].tick_params(labelsize=15)
    
    print(label+label2)
    ax[1].set_title(label+label2, fontsize=15)
    plt.tight_layout()
    plt.savefig('mdet_qa_cuts.pdf', bbox_inches='tight')

def _predefine_bins(d, binname, fname, qa_upper_cuts, qa_upper_cuts_value, nperbin):
    
    # apply cuts
    for i in range(len(qa_upper_cuts)):
        msk = (d[qa_upper_cuts[i]] < qa_upper_cuts_value[i])
        d = d[msk]

    d_bin = d[binname]
    hist = stat.histogram(d_bin, nperbin=nperbin, more=True)
    bin_num = len(hist['hist'])

    print(hist['low'], hist['high'])
    with open(fname, 'wb') as f:
        pickle.dump(hist, fname, protocol=pickle.HIGHEST_PROTOCOL)

    