
import fitsio as fio
import numpy as np
from matplotlib import pyplot as plt

# This catalog only contains 500 tiles. 
d = fio.read('/data/des70.a/data/masaya/metadetect/v2/mdet_test_all_v2.fits')

# histogram
def _get_hist_with_cuts(d, qa_cut_quantity, qa_cut):
    
    fig,ax=plt.subplots(1,3,figsize=(15,7))
    label = 'QA Cuts: '
    total = len(d)
    for i in range(len(qa_cut_quantity)):
        msk = (d[qa_cut_quantity[i]] < qa_cut[i])
        d = d[msk]
        label += qa_cut_quantity[i]+'<'+str(qa_cut[i])+' '
    label2 = 'Remaining: '+str(total*100/len(d))
    
    ax[0].hist(d['MDET_S2N'], bins=100000, histtype='step')
    ax[0].set_xlabel('S/N', fontsize=20)
    ax[0].tick_params(labelsize=15)

    ax[1].hist(d['MDET_T'], bins=100000, histtype='step')
    ax[1].set_xlabel('T', fontsize=20)
    ax[1].tick_params(labelsize=15)

    ax[2].hist(d['MDET_T_RATIO'], bins=100000, histtype='step')
    ax[2].set_xlabel(r'$T_{ratio}$', fontsize=20)
    ax[2].tick_params(labelsize=15)
    
    ax[1].title(label, fontsize=15)
    plt.tight_layout()
    plt.savefig('mdet_qa_cuts.pdf', bbox_inches='tight')
