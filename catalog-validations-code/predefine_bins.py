
import fitsio as fio
import numpy as np
from esutil import stat
import pickle

def _predefine_bins(d, binname, fname, qa_lower_cuts, qa_lower_cuts_value, qa_upper_cuts, qa_upper_cuts_value, nperbin):
    
    # apply lower cuts
    for i in range(len(qa_lower_cuts)):
        msk = (d[qa_lower_cuts[i]] > qa_lower_cuts_value[i])
        d = d[msk]
    # apply upper cuts
    for i in range(len(qa_upper_cuts)):
        msk = (d[qa_upper_cuts[i]] < qa_upper_cuts_value[i])
        d = d[msk]

    d_bin = d[binname]
    hist = stat.histogram(d_bin, nperbin=nperbin, more=True)
    bin_num = len(hist['hist'])

    print(bin_num, hist['low'], hist['high'])
    with open(fname, 'wb') as f:
        pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)

    