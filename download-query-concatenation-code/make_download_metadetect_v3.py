import os
import sys
import fitsio as fio
import numpy as np

def main(argv):
    out_fname = '/data/des70.a/data/masaya/metadetect/v3/mdet_download.txt'
    f = open('/data/des70.a/data/masaya/metadetect/v3/fnames.txt', 'r')
    fs = f.read().split('\n')

    all_files = []
    for fname in fs:
        whole_fname = os.path.join('https://www.cosmo.bnl.gov/Private/gpfs/workarea/beckermr/des-y6-analysis/2021_12_18_test_v6_download/data', fname)
        all_files.append(whole_fname)

    np.savetxt(out_fname, np.array(all_files), fmt="%s")
    # os.system('wget -P /data/des70.a/data/masaya/metadetect/v3/ --user='+sys.argv[1]+' --password='+sys.argv[2]+' -i /data/des70.a/data/masaya/metadetect/mdet_download.txt')

if __name__ == "__main__":
    main(sys.argv)