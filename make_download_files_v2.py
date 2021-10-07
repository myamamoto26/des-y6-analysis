import os
import sys
import fitsio as fio
import numpy as np

def get_coaddtile_geom(section, query, out_fname):
    ## Query Gaia stars and PSF model from DESDM database. 
    os.system('export LD_LIBRARY_PATH=/home/s1/masaya/oml4rclient_install_dir/instantclient_21_1')
    import easyaccess as ea
    conn = ea.connect(section=section)
    curs = conn.cursor()
    conn.query_and_save(query, out_fname)

def main(argv):
    query = """
            select
                concat(fai.filename, fai.compression) as filename,
                fai.path as path,
                m.band as band
            from
                desfile d1,
                proctag t,
                miscfile m,
                file_archive_info fai
            where
                d1.pfw_attempt_id = t.pfw_attempt_id
                and t.tag = 'Y6A2_PIZZACUTTER_TEST_V6'
                and d1.filename = m.filename
                and d1.id = fai.desfile_id
                and fai.archive_name = 'desar2home'
                and d1.filetype = 'coadd_pizza_cutter';
            """
    out_fname = '/data/des70.a/data/masaya/pizza-slice/v2/pizza_slices_coadd_v2.fits'
    if not os.path.exists(out_fname):
        get_coaddtile_geom('desoper', query, out_fname)
    pizza_filepaths = fio.read('/data/des70.a/data/masaya/pizza-slice/v2/pizza_slices_coadd_v2.fits')

    r_files = pizza_filepaths[pizza_filepaths['BAND'] == 'r']
    i_files = pizza_filepaths[pizza_filepaths['BAND'] == 'i']
    z_files = pizza_filepaths[pizza_filepaths['BAND'] == 'z']
    all_r_files = []
    all_i_files = []
    all_z_files = []
    for f in range(len(all_r_files)):
        fname_r_pizza = os.path.join('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/', r_files['PATH'], r_files['FILENAME'])
        all_r_files.append(fname_r_pizza)
    for f in range(len(all_i_files)):
        fname_i_pizza = os.path.join('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/', i_files['PATH'], i_files['FILENAME'])
        all_i_files.append(fname_i_pizza)
    for f in range(len(all_z_files)):
        fname_z_pizza = os.path.join('https://desar2.cosmology.illinois.edu/DESFiles/desarchive/', z_files['PATH'], z_files['FILENAME'])
        all_z_files.append(fname_z_pizza)

    np.savetxt('/data/des70.a/data/masaya/pizza-slice/v2/pizza_r_slice_download.txt', np.array(all_r_files), fmt="%s")
    np.savetxt('/data/des70.a/data/masaya/pizza-slice/v2/pizza_i_slice_download.txt', np.array(all_i_files), fmt="%s")
    np.savetxt('/data/des70.a/data/masaya/pizza-slice/v2/pizza_z_slice_download.txt', np.array(all_z_files), fmt="%s")

    os.system('wget --user='+sys.argv[1]+' --password='+sys.argv[2]+' -i /data/des70.a/data/masaya/pizza-slice/v2/pizza_r_slice_download.txt')
    os.system('wget --user='+sys.argv[1]+' --password='+sys.argv[2]+' -i /data/des70.a/data/masaya/pizza-slice/v2/pizza_i_slice_download.txt')
    os.system('wget --user='+sys.argv[1]+' --password='+sys.argv[2]+' -i /data/des70.a/data/masaya/pizza-slice/v2/pizza_z_slice_download.txt')


if __name__ == "__main__":
    main(sys.argv)