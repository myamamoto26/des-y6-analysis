
import yaml 

f_im = open('DES0003-3832_r_fcut-flist-y3v02.dat', 'r')
f_mag = open('DES0003-3832_r_fcut-flist-y3v02-magzp.dat', 'r')
source_im = f_im.read().split('\n')[:-1]
source_mag = f_mag.read().split('\n')[:-1]

tilename = 'DES0003-3832'
coadd_info = {'band': 'r', 
              'bmask_ext': 'MSK', 
              'bmask_path': '/global/cscratch1/sd/myamamot/y3v02/DES0003-3832/sources-r/OPS/multiepoch/Y3A1/r2590/DES0003-3832/p01/coadd/DES0003-3832_r2590p01_r.fits.fz', 
              'cat_path' : None, 
              'compression': '.fz',
              'crossra0': 'N', 
              'filename': 'DES0003-3832_r2590p01_r.fits',
              'gaia_stars_file': None, 
              'image_ext': 'SCI', 
              'image_flags': 0, 
              'image_path': '/global/cscratch1/sd/myamamot/y3v02/DES0003-3832/sources-r/OPS/multiepoch/Y3A1/r2590/DES0003-3832/p01/coadd/DES0003-3832_r2590p01_r.fits.fz', 
              'image_shape': [10000, 10000], 
              'magzp': 30.0, 
              'path': 'OPS/multiepoch/Y3A1/r2590/DES0003-3832/coadd', 
              'pfw_attempt_id': None, 
              'position_offset': 1, 
              'psf_path': None, 
              'scale': 1.0, 
              'seg_ext': 'SCI', 
              'seg_path': None, 
              'src_info': []}

for i, source in enumerate(source_im):
    
    coadd_info['src_info'].append({'band': 'r', 
                                 'bkg_ext': 'SCI', 
                                 'bkg_path': source[:-3], 
                                 'bmask_ext': 'MSK', 
                                 'bmask_path': source[:-3], 
                                 'ccdnum': int(source[-28:-26]), 
                                 'compression': '.fz', 
                                 'expnum': int(source[-40:-32]), 
                                 'filename' : source[-41:-3], 
                                 'head_path' : None,
                                 'image_ext': 'SCI',
                                 'image_flags': 0, 
                                 'image_path': source[:-3],
                                 'image_shape': [4096, 2048], 
                                 'magzp': source_mag[i], 
                                 'path': source[-101:-42], 
                                 'pfw_attempt_id': None, 
                                 'piff_info': {'ccdnum': 5, 
                                                'desdm_flags': 0, 
                                                'exp_star_t_mean': 0.3464619517326355, 
                                                'exp_star_t_std': 0.025283748283982277, 
                                                'expnum': 258892, 
                                                'fwhm_cen': 0.9715940952301025, 
                                                'nstar': 77, 
                                                'star_t_mean': 0.3434189558029175, 
                                                'star_t_std': 0.010958988219499588}, 
                                 'piff_path': None, 
                                 'position_offset': 1, 
                                 'psf_path': None, 
                                 'psfex_path': None,
                                 'scale': 0.207655721949912, 
                                 'seg_path': None, 
                                 'tilename': tilename,
                                 'weight_ext': 'WGT', 
                                 'weight_path': source[:-3]
                                })

with open('test_pizzacutter.yaml', 'w') as f:
    yaml.dump(coadd_info, f)