import os, sys
import numpy as np
import fitsio as fio
import galsim
from mpi4py import MPI
from past.utils import old_div
from joblib import Parallel, delayed
from tqdm import tqdm

import ngmix
from ngmix import priors, joint_prior
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
from ngmix.jacobian import Jacobian
from ngmix.observation import Observation, ObsList, MultiBandObsList
from ngmix.observation import get_mb_obs
from eastlake.steps.newish_metacal.metacal.metacal_fitter import MetacalFitter

CONFIG = {
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {'psf': 'fitgauss'},

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    },
}

def _run_sim(i_gal, rng, fail):

	if i_gal % 50 == 0:
		print('object number, ', i_gal)

	# galaxy and PSF profile.
	gal_model = galsim.Exponential(half_light_radius=0.5, flux=100.) # i'm not sure how to apply flux that makes s2n > 1000. 
	gal_model = gal_model.shear(g1=0.02, g2=0.00)
	psf_model = galsim.Gaussian(fwhm=0.9, flux=1.)
	gal_model = galsim.Convolve(gal_model, psf_model)

	# stamps. 
	b = galsim.BoundsI( xmin=1,
                        xmax=32,
                        ymin=1,
                        ymax=32)
	gal_stamp = galsim.Image(b, scale=0.263)
	psf_stamp = galsim.Image(b, scale=0.263)
	rand_off = galsim.UniformDeviate(314)
	dx = rand_off() - 0.5
	dy = rand_off() - 0.5
	gal_model.drawImage(image=gal_stamp, offset=np.array((dx,dy)))
	psf_model.drawImage(image=psf_stamp)

	# pixel noise. and apply galaxy flux so that s2n > 1000. 
	# add noise manually. 
	im = gal_stamp.array
	im_psf = psf_stamp.array

	pixel_noise = np.sqrt(np.sum(im**2))/1000
	im_noise = np.random.normal(scale=pixel_noise, size=im.shape)
	im += im_noise

	# make observation object.
	obs_list = ObsList()
	psf_list = ObsList()

	# jacobian. 
	jacob = gal_stamp.wcs.jacobian()
	gal_jacob = Jacobian(
						row=gal_stamp.true_center.y+dy,
						col=gal_stamp.true_center.x+dx,
						dvdrow=jacob.dvdy,
						dvdcol=jacob.dvdx,
						dudrow=jacob.dudy,
						dudcol=jacob.dudx)
	psf_jacob = Jacobian(
						row=psf_stamp.true_center.y,
						col=psf_stamp.true_center.x,
						dvdrow=jacob.dvdy,
						dvdcol=jacob.dvdx,
						dudrow=jacob.dudy,
						dudcol=jacob.dudx)

	weight = np.ones_like(im)/pixel_noise**2
	mask = np.where(weight!=0)
	w = [np.mean(weight[mask])]
	noise = old_div(np.ones_like(weight),w[-1])

	bmask = np.zeros_like(im, dtype=np.int32)
	psf_obs = Observation(im_psf, jacobian=psf_jacob, meta={'offset_pixels':None,'file_id':None})
	obs = Observation(im, weight=weight, bmask=bmask, jacobian=gal_jacob, psf=psf_obs, meta={'offset_pixels':None,'file_id':None,'orig_row':None,'orig_col':None})
	obs.set_noise(noise)
	obs_list.append(obs)

	# mbobs = get_mb_obs(obs_list)
	# mcal = MetacalFitter(CONFIG, 1, rng)
	# try:
	# 	mcal.go([mbobs])
	# 	res = mcal.result
	# 	if res == None:
	# 		fail+=1
	# 		print('cannot fit this object')
	# 		return None
	# except (BootGalFailure, BootPSFFailure): 
	# 	fail += 1
	# 	return None
	T=0.2
	metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'], 'psf': 'gauss'}
	pix_range = old_div(0.263,10.)
	e_range = 0.1
	fdev = 1.
	def pixe_guess(n):
	    return 2.*n*np.random.random() - n

	cp = ngmix.priors.CenPrior(0.0, 0.0, 0.263, 0.263)
	gp = ngmix.priors.GPriorBA(0.3)
	hlrp = ngmix.priors.FlatPrior(1.0e-4, 1.0e2)
	fracdevp = ngmix.priors.Normal(0.5, 0.1, bounds=[0., 1.])
	fluxp = ngmix.priors.FlatPrior(0, 1.0e5)

	prior = joint_prior.PriorSimpleSep(cp, gp, hlrp, fluxp)
	guess = np.array([pixe_guess(pix_range),pixe_guess(pix_range),pixe_guess(e_range),pixe_guess(e_range),T,500.])

	boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)
	psf_model = "gauss"
	gal_model = "gauss"

	lm_pars={'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
	max_pars={'method': 'lm', 'lm_pars':lm_pars}

	Tguess=T**2/(2*np.log(2))
	ntry=2
	boot.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry, metacal_pars=metacal_pars) 
	res_ = boot.get_metacal_result()
	print(res_)
	exit()
	#return res

def main(argv):

	gal_num = 1
	rng = np.random.RandomState(314)
	fail = 0 
	jobs = [
			delayed(_run_sim)(i_gal, rng, fail)
    		for i_gal in tqdm(range(gal_num))]
	res_full = Parallel(n_jobs=-1, verbose=0, backend='loky')(jobs)
	res_full = list(filter(None, res_full))
	
	print(len(res_full))
	print('number of failed fit', fail)
	final_cat = np.concatenate(res_full)
	#fio.write('eastlake-test-mcal.fits', final_cat)

if __name__ == "__main__":

	#comm = MPI.COMM_WORLD
	#rank = comm.Get_rank()
	#size = comm.Get_size()

	main(sys.argv)
