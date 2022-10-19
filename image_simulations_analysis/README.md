# How to run analysis scripts

1. measure_shear_bias.py (this is for metadetection output)
2. measure_shear_cancel_patches.py (this is for newish_metacal output)
(3. measure_shear_nocancel.py; should probably not be used.)

## measure_shear_bias.py
Prep. work: Run, for example, [galsim_montara, true_detection, pizza-cutter, metadetection] steps, and store metadetection output somewhere. Collect the output from all the tiles that were run. 

Then run, ```python measure_shear_bias.py```. 


## measure_shear_cancel_patches.py
Prep. work: Run, for example, [galsim_montara, true_detection, meds, newish_metacal] steps, and store newish_metacal output somewhere as {tilename}_newish_metacal_g10.02_g20.00.fits for (g1, g2)=(0.02, 0.00) measurement. Do this for (g1, g2)=(-0.02, 0.00) as well (i.e., {tilename}_newish_metacal_g1-0.02_g20.00.fits). 

Then run, ```python measure_shear_cancel_patches.py {tilename}_newish_metacal_g10.02_g20.00.fits```. 
You may need to do the following depending on your use-case. 
1. turn on/off ```swap``` parameter in the code to change g1+ to g2+ analysis. 
2. change the true positions catalog file in the code. 