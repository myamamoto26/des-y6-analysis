# Run step [galsim_montara, true_detection, meds, newish_metacal] on eastlake. 
Everything ran accordingly, but shear recovery looks bigger than we expect 4e-4. 
This calibration result is computed only from the input shear (g1,g2)=(0.02,0.00) catalog. 

m: 0.001563 +/- 0.000185

c: -0.000006 +/- 0.000003

With four sets of the simlations (g1,g2)={(0.02,0.00), (-0.02,0.00), (0.00,0.02), (0.00,-0.02)} to cancel shape noise, 
the calibration result improved much better. 

m: 0.000382 +/- 0.000113

c: -0.000008 +/- 0.000002
