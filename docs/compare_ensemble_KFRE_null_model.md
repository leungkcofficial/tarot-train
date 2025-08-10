Now, using the ensembled CIF, we need to compare the perfomance of 
The ensembled CIF arrays are stored in the following locations:

HDF5 Files:
Temporal ensemble CIF: results/full_ensemble/temporal_ensemble_cif.h5
Spatial ensemble CIF: results/full_ensemble/spatial_ensemble_cif.h5
Each HDF5 file contains:

Dataset name: ensemble_cif
Shape: (2, 5, n_samples)
2 events (death and transplant)
5 time points (365, 730, 1095, 1460, 1825 days)
n_samples (42,953 for temporal, 155,042 for spatial)
Attributes:
n_models: 24 (number of models averaged)
method: "simple_average"
created: ISO timestamp of creation
How they were created:
The ensemble CIFs were created by:

Loading the stacked predictions from:
results/final_deploy/stacked_predictions/temporal_stacked_cif.h5 (shape: 24, 2, 5, n_samples)
results/final_deploy/stacked_predictions/spatial_stacked_cif.h5 (shape: 24, 2, 5, n_samples)
Averaging across all 24 model groups using np.mean(stacked, axis=0)
Saving the averaged results to the HDF5 files mentioned above
You can load these ensemble CIF arrays using:

import h5py

# Load temporal ensemble CIF
with h5py.File('results/full_ensemble/temporal_ensemble_cif.h5', 'r') as f:
    temporal_ensemble_cif = f['ensemble_cif'][:]
    
# Load spatial ensemble CIF  
with h5py.File('results/full_ensemble/spatial_ensemble_cif.h5', 'r') as f:
    spatial_ensemble_cif = f['ensemble_cif'][:]

The ensembled CIF arrays are stored in HDF5 format at:

Temporal ensemble CIF: results/full_ensemble/temporal_ensemble_cif.h5
Spatial ensemble CIF: results/full_ensemble/spatial_ensemble_cif.h5
Each file contains:

Dataset name: ensemble_cif
Shape: (2, 5, n_samples) where:
2 = number of events (death and transplant)
5 = time points (365, 730, 1095, 1460, 1825 days)
n_samples = 42,953 for temporal, 155,042 for spatial
Metadata attributes: n_models=24, method="simple_average", created=timestamp
These were created by averaging the 24 stacked model predictions from results/final_deploy/stacked_predictions/ directory.

# You can read /mnt/dump/yard/projects/tarot2/steps/kfre_eval.py and all function and python file it imported to get how the KFRE is calculated

# Here is my plan
1. We already get the ground truth duration and event from spatial and temporal test set
2. Discretinize the duration to discrete labels of [365, 730, 1095, 1460, 1825]
3. use the discretinise label and lifeline aalen jonhansen fitter to fit a null model, and get the CIF <-- this serves as our observed risk
4. as KFRE only have 2 year and 5 year, our ensembled CIF array (array) can be sliced to (2, 2, n_samples) with the second dimention containing 365 and 730 days CIF
5. as KFRE only predict risk of dialysis, only array of Event 1 is analyzed, so the final CIF involved in this evaluation is (2, n_samples)
6. therefore, at this stage, all CIF joined the analysis will be sliced to get the shape of (2, n_samples) -> ([CIF at 730 DAYS, cif AT 1825 DAYS], N_samples)
7. now for CIF at 730 days:
7a get the CIF of ensembled model, KFRE (4 varaiable), KFRE (8 variable), null model at 730 days, at this moment the shape of array should be (n_samples, )
7b. for each CIF, calculate the integrated brier score and concordance index with 95% CI, than we can calculate the index of prediction accuracy (IPA) of ensembled model (IPA = 1 – Brier[model]/Brier[null])
7b the CIF of null model cut to 10 quantiles
7c. in each qualtile, we should be able to get the mean CIF of null model (observed risk), ensembled model, KFRE (4 varaiable), KFRE (8 variable)
7d. do the same at 1825 days
7e. save the data as json
8. Then now we do the plot

