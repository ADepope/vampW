import numpy as np
import sympy
emc = float( sympy.S.EulerGamma.n(10) )
from tteVAMP.problem import Problem
from tteVAMP.simulations import *
from tteVAMP.vamp import infere
from tteVAMP.utils import plot_metrics
import pandas as pd
import zarr
import os
import json
import random  # This ensures you're using Python's built-in random module.

np.random.seed(42)
n=15000
m=15000
p=0.4
la=0.05
# sigma=1
h2=0.9

data_type = "synthetic"
mu=np.full((n,1), 0) 
X = sim_geno(n,m,p)
column_means = np.nanmean(X, axis=0)
column_stds = np.nanstd(X, axis=0)
print(f"Are standard deviations valid? {not 0 in column_stds}")
X = (X - column_means) / column_stds
nan_mask = np.isnan(X)
tot_nans = np.sum(nan_mask)
print(f'The number of NaNs in the matrix is: {tot_nans}')
X = np.nan_to_num(X)
variances = np.var(X[:, :5], axis=0)
beta = sim_beta(m, la, h2/m/la)
y, alpha = sim_pheno_Weibull(X, beta, mu, h2)
print(f"Generated alpha value is: {alpha}")
print(f"Variance of Xb: {np.var(X@beta)}")
print(f"Variance of b: {np.var(beta)}")

# total_columns = 800

# # Randomly select 15,000 columns from the total columns
# random_columns = sorted(random.sample(range(total_columns), 500))

# # Read the BED file with a random subset of 15,000 rows and the selected columns
# X = X[:, random_columns]
# n,m = X.shape

directory = '/nfs/scistore17/robingrp/jbajzik/sgvamp/tte/data/W4/'

# if not os.path.exists(f'{directory}gibbs'):
#     os.makedirs(f'{directory}gibbs')

y = np.log(y)
indices =  np.arange(0,n)
df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})
df.to_csv(f"{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_.phen", sep=' ', index=None, header=None)
vector = np.ones(n)
np.savetxt(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_status.fail', vector, fmt='%d')
zarr.save(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_X.zarr', X)

# Store variables in a dictionary
variables = {
    "n": n,
    "m": m,
    "p": p,
    "la": la,
    "h2": h2,
    "data_type": "synthetic",
    "directory": directory,
    "sigma": "h2/m/la",
    "alpha": alpha,
    "mu": float(mu[0][0]),
    "Variance of Xb": float(np.var(X@beta)),
    "Variance of b": float(np.var(beta)),
    "Scale of y": "log",
    "nans in X": float(tot_nans)
}

# Save dictionary as JSON file
with open(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_hyperparameters.json', 'w') as json_file:
    json.dump(variables, json_file, indent=4)
