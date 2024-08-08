import numpy as np
import sympy
emc = float( sympy.S.EulerGamma.n(10) )
from tteVAMP.problem import Problem
from tteVAMP.simulations import *
from tteVAMP.vamp import *
from tteVAMP.utils import plot_metrics
from bed_reader import open_bed, sample_file
import sympy
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
p=0.4
# lambda "la" is the proportion of the induced sparsity
la=0.05
omega=1
# Heritability
h2=0.9
gam1 = 1e-2
tau1 = 1e-1
data_type = "real"
bed_file = '/nfs/scistore17/robingrp/human_data/adepope_preprocessing/PNAS_traits/HT/800k/ukb22828_UKB_EST_v3_ldp08_fd_HT_test_800k.bed'
bed = open_bed(bed_file)
# # X = np.array(bed.read())
# X = np.array(bed.read(index=np.s_[:15000,:15000]))
# Define the total number of columns
total_columns = 887060

# Randomly select 15,000 columns from the total columns
random_columns = sorted(random.sample(range(total_columns), 15000))

# Read the BED file with a random subset of 15,000 rows and the selected columns
# X = np.array(bed.read(index=(np.s_[:15000], random_columns))) 
X = np.array(bed.read(index=(np.s_[:15000], random_columns))) 
n,m = X.shape
sigma=h2/m/la
print(f"The shape of X is: {n,m}")
nan_mask = np.isnan(X)
tot_nans_before = np.sum(nan_mask)
print(f'The number of NaNs in the matrix before standardization is: {tot_nans_before}')

mu=np.full((n,1), 0) 
# X = sim_geno(n,m,p)
column_means = np.nanmean(X, axis=0)
column_stds = np.nanstd(X, axis=0)
valid_stds = not 0 in column_stds
print(f"Are standard deviations valid? {valid_stds}")
X = (X - column_means) / column_stds
nan_mask = np.isnan(X)
tot_nans_after = np.sum(nan_mask)
print(f'The number of NaNs in the matrix after standardization is: {tot_nans_after}')

X = np.nan_to_num(X)
nan_mask = np.isnan(X)
tot_nans_after = np.sum(nan_mask)
print(f'The number of NaNs in the matrix after standardization and nan to num is: {tot_nans_after}')
variances = np.var(X[:, :5], axis=0)
beta = sim_beta(m, la, sigma)
y, alpha = sim_pheno_Weibull(X, beta, mu, h2)
print(f"Generated alpha value is: {alpha}")
print(f"Variance of Xb: {np.var(X@beta)}")
print(f"Variance of b: {np.var(beta)}")

# directory = '/nfs/scistore17/robingrp/jbajzik/sgvamp/tte/data/W4/'

# # if not os.path.exists(f'{directory}'):
# #     os.makedirs(f'{directory}')

# y = np.log(y)
# indices =  np.arange(0,n)
# df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})
# df.to_csv(f"{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_.phen", sep=' ', index=None, header=None)
# vector = np.ones(n)
# np.savetxt(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_status.fail', vector, fmt='%d')
# zarr.save(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_X.zarr', X)
# np.save(f'{directory}beta.npy', beta)
# z_true = X@beta
# np.save(f'{directory}z_true.npy', z_true)
# # Store variables in a dictionary
# variables = {
#     "n": n,
#     "m": m,
#     "p": p,
#     "la": la,
#     "h2": h2,
#     "data_type": data_type,
#     "directory": directory,
#     "sigma": "h2/m/la",
#     "alpha": alpha,
#     "mu": float(mu[0][0]),
#     "Variance of Xb": float(np.var(X@beta)),
#     "Variance of b": float(np.var(beta)),
#     "Scale of y": "log",
#     "nans in X": float(tot_nans),
#     "validity of stds": f"Boolean: {valid_stds}"
# }

# # Save dictionary as JSON file
# with open(f'{directory}{data_type}_{n}x{m}_h2_{h2}_la_{la}_hyperparameters.json', 'w') as json_file:
#     json.dump(variables, json_file, indent=4)


# X = sim_geno(15000, 15000, p)
# print(X)

# n, m = X.shape
# mu=np.full((n,1), 0) 
# sigma=h2/m/la

# # Compute column means and standard deviations, ignoring NaNs
# column_means = np.nanmean(X, axis=0)
# column_stds = np.nanstd(X, axis=0)
# print(f"Are standard deviations valid? {not 0 in column_stds}")

# # Normalize the data
# X = (X - column_means) / column_stds
# X = np.nan_to_num(X)

# beta = sim_beta(m, la, sigma)
# y, alpha = sim_pheno_Weibull(X, beta, mu, h2)

maxiter = 100
problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model='Weibull', mu=mu)

print("gam1 = ", gam1)
print("tau1 = ", tau1)
print("alpha = ", alpha)

# we start with an initialization that compleately complies with the assumptions
r1 = np.zeros((m,1))
p1 = np.zeros((n,1)) 

problem_instance.prior_instance.distribution_parameters['alpha']=alpha

est, gam1, corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, a, ps, dl_dmus, z1_hats, x1_hats =  infere_con_grad_dampen(X, y, gam1, r1, tau1, p1, problem_instance, maxiter, beta, True, True)
# est, gam1, corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, a, ps, dl_dmus, z1_hats =  infere_dampen(X, y, gam1, r1, tau1, p1, problem_instance, maxiter, beta, False False)
plot_metrics(corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, dl_dmus, a, ps, mu[0][0], alpha, n, m)