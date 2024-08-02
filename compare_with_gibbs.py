import numpy as np
import sympy
emc = float( sympy.S.EulerGamma.n(10) )
from tteVAMP.problem import Problem
from tteVAMP.simulations import *
from tteVAMP.vamp import *
from tteVAMP.utils import plot_metrics
from bed_reader import open_bed, sample_file


np.random.seed(42)
p=0.4
# lambda "la" is the proportion of the induced sparsity
la=0.05
sigma=1
omega=1
# Heritability
h2=0.9
gam1 = 1e-2
tau1 = 1e-1
n,m = 800, 800

mu=np.full((n,1), 0) 
X = sim_geno(n,m,p)
column_means = np.nanmean(X, axis=0)
column_stds = np.nanstd(X, axis=0)
print(f"Are standard deviations valid? {not 0 in column_stds}")
X = (X - column_means) / column_stds
X = np.nan_to_num(X)
variances = np.var(X[:, :5], axis=0)

# Print the variances
for i, var in enumerate(variances, start=1):
    print(f"Variance of column {i}: {var}")

beta = sim_beta(m, la, h2/m/la)
y, alpha = sim_pheno_Weibull(X, beta, mu, h2)

# bed_file = '/nfs/scistore17/robingrp/human_data/adepope_preprocessing/PNAS_traits/HT/800k/ukb22828_UKB_EST_v3_ldp08_fd_HT_test_800k.bed' 
# bed = open_bed(bed_file)
# # X = np.array(bed.read())
# X = np.array(bed.read(index=np.s_[:15000,:15000]))
# # X = sim_geno(800, 800, p)
# # print(X)

# n, m = X.shape
# mu=np.full((n,1), 0) 

# # Compute column means and standard deviations, ignoring NaNs
# column_means = np.nanmean(X, axis=0)
# column_stds = np.nanstd(X, axis=0)
# print(f"Are standard deviations valid? {not 0 in column_stds}")

# # Normalize the data
# X = (X - column_means) / column_stds
# # # Replace NaNs with zeros
# X = np.nan_to_num(X)

# beta = sim_beta(m, la, sigma/m)
# y, alpha = sim_pheno_Weibull(X, beta, mu, h2)

maxiter = 20
problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model='Weibull', mu=mu)

print("gam1 = ", gam1)
print("tau1 = ", tau1)
print("alpha = ", alpha)

# we start with an initialization that compleately complies with the assumptions
r1 = np.zeros((m,1))
p1 = np.zeros((n,1)) 

problem_instance.prior_instance.distribution_parameters['alpha']=alpha

# est, gam1, corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, a, ps, dl_dmus, z1_hats, x1_hats =  infere_con_grad(X, y, gam1, r1, tau1, p1, problem_instance, maxiter, beta, False, False)
est, gam1, corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, a, ps, dl_dmus, z1_hats =  infere(X, y, gam1, r1, tau1, p1, problem_instance, maxiter, beta, True, True)
plot_metrics(corrs_x, l2_errs_x, corrs_z, l2_errs_z, mus, alphas, dl_dmus, a, ps, mu[0][0], alpha, n, m)


# Are standard deviations valid? True
# Variance of column 1: 0.9999999999999883
# Variance of column 2: 0.9999999999999961
# Variance of column 3: 0.99999999999998
# Variance of column 4: 0.9999999999999999
# Variance of column 5: 1.0000000000000195
# Variance of beta: 0.02306609879041371
# gam1 =  0.01
# tau1 =  0.1
# alpha =  3.5040934324097504
# s.shape =  (800,)