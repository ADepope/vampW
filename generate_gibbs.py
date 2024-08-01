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

np.random.seed(42)
n=800
m=800
p=0.4
la=0.05
sigma=1
h2=0.9


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


print(f"Generated alpha value is: {alpha}")
print(f"Variance of Xb: {np.var(X@beta)}")
print(f"Variance of b: {np.var(beta)}")
directory = '/nfs/scistore17/robingrp/jbajzik/sgvamp/tte/data/'

# if not os.path.exists(f'{directory}gibbs'):
#     os.makedirs(f'{directory}gibbs')

indices =  np.arange(0,n)
df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})
df.to_csv(f"{directory}{n}x{m}_h2_0d9_la_0d05.phen", sep=' ', index=None, header=None)
vector = np.ones(n)
np.savetxt(f'{directory}status.fail', vector, fmt='%d')
zarr.save(f'{directory}X.zarr', X)