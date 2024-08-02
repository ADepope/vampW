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

bed_file = '/nfs/scistore17/robingrp/human_data/adepope_preprocessing/PNAS_traits/HT/800k/ukb22828_UKB_EST_v3_ldp08_fd_HT_test_800k.bed' 
bed = open_bed(bed_file)
# X = np.array(bed.read())
X = np.array(bed.read(index=np.s_[:15000,:15000]))
# X = sim_geno(800, 800, p)
# print(X)

n, m = X.shape
mu=np.full((n,1), 0) 

# Compute column means and standard deviations, ignoring NaNs
column_means = np.nanmean(X, axis=0)
column_stds = np.nanstd(X, axis=0)
print(f"Are standard deviations valid? {not 0 in column_stds}")

# Normalize the data
X = (X - column_means) / column_stds
# # Replace NaNs with zeros
X = np.nan_to_num(X)
 
beta = sim_beta(m, la, h2/m/la)
y, alpha = sim_pheno_Weibull(X, beta, mu, h2)
y = np.log(y)
indices =  np.arange(0,n)
df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})



directory = 'change to your directory'
df.to_csv(f"{directory}{n}x{m}_h2_0d9_la_0d05.phen", sep=' ', index=None, header=None)
vector = np.ones(n)
np.savetxt(f'{directory}status.fail', vector, fmt='%d')
zarr.save(f'{directory}X.zarr', X)

# Store variables in a dictionary
variables = {
    "n": n,
    "m": m,
    "p": p,
    "la": la,
    "h2": h2,
    "data_type": "human genotype; synthetic beta and y",
    "directory": directory,
    "sigma": "h2/m/la",
    "alpha": alpha,
    "mu": float(mu[0][0]),
    "Variance of Xb": float(np.var(X@beta)),
    "Variance of b": float(np.var(beta)),
    "Scale of y": "log",
}

# Save dictionary as JSON file
with open(f'{directory}hyperparameters.json', 'w') as json_file:
    json.dump(variables, json_file, indent=4)