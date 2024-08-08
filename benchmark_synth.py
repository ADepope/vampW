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


def calculate_score(x1_hat, beta_true, title="corr(x1_hat, beta_true) = "):
    # corr = np.dot(x1_hat.transpose(), beta_true) / np.linalg.norm(x1_hat) / np.linalg.norm(beta_true)
    # print("corr(x1_hat, beta_true) = ", corr[0][0])
    corr = x1_hat.T @ beta_true / np.linalg.norm(x1_hat) / np.linalg.norm(beta_true)
    print(title, corr[0][0])
def benchmark(X, data, df, beta_true, iters=100):
    average_last_100 = np.mean(data[-iters:], axis=0)
    print(f"Average value of mu over the last {iters} iterations: {np.mean(df['mu'][-100:])}")
    print(f"Average value of alpha over the last {iters} iterations: {np.mean(df['alpha'][-100:])}")

    n, m = X.shape
    calculate_score(average_last_100.reshape(m, 1), beta_true, "alignment beta = ")
    calculate_score(X@average_last_100.reshape(m, 1), X@beta_true,"alignment z = ")
    print(f"l2 error on x: {np.linalg.norm(average_last_100.reshape(m, 1) - beta_true) / np.linalg.norm(beta_true)}")
    print(f"l2 error on z: {np.linalg.norm(X@average_last_100.reshape(m, 1) - X@beta_true) / np.linalg.norm(X@beta_true)}")

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
X = np.array(bed.read(index=(np.s_[:15000], random_columns))) 
n,m = X.shape

mu=np.full((n,1), 0) 
# X = sim_geno(n,m,p)
column_means = np.nanmean(X, axis=0)
column_stds = np.nanstd(X, axis=0)
valid_stds = not 0 in column_stds
print(f"Are standard deviations valid? {valid_stds}")
X = (X - column_means) / column_stds
nan_mask = np.isnan(X)
tot_nans = np.sum(nan_mask)
print(f'The number of NaNs in the matrix is: {tot_nans}')
X = np.nan_to_num(X)
variances = np.var(X[:, :5], axis=0)
beta = sim_beta(m, la, h2/m/la)
y, alpha = sim_pheno_Weibull(X, beta, mu, h2)

print("Real\n\n")
directory = "/nfs/scistore17/robingrp/jbajzik/bayesw_py/output/W3/"
w2_betas_name = "BayesW_out_betas_bayesw_W3_1.tsv"
w2_progress = "BayesW_out_bayesw_W3_1.tsv"
file_path1 = f"{directory}{w2_betas_name}"
file_path2 = f"{directory}{w2_progress}"

# Load the TSV file into a NumPy array
data = np.loadtxt(file_path1, delimiter='\t')
df = pd.read_csv(file_path2, sep='\t')
benchmark(X, data, df, beta, 100)

# print("Synthetic!!!\n\n")
# directory = "/nfs/scistore17/robingrp/jbajzik/bayesw_py/output/W4/"
# w2_betas_name = "BayesW_out_betas_bayesw_W4_1.tsv"
# w2_progress = "BayesW_out_bayesw_W4_1.tsv"
# file_path1 = f"{directory}{w2_betas_name}"
# file_path2 = f"{directory}{w2_progress}"

# # Load the TSV file into a NumPy array
# data = np.loadtxt(file_path1, delimiter='\t')
# df = pd.read_csv(file_path2, sep='\t')
# benchmark(X, data, df, beta, 1000)