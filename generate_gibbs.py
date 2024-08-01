import numpy as np
import sympy
emc = float( sympy.S.EulerGamma.n(10) )
from tteVAMP.problem import Problem
from tteVAMP.simulations import sim_model
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
# This is where the sigma is defined. Note that the scope of this definition extends to gvamp
sigma=1
omega=1
h2=0.9
gam1 = 1e-2
tau1 = 1e-1
mu=np.full((n,1), 0) 
maxiter = 100
problem_instance = Problem(n=n, m=m, la=la, sigmas = [sigma], omegas=[omega], model='Weibull', mu=mu)
X,beta,y,alpha = sim_model(problem_instance,h2,p )
directory = '/nfs/scistore17/robingrp/jbajzik/sgvamp/tte/'

# if not os.path.exists(f'{directory}gibbs'):
#     os.makedirs(f'{directory}gibbs')

indices =  np.arange(0,m)
df = pd.DataFrame({'IID': indices, 'FID': indices, 'y': y.squeeze(-1)})
df.to_csv(f"{directory}800x800_h2_0.6_la_0.5.phen", sep='\t', index=None, header=None)

vector = np.ones(m)
np.savetxt(f'{directory}status.fail', vector, fmt='%d')

zarr.save(f'{directory}X.zarr', X)