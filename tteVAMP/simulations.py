from numpy import random
import numpy as np
import sympy
import scipy

from tteVAMP import problem

# Euler -Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

#function for simultaing genotype matrix and Weibull distributed phenotypes

def sim_geno(n,m,p): # checked!
    X = random.binomial(2, p, size=[n,m]) / np.sqrt(n)
    # for debugging purposes we simulate a Gaussian matrix and scale it 
    #X = random.normal(loc=0.0, scale=1.0, size=[n,m]) / np.sqrt(n)
    return X

def sim_beta(m, la, sigma): # checked!
    beta = random.normal(loc=0.0, scale=np.sqrt(sigma[0]), size=[m,1]) # scale = standard deviation
    beta *= random.binomial(1, la, size=[m,1])
    return beta

def sim_pheno_Weibull(X, beta, mu, h2):
    # logY_i = mu + xi beta + c(wi - Ewi), wi = - standard Gumbel distribution
    # beta is mx1 vector 
    # mu is nx1 vector 
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    varwi = np.pi * np.pi / 6
    c = np.sqrt((1/h2-1) * sigmaG / varwi)
    wi = -random.gumbel(loc=0.0, scale=1.0, size=[n,1])
    y = np.exp( mu + g + c * (wi + emc) )
    alpha = 1.0 / c
    return y, alpha

def sim_pheno_ExpGamma(X, beta, mu, h2, kappa):
    # logY_i = mu + xi beta + sigma * wi, wi = standard Normal variable
    # beta is mx1 vector 
    # mu is nx1 vector 
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    sigmaE = np.sqrt( (1/h2-1) * sigmaG )
    theta = sigmaE / scipy.special.polygamma(1, kappa)
    # mu tilde
    mut = mu + g - theta * scipy.special.polygamma(0, kappa)
    y = random.gamma(shape=kappa, scale=theta, size=[n,1]) + mut
    return y, kappa, theta, mut

def sim_pheno_LogNormal(X, beta, mu, h2):
    # logY_i = mu + xi beta + sigma * wi, wi = standard Normal variable
    # beta is mx1 vector 
    # mu is nx1 vector 
    [n,m] = X.shape
    g = np.matmul(X, beta)
    sigmaG = np.var(g)
    sigma = np.sqrt( (1/h2-1) * sigmaG )
    w = random.normal(loc=0.0, scale=1.0, size=[n,1])
    y = np.exp(mu + g + sigma * w)
    return y, sigma
    
def sim_model(problem,h2,p, kappa=None):
    X = sim_geno(problem.n, problem.m, p)
    beta = sim_beta(problem.m, problem.prior.la, problem.prior.sigmas)
    mu = np.zeros((problem.n,1))
    print(problem.model)
    if problem.model == 'Weibull':
        return X, beta, sim_pheno_Weibull(X, beta, mu, h2)
    elif problem.model == 'Gamma':
        return X, beta, sim_pheno_ExpGamma(X, beta, mu, h2, kappa)
    elif problem.model == 'LogNormal':
        return X, beta, sim_pheno_LogNormal(X, beta, mu, h2)
    else:
        raise Exception(problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'")
    