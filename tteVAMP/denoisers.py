from scipy.stats import norm
import numpy as np
import sympy
import scipy
from tteVAMP.problem import *

# definition of Euler-Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

# denoiser of the signal beta
def den_beta(r,gam1,problem): # checked!
    """
    This function returns the conditional expectation of the coefficients beta given the noisy estimate r
    The expectation is of the posterior distribution with the form of Spike and Slab mixture of Gaussians
    """
    prior = problem.prior_instance
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1)) # scale = standard deviation
    # Make the variance here match the generated beta
    # h2 / m / la
    print(f"Denoiser is using sigma: {prior.sigmas[0]}")
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(prior.sigmas[0] + 1.0/gam1))
    ratio = gam1 * r / (gam1 + 1/prior.sigmas[0]) * B / (A + B)
    return ratio

def der_den_beta(r,gam1,problem): # checked!
    prior = problem.prior_instance
    # Derivative of the Gaussians with respect to r
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(prior.sigmas[0] + 1.0/gam1))
    print("B / (A+B) = ", B[1] / (A[1]+B[1]))
    Ader = A * (-r*gam1)
    Bder = B * (-r) / (prior.sigmas[0] + 1.0/gam1)
    BoverAplusBder = ( Bder * A - Ader * B ) / (A+B) / (A+B)
    print("gam1 / (gam1 + 1/sigma) = ", gam1 / (gam1 + 1/prior.sigmas[0]))
    print("alpha1 part I = ", gam1 / (gam1 + 1/prior.sigmas[0]) * B[1] / (A[1] + B[1]))
    print("alpha2 part II = ", BoverAplusBder[1] * r[1] * gam1 / (gam1 + 1.0/prior.sigmas[0]) )
    ratio = gam1 / (gam1 + 1/prior.sigmas[0]) * B / (A + B) + BoverAplusBder * r * gam1 / (gam1 + 1.0/prior.sigmas[0])
    return ratio


# denoiser of z
def den_z(p1, tau1, y, problem):
    #  print(f"Inside denoiser den_z! {problem.model}")
     d = problem.prior_instance.distribution_parameters
    #  print(f"Mu: \n\n\n")
    #  print(d['mu'])
     if problem.model == 'Weibull':
         
         alpha, mu = None, None
         if 'alpha' in d: alpha = d['alpha']
         if 'mu' in d: mu = d['mu'][0][0]
        #  print(f"alpha: {alpha}, mu: {mu}")
         r = den_z_Weibull(p1, tau1, y, alpha, mu)
        #  print(f"r: {r}")
         return r
     elif problem.model == 'Gamma':
         theta, kappa, mu = None, None, None
         if 'theta' in d: alpha = d['theta']
         if 'kappa' in d: alpha = d['kappa']
         if 'mu' in d: mu = d['mu'][0][0]
         return den_z_Gamma(p1, tau1, y, kappa, theta, mu)
     elif problem.model == 'LogNormal':
         sigma, mu = None, None
         if 'sigma' in d: alpha = d['sigma']
         if 'mu' in d: mu = d['mu'][0][0]
         return den_z_LogNormal(p1, tau1, y, sigma, mu)      

def der_den_z(p1, tau1, y, problem):
     d = problem.prior_instance.distribution_parameters
     if problem.model == 'Weibull':
         alpha, mu = None, None
         if 'alpha' in d: alpha = d['alpha']
         if 'mu' in d: mu = d['mu'][0][0]
         r = der_den_z_Weibull(p1, tau1, y, alpha, mu)
         return r
     elif problem.model == 'Gamma':
         theta, kappa, mu = None, None, None
         if 'theta' in d: alpha = d['theta']
         if 'kappa' in d: alpha = d['kappa']
         if 'mu' in d: mu = d['mu'][0][0]
         return der_den_z_Gamma(p1, tau1, y, kappa, theta, mu)
     elif problem.model == 'LogNormal':
         sigma, mu = None, None
         if 'sigma' in d: alpha = d['sigma']
         if 'mu' in d: mu = d['mu'][0][0]
         return der_den_z_LogNormal(p1, tau1, y, sigma, mu) 
         
# Weibull model
def den_z_non_lin_eq_Weibull(z, tau1, p1, y, alpha, mu):
    """
    Performs MAP estimation of z
    Defines the objective to maximize
    Maximizing the expression below is equivalent to maximizing the likelihood of z
    We can treat the components of z as independent under the simplifying assumptions
    """ 
    res = tau1 * (z-p1) + alpha - alpha * np.power(y, alpha) * np.exp(- alpha * (mu + z) - emc)
    return res
    
def den_z_Weibull(p1, tau1, y, alpha, mu): 
    n,_ = p1.shape
    out = np.zeros((n,1))
    for i in range(0, n):
        out[i] = scipy.optimize.fsolve(den_z_non_lin_eq_Weibull, x0 = p1[i], args=(tau1, p1[i], y[i], alpha, mu) )
    # print(f"Out: {out}")
    return out

def der_den_z_Weibull(p1, tau1, y, alpha, mu):
    z = den_z(p1, tau1, y, Problem(model = 'Weibull', mu=np.full((y.shape[0],1), 0), alpha=alpha))
    # print(f"Z: {z}")
    nom = tau1
    den = tau1 + alpha * alpha * np.power(y, alpha) * np.exp(- alpha * (mu + z) - emc)
    return nom / den

# LogNormal model
def den_z_LogNormal(p1, tau1, y, sigma, mu):
    out = (np.log(y) - mu + p1 * tau1 * sigma * sigma) / (1 + tau1 * sigma * sigma)
    return out

def der_den_z_LogNormal(p1, tau1, y, sigma, mu): 
    z = tau1 * sigma * sigma / (1 + tau1 * sigma * sigma)
    return nom / den

# ExpGamma model
def den_z_non_lin_eq_Gamma(z, tau1, p1, y, kappa, theta, mu): 
    res = theta * (np.log(theta * tau1 * (z-p1) + kappa) - scipy.special.polygamma(0, kappa)) + (np.log(y) - mu - z)
    return res

def den_z_Gamma(p1, tau1, y, kappa, theta, mu): 
    n,_ = p1.shape
    out = np.zeros((n,1))
    for i in range(0, n):
        out[i] = scipy.optimize.fsolve(den_z_non_lin_eq_Gamma, x0 = p1[i], args=(tau1, p1[i], y[i], kappa, theta, mu) )
    return out

def der_den_z_Gamma(p1, tau1, y, kappa, theta, mu):
    z = den_z(p1, tau1, y, kappa, theta, mu)
    out = tau1 * theta ( tau1 * theta - tau1 * (z-p1) )
    return out