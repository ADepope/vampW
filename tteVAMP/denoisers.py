from scipy.stats import norm
import numpy as np
import sympy
import scipy

# definition of Euler-Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

# denoiser of the signal beta
def den_beta(r,gam1,prior): 
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1)) # scale = standard deviation
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(sigma + 1.0/gam1))
    ratio = gam1 * r / (gam1 + 1/sigma) * B / (A + B)
    return ratio

def der_den_beta(r,gam1,prior): 
    A = (1-prior.la) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
    B = prior.la * norm.pdf(r, loc=0, scale=np.sqrt(sigma + 1.0/gam1))
    print("B / (A+B) = ", B[1] / (A[1]+B[1]))
    Ader = A * (-r*gam1)
    Bder = B * (-r) / (sigma + 1.0/gam1)
    BoverAplusBder = ( Bder * A - Ader * B ) / (A+B) / (A+B)
    print("gam1 / (gam1 + 1/sigma) = ", gam1 / (gam1 + 1/sigma))
    print("alpha1 part I = ", gam1 / (gam1 + 1/sigma) * B[1] / (A[1] + B[1]))
    print("alpha2 part II = ", BoverAplusBder[1] * r[1] * gam1 / (gam1 + 1.0/sigma) )
    ratio = gam1 / (gam1 + 1/sigma) * B / (A + B) + BoverAplusBder * r * gam1 / (gam1 + 1.0/sigma)
    return ratio


# denoiser of z
def den_z(p1, tau1, y, problem):
     if problem.model == 'Weibull':
         return den_z_Weibull(p1, tau1, y, problem.alpha, problem.mu)
     elif problem.model == 'Gamma':
         return den_z_Gamma(p1, tau1, y, problem.kappa, problem.theta, problem.mu)
     elif problem.model == 'LogNormal':
         return den_z_LogNormal(p1, tau1, y, problem.sigma, problem.mu)      

def der_den_z(p1, tau1, y, problem):
     if problem.model == 'Weibull':
         return der_den_z_Weibull(p1, tau1, y, problem.alpha, problem.mu)
     elif problem.model == 'Gamma':
         return der_den_z_Gamma(p1, tau1, y, problem.kappa, problem.theta, problem.mu)
     elif problem.model == 'LogNormal':
         return der_den_z_LogNormal(p1, tau1, y, problem.sigma, problem.mu) 
         
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
    return out

def der_den_z_Weibull(p1, tau1, y, alpha, mu):
    z = den_z(p1, tau1, y, alpha, mu)
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