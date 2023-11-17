import scipy
import numpy as np
import math

# Euler -Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

def update_Weibull_alpha_eq(alpha, y, mu, z_hat, xi):
    res = np.log(y) - mu - z_hat
    sum_res = np.sum(res)
    out = np.exp(-emc) * np.sum( np.exp(alpha * res + alpha**2/2/xi) * (res + alpha/xi) )
    return out - sum_res

def update_Weibull_alpha(y, mu, z_hat, alpha_old, xi):
    # y.shape = [n,1]
    # z_hat.shape = [n,1]
    out = scipy.optimize.fsolve(update_Weibull_alpha_eq, x0 = alpha_old, args=(y, mu, z_hat, xi))
    return out

def update_Weibull_alpha(y, mu_old, z_hat, alpha, xi):
    # y.shape = [n,1]
    # z_hat.shape = [n,1]
    n,_ = y.shape
    out = - np.log(n) / alpha - np.sum(np.log(y) - z_hat + alpha/2/xi) - emc/alpha
    return out

def update_LogNormal_mu_eq(y, mu, z_hat, sigma, censored):
    n,_ = y.shape
    out = np.zeros(n)
    #contribution of non-censored individuals
    out[censored==0] = (np.log(y[censored==0])-mu-z_hat[censored==0])/sigma/sigma
    #contribution of censored individuals (for such corresponding values of z_hat = x_i^T * beta_hat and y = censoring time)
    out[censored==1] = np.exp(-np.power(mu + z1_hat[censored==1]-np.log(y[censored==1),2) / 2 / sigma / sigma) * np.sqrt(2/math.pi) / sigma / (1+math.erf(mu + z1+hat[censored==1]-np.log(y[censored==1])))
    out = np.sum(out)
    return out   


def update_Prior(old_prior, r1, gam1):
    prior = old_prior
    r1 = np.asmatrix(r1)
    omegas = np.asmatrix(omegas)
    sigmas = np.asmatrix(sigmas)
    sigmas_max = old_prior.sigmas.max()
    gam1inv = 1.0/gam1
    # np.exp( - np.power(np.transpose(r1),2) / 2 @ (sigmas_max - sigmas) / (sigmas_max + gam1inv) / (sigmas + gam1inv)) has shape = (P,L) and  omegas / np.sqrt(gam1inv + sigmas) has shape = (1, L)
    beta_tilde=np.multiply( np.exp( - np.power(np.transpose(r1),2) / 2 @ (sigmas_max - sigmas) / (sigmas_max + gam1inv) / (sigmas + gam1inv)), omegas / np.sqrt(gam1inv + sigmas) )
    sum_beta_tilde = beta_tilde.sum(axis=1)
    beta_tilde=beta_tilde / sum_beta_tilde
    # pi.shape = (P, 1)
    pi = 1.0 / ( 1.0 + (1-prior.la * np.exp(-np.power(np.transpose(r1),2) / 2 * sigmas_max * gam1 / (sigmas_max + gam1inv) ) / np.sqrt(gam1inv) ) / sum_beta_tilde )
    gamma = np.divide(np.transpose(r1) * gam1, gam1 + 1.0/sigmas )
    # v.shape = (1,L)
    v = 1.0 / (gam1 + 1.0/sigmas)

    #updating sparsity level
    prior.la = np.mean(pi)
    #updating variances in the mixture
    prior.sigmas = (np.transpose(pi) @ np.multiply( beta_tilde , (np.power(gamma,2) + v)) ) / (np.transpose(pi) @ beta_tilde)
    #updating prior probabilities in the mixture
    prior.omegas = (np.transpose(pi) @ beta_tilde ) / np.sum(pi)
    
    return prior

