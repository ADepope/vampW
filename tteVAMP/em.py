import scipy
import numpy as np

# Euler -Mascheroni constant
emc = float( sympy.S.EulerGamma.n(10) )

def update_Weibull_alpha_ew(alpha, y, mu, z_hat, xi):
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