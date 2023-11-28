import time
import numpy as np
from scipy.stats import norm
from tteVAMP import problem, denoisers, em, state_evolution

# prior, y, alpha, mu, maxiter, beta_true
def infere(X, y, gam1, r1, tau1, p1, problem, maxiter, beta_true=None):
    # accessing information about prior and model hyperparameters
    prior = problem.prior
    alpha = prior.alpha
    mu = prior.mu
    sigma = prior.sigma
    theta = prior.theta
    kappa = prior.kappa
    
    #computing SVD decomposition of X
    [n,m] = X.shape
    tSVD_start = time.perf_counter()
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    tSVD_stop = time.perf_counter()
    print(f"[numpy] Elapsed time: {tSVD_stop - tSVD_start:.3f} s")
    print("s.shape = ", s.shape)
    Xbeta_true = X @ beta_true

    #storing measure of recovery quality
    l2_errs_x = []
    corrs_x = []
    l2_errs_z = []
    corrs_z = []
    
    for it in range(maxiter):
        print("**** iteration = ", it, " **** \n" )
        
        # Denoising x
        print("->DENOISING")
        vect_den_beta = lambda x: den_beta(x, gam1, prior)
        x1_hat = vect_den_beta(r1)
        #print("shape of x1_hat = ", x1_hat.shape )
        #print("shape of beta_true = ", beta_true.shape )
        print("x1_hat[2] = ", x1_hat[2])
        if np.linalg.norm(x1_hat) != 0:
            # reporting quality of estimation
            corr = np.dot(x1_hat.transpose(), beta_true) / np.linalg.norm(x1_hat) / np.linalg.norm(beta_true)
            print("corr(x1_hat, beta_true) = ", corr[0][0])
            corrs_x.append(corr[0][0])
            l2_err = np.linalg.norm(x1_hat - beta_true) / np.linalg.norm(beta_true)
            print("l2 error for x1_hat = ", l2_err)
            l2_errs_x.append(l2_err)
        alpha1 = np.mean( der_den_beta(r1, gam1, prior) )
        print("alpha1 = ", alpha1)
        r2 = (x1_hat - alpha1 * r1) / (1-alpha1)
        gam2 = gam1 * (1-alpha1) / alpha1
        print("true gam2 = ", 1.0 / np.var(r2 - beta_true))
        print("gam2 = ", gam2)
        
        # Denoising z
        z1_hat = den_z(p1, tau1, y, alpha, mu) 
        # reporting quality of estimation
        corr = np.dot(z1_hat.transpose(), Xbeta_true) / np.linalg.norm(z1_hat) / np.linalg.norm(Xbeta_true)
        print("corr(z1_hat, X*beta_true) = ", corr[0][0])
        corrs_z.append(corr[0][0])
        l2_err = np.linalg.norm(z1_hat - Xbeta_true) / np.linalg.norm(Xbeta_true)
        print("l2 error for z1_hat = ", l2_err)
        l2_errs_z.append(l2_err)
        beta1 = np.sum( der_den_z(p1, tau1, y, alpha, mu) ) / n
        print("beta1 = ", beta1)
        p2 = (z1_hat - beta1 * p1) / (1-beta1)
        tau2 = tau1 * (1-beta1) / beta1
        print("true tau2 = ", 1.0 / np.var(p2 - Xbeta_true))
        print("tau2 =", tau2)
        
        # LMMSE estimation of x
        print("->LMMSE")
        dk = 1.0 / (tau2 * s * s + gam2)
        x2_hat = vh.transpose() @ np.diag(dk) @ (tau2 * np.diag(s).transpose() @ u.transpose() @ p2 + gam2 * vh @ r2)
        print("corr(x2_hat, beta_true) = ", np.dot(x2_hat.transpose(), beta_true) / np.linalg.norm(x2_hat) / np.linalg.norm(beta_true))
        print("l2 error for x2_hat = ", np.linalg.norm(x2_hat - beta_true) / np.linalg.norm(beta_true))
        alpha2 = np.sum( gam2 / (tau2 * s * s + gam2) ) / m;
        print("alpha2 = ", alpha2)
        r1 = (x2_hat - alpha2 * r2) / (1-alpha2)
        gam1 = gam2 * (1-alpha2) / alpha2
        print("true gam1 = ", 1.0 / np.var(r1 - beta_true))
        print("gam1 = ", gam1)
        
        # LMMSE estimation of z
        z2_hat = np.matmul(X, x2_hat)
        print("corr(z2_hat, beta_true) = ", np.dot(z2_hat.transpose(), Xbeta_true) / np.linalg.norm(z2_hat) / np.linalg.norm(Xbeta_true))
        print("l2 error for z2_hat = ", np.linalg.norm(z2_hat - Xbeta_true) / np.linalg.norm(Xbeta_true))
        beta2 = (1-alpha2) * m / n;
        p1 = (z2_hat - beta2 * p2) / (1-beta2)
        tau1 = tau2 * (1-beta2) / beta2
        print("true tau1 = ", 1.0 / np.var(p1 - Xbeta_true))
        print("tau1 = ", tau1)
        print("\n")
    return x1_hat, gam1, corrs_x, l2_errs_x, corrs_z, l2_errs_z