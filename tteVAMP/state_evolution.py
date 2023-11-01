import numpy as np
from numpy import random
from tteVAMP import denoisers

def Abeta1(gam1, beta_true, prior):
    R1 = random.normal(loc=beta_true, scale=1.0/np.sqrt(gam1), size=None) #scale = standard deviation
    return np.mean( der_den_beta(R1, gam1, prior) )

def Az1(tau1, z_true, y, alpha)
    n = z_true.shape[0]
    P1 = random.normal(loc=z_true, scale=1.0/np.sqrt(tau1), size=None) #scale = standard deviation
    return (np.sum( der_den_z(p1, tau1, y, alpha, mu) ) / n)

def Abeta2(gam2, tau2, s):
    return np.mean(1.0/(tau2 * s + gam2))

def Az2(gam2, tau2, s):
    return np.mean(tau2 * np.power(s,2) / (tau2 * np.power(s,2) + gam2))
    
def state_evolution(maxiter, s, gam1, tau1, beta_true, z_true, prior, y, alpha):
    # we assume beta_true is np.array
    m = beta_true.shape[0] 
    print("****STATE EVOLUTION****")
    gam1s = []
    gam2s = []
    tau1s = []
    tau2s = []
    for it in range(maxiter):
        print("**** iteration = ", it, " **** \n" )
        # DENOISING STEP
        alpha1 = Abeta1(gam1, beta_true, prior)
        print("alpha1 = ", alpha1)
        v1 = Az1(tau1, z_true, y, alpha)
        print("v1 = ", v1)
        gam2 = gam1 * (1-alpha1) / alpha1
        print("gam2 = ", gam2)
        gam2s.append(gam2)
        tau2 = tau1 * (1-v1) / v1
        print("tau2 = ", tau2)
        tau2s.append(tau2s)
        # LMMSE STEP
        alpha2 = Abeta2(gam2, tau2, s)
        print("alpha2 = ", alpha2)
        v2 = Az2(gam2, tau2)
        print("v2 = ", v2)
        gam1 = gam2 * (1-alpha2) / alpha2
        print("gam1 = ", gam1)
        gam1s.append(gam1)
        tau1 = tau2 * (1-v2) / v2
        print("tau1 = ", tau1)
        tau1s.append(tau1)
    
        
        


    