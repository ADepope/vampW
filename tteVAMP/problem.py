import numpy as np

class Problem:
    n = None
    m = None
    model = None
    learn_params = None
    censored = None

    def __init__(self, n=None, m=None, la=None, sigmas=None, omegas=None, model=None, **kwargs):
        self.prior_instance = self.Prior(la, sigmas, omegas, **kwargs)
        self.model = model
        self.n = n
        self.m = m
        self.censored = np.zeros(n)
    
    class Prior:
        la = 0.5
        sigmas = [1]
        omegas = [1]
        # distribution_parameters = {}

        def __init__(self, la, sigmas, omegas, **kwargs):
            self.la = la
            self.sigmas = sigmas
            self.omegas = omegas
            self.distribution_parameters = {k: v for k, v in kwargs.items()}

    class Hyperparams:
        alpha = None
        mu = None
        sigma = None
        theta = None
        kappa = None
        def __init__(self, model, *args):
            if Problem.model == 'Weibull':
                # mu = args[0], alpha = args[1]
                self.mu = args[0]
                self.alpha = args[1]
            elif Problem.model == 'Gamma':
                # mu = args[0], kappa = args[1], theta = args[2]
                self.mu = args[0]
                self.kappa = args[1]
                self.theta = args[2]
            elif Problem.model == 'LogNormal':
                # mu = args[0], sigma = args[1]
                self.mu = args[0]
                self.sigma = args[1]
            else:
                raise Exception(Problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'.")
