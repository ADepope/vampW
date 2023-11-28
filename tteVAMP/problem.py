class problem:
    n = None
    m = None
    model = None
    learn_params = None
    censored = None

    def __init__(self, n, m, la, sigmas, omegas, model):
        self.prior(la, sigmas, omegas)
        self.model = model
        self.n = n
        self.m = m
        self.censored=np.zeros(n)
    
    class prior:
        la = 0.5
        sigmas = [1]
        omegas = [1]
        def __init__(self, la, sigmas, omegas):
            self.la = la
            self.sigmas = sigmas
            self.omegas = omegas

    class hyperparams:
        alpha = None
        mu = None
        sigma = None
        theta = None
        kappa = None
        def __init__(self, model, *args):
            if problem.model == 'Weibull':
                # mu = args[0], alpha = args[1]
                self.mu = args[0]
                self.alpha = args[1]
            elif problem.model == 'Gamma':
                # mu = args[0], kappa = args[1], theta = args[2]
                self.mu = args[0]
                self.kappa = args[1]
                self.theta = args[2]
            elif problem.model == 'LogNormal':
                # mu = args[0], sigma = args[1]
                self.mu = args[0]
                self.sigma = args[1]
            else:
                raise Exception(problem.model, " is not a valid model. Allowed models are: 'Weibull', 'Gamma' and 'LogNormal'.")
            

