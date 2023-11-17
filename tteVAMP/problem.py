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


