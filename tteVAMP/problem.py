class problem:
    n = None
    m = None
    model = None
    learn_params = None

    def __init__(self, n, m, la, sigma, model):
        self.prior(la, sigma)
        self.model = model
        self.n = n
        self.m = m
    
    class prior:
        la = 0.5
        sigma = 1
        def __init__(self, la, sigma):
            self.la = la
            self.sigma = sigma


