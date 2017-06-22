import numpy as np

from .features import LJPotential
from .utils import calc_distances
from .hmc import HMC

class NoPrior(object):
    """
    Fake prior used to switch off the physical prior
    """
    @property
    def r_min(self):
        return 1.

    @property
    def epsilon(self):
        return 1.

    def __call__(self, x):
        return 0.

    def gradient(self, x):
        return 0.

class Posterior(object):
    """Posterior

    Generic posterior class
    """
    def __init__(self, likelihood, prior, beta=1.):

        self.beta       = float(beta)
        self.likelihood = likelihood
        self.prior      = prior
        self.update     = True
        
    @property
    def params(self):
        return self.likelihood.params

class PosteriorX(Posterior):

    def __init__(self, likelihood, pi=LJPotential()):

        super(PosteriorX, self).__init__(likelihood, pi)

        pi.params = self.params.theta

        self.sampler = HMC(self)
        
    def __call__(self, x):
        return self.beta * self.likelihood.energy(x) + self.prior(x)

    def gradient(self, x):
        return self.beta * self.likelihood.gradient(x) + self.prior.gradient(x)

    def sample(self):

        if not self.update: return
            
        self.params.X = self.sampler.run(self.params.X.copy())
                
class PosteriorZ(Posterior):

    def __init__(self, likelihood):

        super(PosteriorZ, self).__init__(likelihood, None)

    def sample(self):

        if not self.update: return

        P = self.likelihood.probs

        self.params.Z = np.array([np.random.multinomial(1,p/p.sum()) for p in P])
        
class PosteriorS(Posterior):

    def __init__(self, likelihood):

        super(PosteriorS, self).__init__(likelihood, None)

    def sample(self):

        if not self.update: return

        b = 1e-1 + 0.5 * self.likelihood.chi2
        a = 1e-1 + 0.5 * np.prod(self.likelihood.data.shape)

        self.params.s = (b / np.random.gamma(a)) ** 0.5

class PosteriorTheta(Posterior):

    def __init__(self, likelihood, pi=LJPotential()):

        super(PosteriorTheta, self).__init__(likelihood, pi)
        self._distances = None
        
    def update_distances(self):
        self._distances = calc_distances(self.params.X)

    def invalidate_distances(self):        
        self._distances = None

    def calc_A(self):

        X = self.params.X
        F = np.array([f.gradient(X, self._distances).flatten() for f in self.prior.features])

        return np.dot(F,F.T)

    def calc_b(self):

        X = self.params.X

        return np.array([f.laplacian(X, self._distances) for f in self.prior.features])

    def sample(self):

        if not self.update: return

        self.update_distances()

        A = self.calc_A()
        b = self.calc_b()

        theta = np.dot(np.linalg.pinv(A), b)

        self.prior.params = self.params.theta = theta
        
