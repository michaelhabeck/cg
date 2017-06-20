import numpy as np

from .utils import calc_distances

from scipy.spatial.distance import squareform

class DistanceFeature(object):
    """DistanceFeature
    
    feature is \sum_{i<j} \|x_i-x_j\|^{-n}
    """
    def __init__(self, n):
        self.n = int(n)

    def __call__(self, x, d=None):
        """
        Evaluate feature
        """
        d = calc_distances(x) if d is None else d

        return np.sum(d**(-self.n))

    def gradient(self, x, D=None):
        """
        Return gradient
        """
        dx = np.array([np.subtract.outer(x[:,d],x[:,d]) for d in range(x.shape[1])])

        D  = calc_distances(x) if D is None else D
        D  = D**(-self.n-2)
        D  = squareform(D)
        D += np.eye(len(x))

        g = - self.n * dx * D

        return g.sum(-1).T

    def laplacian(self, x, d=None):
        """
        Laplacian of distance feature
        """
        d = calc_distances(x) if d is None else d

        return 2 * self.n * (self.n-1) * np.sum(d**(-self.n-2))

    def hessian(self, x, d=None):
        """
        Computes Hessian matrix
        """
        d = calc_distances(x) if d is None else d
        if d.ndim == 1: d = squareform(d)

        H = np.zeros((3*len(x), 3*len(x)))
        n = self.n
        
        for i in range(len(x)):
            for j in range(len(x)):

                if j == i: continue

                dx = x[i]-x[j]
                r  = d[i,j]
                h  = n / r**(n+4) * ((n+2) * np.multiply.outer(dx,dx) - np.eye(3) * r**2)

                H[3*i:3*(i+1), 3*j:3*(j+1)]  = -h 
                H[3*i:3*(i+1), 3*i:3*(i+1)] +=  h

        return H

    def profile(self, r):
        
        return r**(-self.n)
    
class DistancePotential(object):
    """DistancePotential

    Distance-dependent pairwise potential that is a linear combination of power-law
    decays
    """
    def __init__(self, *orders):

        self.features   = [DistanceFeature(n) for n in orders]
        self.params     = np.ones(len(orders))

        ## distance cache

        self._distances = None
        
    @property
    def K(self):
        """
        Number of features
        """
        return len(self.params)

    @property
    def n(self):
        return np.array([f.n for f in self.features])

    def f(self, x, d=None):
        """
        Returns the vector of all evaluated features
        """
        return np.array([f(x, d=None) for f in self.features])

    @property
    def distances(self):
        """
        Distance cache
        """
        return self._distances

    def update_distances(self, x):
        """
        Update distance cache
        """
        self._distances = calc_distances(x)
        
    def __call__(self, x):
        """
        Evaluate pairwise potential
        """
        return np.dot(self.params, self.f(x, self._distances))

    def gradient(self, x):

        d = self._distances
        if d is not None and np.ndim(d) == 1: d = squareform(d)

        return np.sum([self.params[k] * self.features[k].gradient(x,d)
                       for k in range(self.K)],0)

    def laplacian(self, x):

        d = self._distances

        return np.sum([self.params[k] * self.features[k].laplacian(x,d)
                       for k in range(self.K)],0) 

    def hessian(self, x):

        d = self._distances

        return np.sum([self.params[k] * self.features[k].hessian(x,d)
                       for k in range(self.K)],0) 

    def profile(self, r):

        return np.sum([self.params[k] * self.features[k].profile(r)
                       for k in range(self.K)],0)
 
class LJPotential(DistancePotential):
    """LJPotential

    Lennard-Jones potential
    """
    def __init__(self):
        super(LJPotential, self).__init__(6, 12)

    @property
    def epsilon(self):
        return -self.params[0] / (4.0*self.sigma**6.0)

    @property
    def sigma(self):
        return (-self.params[1] / self.params[0])**(1/6.0)

    @property
    def r_min(self):
        return 2**(1/6.) * self.sigma

class LJPotentialFast(LJPotential):

    def __call__(self, x):

        from .lj import energy

        eps = self.params[0]**2 / self.params[1]
        sig = -self.params[1] / self.params[0]

        return energy(np.ascontiguousarray(x.reshape(-1,)), float(sig), float(eps))

    def gradient(self, x):

        from .lj import gradient

        eps = self.params[0]**2 / self.params[1]
        sig = - self.params[1] / self.params[0]

        g = np.zeros(np.prod(x.shape))

        gradient(np.ascontiguousarray(x.reshape(-1,)), g, float(sig), float(eps))

        return g.reshape(x.shape)

class PotentialEstimator(object):

    def __init__(self, features):

        self.features   = features
        self._distances = None

    def update_distances(self, x):
        self._distances = calc_distances(x)

    def invalidate_distances(self):
        self._distances = None

    @property
    def distances(self):
        return self._distances

    def calc_A(self, x):
        F = np.array([f.gradient(x,self._distances).flatten() for f in self.features])
        return np.dot(F,F.T)

    def calc_b(self, x):
        return np.array([f.laplacian(x,self._distances) for f in self.features])

    def __call__(self, x):

        self.update_distances(x)

        A = self.calc_A(x)
        b = self.calc_b(x)

        params = np.dot(np.linalg.pinv(A), b)

        return params

    
