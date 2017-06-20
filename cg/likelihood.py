import numpy as np

from csb.bio.utils import distance_matrix
from csb.numeric import log_sum_exp

from scipy.spatial import cKDTree
from scipy.ndimage.measurements import sum as partial_sum

class Likelihood(object):

    @property
    def labels(self):
        return self.params.Z.argmax(1)

    def __init__(self, data, params):
        """
        Parameters
        ----------
        data : numpy array
          coordinate array (i.e. N x 3 matrix)

        params : numpy array
          fitting parameters
        """
        self.data   = data
        self.params = params

        self.invalidate_distances()
        self.invalidate_stats()

    def __call__(self):
        """
        Evaluate between fine- and coarse-grained atoms
        """
        return - 0.5 * (self.chi2 / self.params.s**2 + \
                        np.prod(self.data.shape) * np.log(self.params.s**2))

    @property
    def distances(self):
        if self._distances is None:
            self._distances = distance_matrix(self.data, self.params.X)

        return self._distances

    def invalidate_distances(self):
        self._distances = None

    @property
    def N(self):
        if not self.is_valid: self.update_stats()
        return self._N

    @property
    def mu(self):
        if not self.is_valid: self.update_stats()
        return self._mu

    def update_stats(self):

        self._N = self.params.Z.sum(0) + 1e-10
        self._mu = (np.dot(self.params.Z.T, self.data).T / self._N).T

    def invalidate_stats(self):
        self._N, self._mu = None, None

    @property
    def is_valid(self):
        return self._N is not None and self._mu is not None

    @property
    def chi2(self):
        return np.sum(self.params.Z * self.distances**2)
    
    @property
    def probs(self):
        """
        Soft assignments between points and coarse grained sites. 
        """
        log_prob  = -0.5 * self.distances.T**2 / self.params.s**2
        log_prob -= log_sum_exp(log_prob, 0)

        return np.exp(log_prob.T)

    def energy(self, X=None):
        X = X if X is not None else self.params.X
        return 0.5 * np.sum(np.dot(self.N, (X-self.mu)**2)) / self.params.s**2

    def gradient(self, X=None):
        X = X if X is not None else self.params.X
        return ((X-self.mu).T * self.N).T / self.params.s**2

class KDLikelihood(Likelihood):
    """KDLikelihood
    
    Approximate likelihood based on KD-trees
    """
    @property
    def labels(self):
        return np.array([self.neighbors[i,j] for i,j in enumerate(self.params.Z.argmax(1))])
        
    def __init__(self, data, params, k=10):

        super(KDLikelihood, self).__init__(data, params)

        self.k = int(k)
        self.params._Z = np.zeros((len(data),k),'i')
        
    def __call__(self):

        return - 0.5 * self.chi2 / self.params.s**2 \
               - 1.5 * self.N.sum() * np.log(self.params.s**2)

    def invalidate_distances(self):
        self._tree = cKDTree(self.params.X)
        self._distances, self._neighbors = None, None

    @property
    def distances(self):

        if self._distances is None:
            self._distances, self._neighbors = self._tree.query(self.data, self.k)

        return self._distances

    @property
    def neighbors(self):

        if self._distances is None:
            distances = self.distances

        return self._neighbors

    def update_stats(self):

        distances = self.distances
        neighbors = self.neighbors.astype('i')

        Z  = self.params.Z.astype('d')
        mu = np.zeros(list(reversed(self.params.X.shape)))
        K  = np.arange(self.params.K)
        
        for k in range(self.k):
            for d in range(self.params.X.shape[1]):
                mu[d] += partial_sum(self.data[:,d] * Z[:,k], neighbors[:,k], K)

        N = partial_sum(Z.flatten(), neighbors.flatten(), K) + 1e-10

        self._mu = (mu / N).T
        self._N  = N

