"""
Class for storing all inference parameters. Supports callbacks whenever
X and Z change.
"""
import numpy as np

class Parameters(object):

    def __init__(self, x, X):
        """
        Parameters
        ----------

        x : numpy array
          coordinates of fine-grained atoms

        X : numpy array
          coordinates of coarse-grained atoms
        """
        self.N = len(x)

        self._X = X
        self._Z = np.zeros((self.N,self.K),'i')
        
        self.s = 1.
        self.theta = np.array([-1,1.])

        self._callbacks_setX = []
        self._callbacks_setZ = []

        self.randomize_Z()

    def attach_callback_setX(self, f):
        self._callbacks_setX.append(f)

    def attach_callback_setZ(self, f):
        self._callbacks_setZ.append(f)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X[:,:] = X
        for f in self._callbacks_setX: f()

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z[:,:] = Z
        for f in self._callbacks_setZ: f()

    @property
    def K(self):
        return len(self._X)

    def randomize_Z(self):

        z = np.random.randint(0,self.K,self.N)
        self.Z = np.equal.outer(z,np.arange(self.K)).astype('i')
        
