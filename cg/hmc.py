import numpy as np

from csb.bio.utils import rmsd

class Box(object):
    """
    Bounding box for particles
    """
    def __init__(self, lower, upper):

        self.lower = np.array(lower)
        self.upper = np.array(upper)
        
    def project(self):
        pass

class HMC(object):

    def __init__(self, U, n_steps=100, stepsize=1e-3, box=None, rmsd_max=None):

        self.U  = U
        self.T  = int(n_steps)
        self.dt = float(stepsize)

        self.adapt_dt = True
        self.rmsd_max = rmsd_max
        
    def run(self, q, p=None):

        p = np.random.standard_normal(q.shape)

        p0, q0 = p.copy(), q.copy()

        p -= 0.5 * self.dt * self.U.gradient(q)

        for t in range(self.T-1):
            q += self.dt * p
            p -= self.dt * self.U.gradient(q)

        q += self.dt * p
        p -= 0.5 * self.dt * self.U.gradient(q)

        dH = (np.sum(p**2)-np.sum(p0**2)) / 2 + self.U(q)-self.U(q0)

        accept = False
        if dH <= 0 or np.log(np.random.uniform()) < -dH:
            accept = True

        if self.rmsd_max is not None:
            accept &= rmsd(q,q0) < self.rmsd_max

        if self.adapt_dt:
            self.dt *= 1.02 if accept else 0.98

        return q

