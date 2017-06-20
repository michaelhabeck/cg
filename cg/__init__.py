"""
Bayesian coarse-graining of large biomolecular structures
"""
import numpy as np

from csb.bio.utils import radius_of_gyration

from .features import DistanceFeature, DistancePotential, LJPotential, PotentialEstimator
from .hmc import HMC
from .likelihood import Likelihood, KDLikelihood
from .params import Parameters
from .posterior import PosteriorX, PosteriorS, PosteriorZ, PosteriorTheta
from .utils import calc_distances, load_example, rel_error

try:
    from sklearn.cluster import KMeans
    
    def kmeans(X, K):
        km = KMeans(K).fit(X)
        return km.cluster_centers_

except:
    from scipy.cluster import vq

    def kmeans(X, K):
        return vq.kmeans(X, K)[0]

def setup_posterior(coords, K, k=10, run_kmeans=True):
    """
    Parameters
    ----------
    coords : numpy array
      coordinates of fine-grained atoms

    K : positive int
      number of coarse-grained particles

    k : positive int or None  
      number of nearest neighbors that are considered in the
      assignment of fine to coarse atoms

    run_kmeans : boolean
      flag specifying if K-means should be used for initialization
    """
    coarse = kmeans(coords, K) if run_kmeans else \
             coords[np.random.permutation(len(coords))[:K]]

    params = Parameters(coords, coarse)
    
    L = Likelihood(coords, params) if k is None else \
        KDLikelihood(coords, params, k)

    prior = LJPotential() #LJPotentialFast(K)

    params.attach_callback_setX(L.invalidate_distances)
    params.attach_callback_setZ(L.invalidate_stats)

    p_X = PosteriorX(L, prior)

    p_Z = PosteriorZ(L)
    p_Z.sample()

    p_s = PosteriorS(L)

    p_theta = PosteriorTheta(L, prior)

    return p_Z, p_s, p_X, p_theta

def run_gibbs(p_Z, p_s, p_X, p_theta, n_iter=1e4, verbose=1, dt_max=1e-3, dt_start=1e-5):

    L = p_X.likelihood
    params = p_X.params

    p_X.sampler.dt = dt_start
    p_X.stop = False
    p_theta.sample()

    K = len(params.X)

    X     = [params.X.copy()]
    s     = [params.s]
    theta = [params.theta]
    r_min = [p_X.prior.r_min]
    LL    = [p_X.likelihood()]
    N     = [L.N.copy()]
    Rg    = [radius_of_gyration(params.X)]
    
    for i in range(int(n_iter)):

        p_Z.sample()
        p_s.sample()
        p_X.sample()
        p_theta.sample()

        if i and verbose and not i % verbose:
            print i, K, rmsd(X[-1],params.X), params.s, p_X.prior.r_min, p_X.prior.epsilon, \
                  p_X.sampler.dt, np.mean(L.N==0), params.theta, Rg[-1]

        if p_X.sampler.dt > dt_max:
            p_X.sampler.dt = dt_max
            p_X.sampler.adapt_dt = False
            
        X.append(params.X.copy())
        X = X[-1000:]
        s.append(params.s)
        theta.append(params.theta.copy())
        r_min.append(p_X.prior.r_min)
        N.append(L.N.copy())
        LL += [p_X.likelihood()]
        Rg += [radius_of_gyration(params.X)]
        
        if p_X.stop: break

    results = {'X': X, 's': s, 'theta': theta, 'logL': LL, 'N': N}

    return results

