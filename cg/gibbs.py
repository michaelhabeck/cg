from __future__ import print_function

import numpy as np

from csb.bio.utils import radius_of_gyration, rmsd

from .utils import kmeans
from .params import Parameters
from .features import LJPotentialFast
from .likelihood import Likelihood, KDLikelihood
from .posterior import PosteriorX, PosteriorZ, PosteriorS, PosteriorTheta

class GibbsSampler(object):

    output = 'it={0:d}: K={1:d}, rmsd={2:.2f}, s={3:.2f}, r_min={4:.2f}, eps={5:.2f}, ' + \
             'dt={6:.2e}, #{{unassigned}}={7:d}, Rg={8:.2f}'

    def __init__(self, coords, K, k=10, run_kmeans=True, weights=None):
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

        L = Likelihood(coords, params, weights) if k is None else \
            KDLikelihood(coords, params, k, weights)

        prior = LJPotentialFast()

        params.attach_callback_setX(L.invalidate_distances)
        params.attach_callback_setZ(L.invalidate_stats)

        self.posteriors = {'X': PosteriorX(L, prior),
                           'Z': PosteriorZ(L),
                           's': PosteriorS(L),
                           'theta': PosteriorTheta(L, prior)}

        self.posteriors['Z'].sample()
        self.posteriors['theta'].sample()

    def next(self):

        for name in ('Z', 's', 'X', 'theta'):
            self.posteriors[name].sample()

    def store_sample(self, samples):

        params = self.posteriors['X'].params

        for name in ('X', 's', 'theta'):
            samples[name].append(np.copy(getattr(params, name)))
        samples['X'] = samples['X'][-1000:]

        samples['r_min'].append(self.posteriors['X'].prior.r_min)
        samples['eps'].append(self.posteriors['X'].prior.epsilon)
        samples['LL'].append(self.posteriors['X'].likelihood())

    def report_progress(self, samples):

        info = (len(samples['s']),
                len(samples['X'][-1]),
                rmsd(*samples['X'][-2:]),
                float(samples['s'][-1]),
                samples['r_min'][-1],
                samples['eps'][-1],
                self.posteriors['X'].sampler.dt,
                np.sum(self.posteriors['X'].likelihood.N<1.), 
                radius_of_gyration(samples['X'][-1]))
                
        print(GibbsSampler.output.format(*info))

    def run(self, n_iter=1e4, verbose=1, dt_max=1e-1, dt_start=1e-4):

        self.stop = False

        p_X    = self.posteriors['X']
        params = p_X.params

        p_X.sampler.dt = dt_start

        samples = dict(X=[], s=[], theta=[], r_min=[], eps=[], LL=[])
        self.store_sample(samples)
    
        ## gibbs sampling

        for i in range(int(n_iter)):

            next(self)

            self.store_sample(samples)

            if i and verbose and not i % verbose:
                self.report_progress(samples)

            ## stop stepsize adjustment if stepsize becomes too large

            if p_X.sampler.dt > dt_max:
                p_X.sampler.dt = dt_max
                p_X.sampler.adapt_dt = False

            if self.stop: break

        return samples


