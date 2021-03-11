from __future__ import print_function

import time
import contextlib
import numpy as np
import pylab as plt

from scipy.spatial.distance import squareform

from csb.bio.io import StructureParser

try:
    from sklearn.cluster import KMeans
    
    def kmeans(X, K):
        km = KMeans(K).fit(X)
        return km.cluster_centers_
except:
    from scipy.cluster import vq

    def kmeans(X, K):
        return vq.kmeans(X, K)[0]

def format_time(t):

    units = [(1.,'s'),(1e-3,'ms'),(1e-6,'us'),(1e-9,'ns')]
    for scale, unit in units:
        if t > scale or t==0: break
        
    return '{0:.1f} {1}'.format(t/scale, unit)

@contextlib.contextmanager
def take_time(desc):
    t0 = time.clock()
    yield
    dt = time.clock() - t0
    print('{0} took {1}'.format(desc, format_time(dt)))

def load_coords(pdbfile, chainids=None):

    struct = StructureParser(pdbfile).parse()

    if chainids is None:
        coords = struct.get_coordinates()
    else:
        coords = np.concatenate([struct[chainid].get_coordinates() for chainid
                                 in chainids],0)
    return coords

def load_example(code='1ake', path='./data/{}.pdb'):
    return load_coords(path.format(code))

def rel_error(a, b):
    return np.fabs(a-b) / (np.fabs(a) + np.fabs(b) + 1e-300)

def calc_distances(coords):
    """
    Calculate the upper diagonal of a distance matrix
    """
    from .lj import squared_distances

    x = np.ascontiguousarray(coords.flatten())
    d = np.zeros(len(coords) * (len(coords)-1) / 2)

    squared_distances(x, d)

    return d

def rdf(coords, bins=100, r_max=None):
    """
    Radial distribution function

    Parameters
    ----------

    coords :
      list of coordinate arrays

    bins : int or numpy array
      distance bins

    r_max : positive float or None
      maximum distance
    """
    if np.ndim(coords) == 2: coords = [coords]

    d = np.sqrt(np.concatenate(map(calc_distances, coords), 0))
    if r_max is not None: d = d[d<r_max]
        
    g, bins = np.histogram(d, bins=bins)
    r = 0.5 * (bins[1:]+bins[:-1])

    return r, g/r**2

def plot_samples(samples, burnin=1000, r_max=35):

    from .features import LJPotential

    kw_hist = dict(normed=True, bins=100, alpha=0.7, color='k')
    fig, ax = plt.subplots(2,3,figsize=(12,8))

    ax = ax.flat

    names   = ('s', 'r_min', 'eps')
    xlabels = (r'standard deviation $s$',
               r'bead radius $R_\mathrm{CG}$',
               r'$\epsilon$')

    for i, name in enumerate(names):

        x = np.array(samples[name][burnin:])
        x = x[~np.isnan(x)]

        ax[i].hist(x, **kw_hist)
        ax[i].set_xlabel(xlabels[i])

    ax[3].scatter(*np.transpose(samples['theta'][burnin:]), alpha=0.2, s=20, color='k')
    ax[3].set_xlabel(r'$\lambda_1$')
    ax[3].set_ylabel(r'$\lambda_2$')

    r, g = rdf(samples['X'][::10],r_max=r_max, bins=100)

    ax[4].plot(r, g/g.max(), lw=3, color='k', alpha=0.7)
    ax[4].set_xlabel(r'distance [$\AA$]')
    ax[4].set_ylabel(r'RDF')

    prior = LJPotential()
    prior.params[...] = np.mean(samples['theta'][-1000:],0)

    R = np.linspace(prior.r_min*0.85, prior.r_min*3, 100) * 2

    ax[5].axhline(0.,ls='--', lw=3, color='k')
    ax[5].plot(R, prior.profile(R), lw=5, alpha=0.3, color='k', label='CG potential')
    ax[5].plot(r, -np.log(g/g.max())-prior.epsilon, lw=2, ls='--', alpha=0.9, color='k', label='PMF')
    ax[5].legend()
    ax[5].set_xlim(R.min(), R.max())
    ax[5].set_ylim(-1.1*prior.epsilon, 2*prior.epsilon)

    fig.tight_layout()

    return fig
