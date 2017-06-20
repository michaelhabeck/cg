import time
import contextlib
import numpy as np

from scipy.spatial.distance import squareform

from csb.bio.io import StructureParser
from csb.bio.utils import distance_matrix

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
    print '{0} took {1}'.format(desc, format_time(dt))

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

def calc_distances(coords, sep=None):
    """
    Calculate the upper diagonal of a distance matrix
    """
    d = squareform(distance_matrix(coords), checks=False)

    if sep is not None:
        m = 1 - np.sum([np.eye(len(coords),k=k).astype('i')
                        for k in range(1,sep+1)],0)
        m = squareform(m,checks=False)
        d = np.compress(m, d)

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

    d = np.concatenate(map(calc_distances, coords), 0)    
    if r_max is not None: d = d[d<r_max]
        
    g, bins = np.histogram(d, bins=bins)
    r = 0.5 * (bins[1:]+bins[:-1])

    return r, g/r**2

def calc_LJ_params(theta):

    mask   = np.logical_and(theta[:,0]<0, theta[:,1]>0)
    #if not mask.sum(): raise
    theta  = np.compress(mask,theta,0)
    r_vwd  = (-theta[:,1]/theta[:,0])**(1/6.)
    r_vwd *= 2**(1/6.) * 0.5

    eps = theta[:,0]**2 / theta[:,1] / 4.

    return r_vwd, eps

def calc_lambda(r_min, eps):

    lambda1 = - 128. * eps * r_min**6
    lambda2 = lambda1**2 / 4 / eps

    return lambda1, lambda2

