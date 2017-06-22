import time
import contextlib
import numpy as np

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

