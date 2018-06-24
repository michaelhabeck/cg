"""
Methods for coarse graining EM maps
"""
import os
import csb.io
import numpy as np

def find_cutoff(emmap, percentage):

    rho  = emmap.data
    prob = np.sort(rho[rho>0])[::-1]
    cum  = np.add.accumulate(prob) / prob.sum()

    return prob[np.sum(cum <= percentage)]

def map2cloud(emmap, cutoff, order=None):

    origin  = emmap.origin
    spacing = emmap.spacing
    shape   = emmap.shape
    rho     = emmap.data

    if order is None: order = range(rho.ndim) 

    rho     = rho.swapaxes(0,2)

    mask    = rho > cutoff
    axes    = [origin[i] + np.arange(shape[i]) * spacing[i]
               for i in range(rho.ndim)]
    grid    = np.meshgrid(*axes)
    coords  = np.array([grid[i][mask] for i in order]).T

    return coords, rho[mask]

def fetch_emd(emd_code, dst='/tmp'):

    path = 'ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-{0}/map'.format(emd_code)
    name = 'emd_{0}'.format(emd_code)

    stream = csb.io.urllib.urlopen(os.path.join(path, name) + '.map.gz')
    out = open(os.path.join(dst, name) + '.map.gz', 'wb')
    out.write(stream.read())
    out.flush()

    os.system('gunzip -f {0}.map.gz'.format(os.path.join(dst, name)))

    return os.path.join(dst, name) + '.map'
    
