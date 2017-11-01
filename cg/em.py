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

def map2cloud(emmap, cutoff):

    origin  = emmap.origin
    spacing = emmap.spacing
    shape   = emmap.shape
    rho     = emmap.data

    mask    = rho > cutoff
    axes    = [origin[i] + np.arange(0, shape[i]) * spacing[i]
               for i in range(rho.ndim)]
    grid    = np.meshgrid(*axes)
    coords  = np.array([x[mask] for x in grid]).T

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
    
