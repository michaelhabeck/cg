"""
Script for coarse-graining an EM map downloaded from EMD
"""
import cg
import numpy as np

from cg import em
from csb.bio.io import mrc

## change accordingly ...

code      = 1290    ## EMD map that will be coarse-grained
n_coarse  = None    ## number of coarse-grained atoms (if None, K=100)
n_nearest = 10      ## nearest-neighbor cutoff for speeding up the sampler
n_gibbs   = 3e3     ## number of Gibbs sampling steps
perc      = 0.9     ## percentage of total density that will be represented by beads

## download map and extract voxels carrying the most significant mass

filename  = em.fetch_emd(code)
reader    = mrc.DensityMapReader(filename)
emmap     = reader.read()
cutoff    = em.find_cutoff(emmap, perc)

print 'Coarse graining EMD-{0} at a cutoff of {2:.3f} ({1:0.1f}% of total density)'.format(code, 100*perc, cutoff)

(coords, 
 weights) = em.map2cloud(emmap, cutoff)
n_coarse  = 100 if n_coarse is None else n_coarse

coords    = coords[weights.argsort()[::-1]]
weights   = np.sort(weights)[::-1]

## setup and run Gibbs sampler

gibbs   = cg.GibbsSampler(coords, n_coarse, n_nearest, run_kmeans=False, weights=weights)
samples = gibbs.run(n_gibbs)

## show results

cg.utils.plot_samples(samples)

