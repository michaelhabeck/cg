"""
Script for coarse-graining a PDB entry
"""
import cg
import numpy as np

from csb.bio.io.wwpdb import RemoteStructureProvider

PDB = RemoteStructureProvider('https://files.rcsb.org/download/', '.pdb')

## change accordingly ...

code      = '1tyq'  ## PDB file of structure that shall be coarse-grained
chains    = None    ## ids of selected chains (if desired)
n_coarse  = None    ## number of coarse-grained atoms (if None, K=N/250)
n_nearest = 20      ## nearest-neighbor cutoff for speeding up the sampler
n_gibbs   = 3e3     ## number of Gibbs sampling steps

## download structure and extract atomic coordinates

struct   = PDB.get(code)
chains   = list(struct) if chains is None else chains
coords   = np.concatenate([struct[chainid].get_coordinates() for chainid in chains])
n_coarse = len(coords) / 250 if n_coarse is None else n_coarse

## setup Gibbs sampler

gibbs = cg.GibbsSampler(coords, n_coarse, n_nearest)

## run Gibbs sampler

samples = gibbs.run(n_gibbs)

## show results

fig = cg.utils.plot_samples(samples)
