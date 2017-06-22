import cg
import numpy as np

from csb.bio.utils import rmsd

x = cg.load_example('4ake')
N = len(x)
K = N / 20
X = cg.kmeans(x, K)
K = len(X)

k = 10

params  = cg.Parameters(x,X)
params2 = cg.Parameters(x,X)
params2._Z = np.zeros((len(x),k),'i')

print params.Z.shape
print params._Z.shape

L  = cg.Likelihood(x, params)
L2 = cg.KDLikelihood(x, params2, k=k)

params.Z[...]  = np.equal.outer(L.distances.argmin(1), np.arange(K))
params2.Z[...] = np.equal.outer(L2.distances.argmin(1), np.arange(k))

print L.chi2, L2.chi2

L.update_stats()
L2.update_stats()

print rmsd(L.mu, L2.mu), np.fabs(L.N-L2.N).max()

