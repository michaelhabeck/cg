import cg
import numpy as np

from csb.io import load
from csb.bio.utils import rmsd, radius_of_gyration

x = cg.load_example(['4ake','1oelA','1tyq'][2])
N = len(x)
K = 500
k = 20

p_Z, p_s, p_X, p_theta = cg.setup_posterior(x, K, k, run_kmeans=not False)

L = p_Z.likelihood
params = L.params

p_X.sampler.T  = 50
p_X.sampler.dt = 1e-7
dt_max = 1e-3

p_theta.sample()

params.s = 250. / K

X     = [params.X.copy()]
s     = [params.s]
theta = [params.theta]
r_min = [p_X.prior.r_min]
eps   = [p_X.prior.epsilon]

output = 'it={0:d}: K={1:d}, rmsd={2:.2f}, s={3:.2f}, r_min={4:.2f}, eps={5:.2f}, dt={6:.2e}, ' + \
         '#{{unassigned}}={7:d}, Rg={8:.2f}'

for i in range(10000):

    ## gibbs sampling

    p_Z.sample()
    p_s.sample()
    p_X.sample()
    p_theta.sample()

    if i and not i % 1:
        print output.format(i, K, rmsd(X[-1],params.X), params.s, p_X.prior.r_min,
                            p_X.prior.epsilon, p_X.sampler.dt, np.sum(L.N==0.), 
                            radius_of_gyration(params.X))

    if p_X.sampler.dt > dt_max:
        p_X.sampler.dt = dt_max
        p_X.sampler.adapt_dt = False

    X     += [params.X.copy()]
    s     += [params.s]
    theta += [params.theta.copy()]
    r_min += [p_X.prior.r_min]
    eps   += [p_X.prior.epsilon]
    
    X = X[-1000:]

r_min = np.array(r_min)
eps   = np.array(eps)
mask  = np.logical_and(np.logical_not(np.isnan(r_min)), eps>0)

