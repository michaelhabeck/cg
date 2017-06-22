import cg
import numpy as np
import pylab as plt

code = ['4ake','1oelA','1tyq'][2]

x = cg.load_example(code)
K = {'4ake': 50, '1oelA': 150, '1tyq': 500}[code]
k = 20

gibbs   = cg.GibbsSampler(x, K, k, run_kmeans=True)
samples = gibbs.run(1e4)

## show results

burnin  = 1000
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

r, g = cg.utils.rdf(samples['X'][::10],r_max=35., bins=100)

ax[4].plot(r, g/g.max(), lw=3, color='k', alpha=0.7)
ax[4].set_xlabel(r'distance [$\AA$]')
ax[4].set_ylabel(r'RDF')

prior = gibbs.posteriors['X'].prior
prior.params[...] = np.mean(samples['theta'][-1000:],0)

R = np.linspace(prior.r_min*0.85, prior.r_min*3, 100) * 2

ax[5].axhline(0.,ls='--', lw=3, color='k')
ax[5].plot(R, prior.profile(R), lw=5, alpha=0.3, color='k', label='CG potential')
ax[5].plot(r, -np.log(g/g.max())-prior.epsilon, lw=2, ls='--', alpha=0.9, color='k', label='PMF')
ax[5].legend()
ax[5].set_xlim(R.min(), R.max())
ax[5].set_ylim(-1.1*prior.epsilon, 2*prior.epsilon)

fig.tight_layout()
