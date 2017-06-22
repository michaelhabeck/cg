import cg
import numpy as np
import pylab as plt

x = cg.load_example(['4ake','1oelA','1tyq'][2])
K = 500
k = 20

gibbs   = cg.GibbsSampler(x, K, k, run_kmeans=not False)
samples = gibbs.run(1e4)

## show results

kw_hist = dict(normed=True, bins=100, alpha=0.7, color='k')
fig, ax = plt.subplots(2,2,figsize=(8,8))
ax = ax.flat

ax[0].hist(samples['s'][burnin:], **kw_hist)
ax[0].set_xlabel(r'standard deviation $s$')

ax[1].hist(samples['r_min'][burnin:], **kw_hist)
ax[1].set_xlabel(r'bead radius $R_\mathrm{CG}$')

ax[2].hist(samples['eps'][burnin:], **kw_hist)
ax[2].set_xlabel(r'$\epsilon$')

r, g = cg.utils.rdf(samples['X'][::10],r_max=35., bins=100)

ax[3].plot(r, g, lw=3, color='k', alpha=0.7)
ax[3].set_xlabel(r'distance [$\AA$]')
ax[3].set_ylabel(r'RDF')

