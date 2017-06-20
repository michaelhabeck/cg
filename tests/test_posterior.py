import cg
import numpy as np

from csb.io import load
from csb.bio.utils import rmsd, radius_of_gyration

x = cg.load_example(['4ake','1oelA','1tyq'][2])
N = len(x)
K = 500
k = 20

p_Z, p_s, p_X, p_theta = cg.setup_posterior(x, K, k)

L = p_Z.likelihood
params = L.params

p_X.sampler.dt = 1e-5
p_X.sampler.T  = 20
dt_max = 1e-2

p_theta.sample()

params.s = 250. / K

X     = [params.X.copy()]
s     = [params.s]
theta = [params.theta]
r_min = [p_X.prior.r_min]

output = 'it={0:d}: K={1:d}, rmsd={2:.2f}, s={3:.2f}, r_min={4:.2f}, eps={5:.2f}, dt={6:.2e}, ' + \
         '#{{unassigned}}={7:d}, Rg={8:.2f}'

for i in range(1000):

    ## gibbs sampling

    p_Z.sample()
    p_s.sample()
    p_X.sample()
    p_theta.sample()

    if i and not i % 1:
        print output.format(i, K, rmsd(X[-1],params.X), params.s, p_X.prior.r_min,
                            p_X.prior.epsilon, p_X.sampler.dt, params.Z.sum(0).min(),
                            radius_of_gyration(params.X))

    if p_X.sampler.dt > dt_max:
        p_X.sampler.dt = dt_max
        p_X.sampler.adapt_dt = False

    X     += [params.X.copy()]
    s     += [params.s]
    theta += [params.theta.copy()]
    r_min += [p_X.prior.r_min]

r_min = np.array(r_min)

if False:

    from features import rdf

    Y = [kmeans2(x,K)[0] for _ in range(100)]

    a = rdf(np.array(Y),bins=70,r_max=25.)
    b = rdf(np.array(X[-len(Y):]),bins=70,r_max=25.)

    fig = figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.plot(a[0],(a[1]),lw=3,color='r',alpha=0.5,label='K-means')
    ax.plot(b[0],(b[1]),lw=3,color='k',alpha=0.5,label='Bayes')
    ax.axvline(np.mean(r_min[-10]),ls='--',lw=3,color='k',label=r'$2 \times R_{\min}$')
    ax.set_xlabel(r'distance $R_{kl}$')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(a[0],-np.log(a[1]/a[1].sum())-4,lw=3,color='r',alpha=0.5,label='K-means')
    ax.plot(b[0],-np.log(b[1]/b[1].sum())-4,lw=3,color='k',alpha=0.5,label='Bayes')
    ax.axvline(np.mean(r_min[-10]),ls='--',lw=3,color='k',label=r'$2 \times R_{\min}$')
    ax.set_xlabel(r'distance $R_{kl}$')
    ax.legend()
    fig.tight_layout()

    if False:
        fig.savefig('../latex/fig_rdf.pdf', bbox_inches='tight')

    r = np.linspace(7., 25., 10000)
    E = p_X.prior.profile(r)

    pmf = -np.log(b[1])
    pmf-= (pmf.min() - E.min())

    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(r, E,lw=3,color='r',alpha=0.5, label='CG potential')
    ax.plot(b[0],pmf,lw=3,color='k',alpha=0.5,label='PMF')
    ax.axhline(0.,ls='--',lw=2,color='k')
    ax.set_ylim(-2., 4)
    ax.set_xlabel(r'distance $R_{kl}$')
    ax.legend()

    if False:
        fig.savefig('../latex/fig_lj.pdf', bbox_inches='tight')

    ## compute number of nearest neighbors

    kdtree = KDTree()

if False:

    K = sorted(results.keys())
    s = [np.mean(results[k][1][-100:]) for k in K]
    r_min = [np.mean([x for x in results[k][2][-100:] if not np.isnan(x)]) for k in K]

    fig = figure(figsize=(12,4))
    ax = fig.add_subplot(131)
    ax.scatter(K, r_min,s=80,alpha=0.7,color='k')
    ax.plot(K,r_min,lw=2,ls='--',color='k')
    ax.set_xlabel(r'$K$')
    ax.set_ylabel(r'$r_{\min}$')

    ax = fig.add_subplot(132)
    ax.scatter(K, s,s=80,alpha=0.7,color='k')
    ax.plot(K, s,lw=2,ls='--',color='k')
    ax.set_ylabel(r'$s$')
    ax.set_xlabel(r'$K$')

    ax = fig.add_subplot(133)
    ax.scatter(s, r_min,s=80,alpha=0.7,color='k')
    ax.plot(s,r_min,lw=2,ls='--',color='k')
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$r_{\min}$')

    fig.tight_layout()

    if False:
        fig.savefig('../latex/fig_radii_1oel.pdf', bbox_inches='tight')

    if False:

        from features import rdf

        fig = figure()

        for i, K in enumerate([25, 100, 200, 300],1):
            ax = fig.add_subplot(5,1,i)
            r, g = rdf(results[K][0],bins=50,r_max=25.)
            ax.plot(r,-np.log(g+1e-10),label=str(K))
            ax.set_xlim(1., 25.)
            ax.legend(loc=2)
        ax = fig.add_subplot(5,1,i+1)
        r, g = rdf(np.array([r['CA'].vector for r in chain if r.has_structure]),bins=25,r_max=25.)
        ax.plot(r,-np.log(g+1e-10),label='fine')
        ax.set_xlim(1., 25.)
        ax.legend(loc=2)
