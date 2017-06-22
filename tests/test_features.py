"""
Testing gradient, Laplacian and Hessian as well as parameter estimation for the
Lennard-Jones potential
"""
import cg
import numpy as np
import pylab as plt

from scipy.optimize import approx_fprime

coords = cg.load_example('1ake')

K = 100
x = coords[np.random.permutation(len(coords))[:K]]
f = cg.LJPotential()

eps = 1e-5

a = f.gradient(x).flatten()
g = lambda x: f(x.reshape(-1,3))

print 'check gradient'
print '-' * 30

for eps in np.logspace(-3, -7, 5):
    b = approx_fprime(x.flatten(), g, eps)
    print 'eps={0:1.0e}, rel. error={1:.2e}'.format(eps, cg.rel_error(a, b).max())

print '-' * 30

a = f.laplacian(x)
g = f.gradient(x)

print 'check laplacian'
print '-' * 30

msg = 'eps={0:1.0e}, analytical={1:.5e}, numerical={2:.5e}, numerial={3:.5e}, ' + \
      'error={4:.2e}, error={5:.2e}'

for eps in [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-6, 1e-7]:

    b = 0.
    c = 0.

    for k in range(x.shape[0]):
        for d in range(x.shape[1]):

            x_plus  = x.copy()
            x_minus = x.copy()

            x_plus[k,d]  += eps
            x_minus[k,d] -= eps

            c += (f.gradient(x_plus)-g)[k,d] / eps
            b += (f(x_plus) - 2*f(x) + f(x_minus)) / eps**2

    print msg.format(eps, a, b, c, abs(a-b), abs(a-c))

print '-' * 30

estimator = cg.PotentialEstimator(f.features)

A = estimator.calc_A(x)
b = estimator.calc_b(x)

f.params = estimator(x)

print 'estimate LJ potential'
print '-' * 30
print 'params', f.params
print 'LJ parameters: eps={0:.2f}, sigma={1:.2f}, r_min={2:.2f}'.format(f.epsilon, f.sigma, f.r_min)
print '-' * 30

## plot potential

r = np.linspace(f.r_min*0.85, f.r_min*3, 100) * 2

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.axhline(0.,ls='--', lw=3, color='k')
ax.plot(r, f.profile(r), lw=5, alpha=0.3, color='k')

## check Hessian

import numdifftools as diff

n = 15
x = np.random.random((n,3))
f = cg.DistanceFeature(6)
g = lambda y : f(y.reshape(-1,3))
A = f.hessian(x)
h = diff.Hessian(g)
B = h(x.copy().flatten())

print 'check hessian'
print '-' * 30
print 'rel. error={0:.2e}, corr-coef={1:.2f} %'.format(
    cg.rel_error(A, B).max(), 100*np.corrcoef(A.flatten(), B.flatten())[0,1])
