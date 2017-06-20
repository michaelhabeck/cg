import cg
import numpy as np

x = cg.load_example('4ake')
a = cg.LJPotential()
b = cg.LJPotentialFast()

a.params[...] = [-np.random.random(), np.random.random()]
b.params[...] = a.params

with cg.take_time('python'):
    g = a.gradient(x)
    print a(x)

with cg.take_time('cython'):
    f = b.gradient(x)    
    print b(x)

print cg.rel_error(g, f).max()
