import cg
import numpy

x = cg.load_example('1ake')

with cg.take_time('python'):
    a = cg.utils.calc_distances(x)

with cg.take_time('cython'):
    b = cg.utils.calc_distances_fast(x, return_square=False)
