import numpy
cimport numpy
cimport cython

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t

DTYPE_DOUBLE = numpy.double
ctypedef numpy.double_t DTYPE_DOUBLE_t

DTYPE_LONG = numpy.long
ctypedef numpy.long_t DTYPE_LONG_t

@cython.boundscheck(False)
@cython.wraparound(False)
def squared_distances(double [::1] x, double [::1] d):

    cdef Py_ssize_t i, j, k
    cdef double dx, dy, dz
    cdef int n = len(x) / 3

    k = 0
    
    for i in range(n-1):
        for j in range(i+1, n):

            dx = x[3*i+0] - x[3*j+0]
            dy = x[3*i+1] - x[3*j+1]
            dz = x[3*i+2] - x[3*j+2]

            d[k] = dx*dx + dy*dy + dz*dz

            k += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def energy(double [::1] x, double sigma, double eps):

    cdef Py_ssize_t i, j
    cdef double dx, dy, dz, r2, r6, E
    cdef int n = len(x) / 3

    E = 0.0

    for i in range(n-1):
        for j in range(i+1, n):

            dx = x[3*i+0] - x[3*j+0]
            dy = x[3*i+1] - x[3*j+1]
            dz = x[3*i+2] - x[3*j+2]

            r2 = dx*dx + dy*dy + dz*dz

            r6 = sigma / (r2*r2*r2)
            E += r6 * (r6-1) 

    return eps * E 

@cython.boundscheck(False)
@cython.wraparound(False)
def gradient(double [::1] x, double [::1] g, double sigma, double eps):

    cdef Py_ssize_t i, j
    cdef double dx, dy, dz, r2, r6, E, F
    cdef int n = len(x) / 3

    E = 0.0

    for i in range(n-1):
        for j in range(i+1, n):

            dx = x[3*i+0] - x[3*j+0]
            dy = x[3*i+1] - x[3*j+1]
            dz = x[3*i+2] - x[3*j+2]

            r2 = dx*dx + dy*dy + dz*dz

            r6 = sigma / (r2*r2*r2)
            E += r6 * (r6-1) 
            F  = - 12 * eps * r6 * (r6-0.5) / r2
	  
            g[3*i+0] += dx * F
            g[3*j+0] -= dx * F
            
            g[3*i+1] += dy * F
            g[3*j+1] -= dy * F

            g[3*i+2] += dz * F
            g[3*j+2] -= dz * F

    return eps * E

