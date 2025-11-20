import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil, isnan
from cython cimport boundscheck, wraparound, nonecheck, cdivision

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def interp3DtoStd(DTYPE_t[:,:,::1] v, DTYPE_t[::1] vs):

    cdef int X, Y, Z, X1, Y1, Z1
    cdef DTYPE_t vs_x, vs_y, vs_z
    vs_x, vs_y, vs_z = vs[0], vs[1], vs[2]
    X,Y,Z = v.shape[0], v.shape[1], v.shape[2]
    X1,Y1,Z1 = <int>ceil(X*vs_x), <int>ceil(Y*vs_y), <int>ceil(Z*vs_z)
    if X1==int(X*vs_x): 
        X+=1
        v = np.pad(v, ((0,1),(0,0),(0,0)))
    if Y1==int(Y*vs_y): 
        Y+=1
        v = np.pad(v, ((0,0),(0,1),(0,0)))
    if Z1==int(Z*vs_z): 
        Z+=1
        v = np.pad(v, ((0,0),(0,0),(0,1)))

    cdef np.ndarray[DTYPE_t, ndim=3] interpolated = np.zeros((X1, Y1, Z1), dtype=DTYPE)

    _interp3D(&v[0,0,0], &interpolated[0,0,0], X, Y, Z, X1, Y1, Z1, vs_x, vs_y, vs_z)
    return interpolated

def interp3DfromStd(DTYPE_t[:,:,::1] v, DTYPE_t[::1] vs):

    cdef int X, Y, Z, X1, Y1, Z1
    cdef DTYPE_t vs_x, vs_y, vs_z
    vs_x, vs_y, vs_z = 1/vs[0], 1/vs[1], 1/vs[2]
    X,Y,Z = v.shape[0], v.shape[1], v.shape[2]
    X1,Y1,Z1 = <int>ceil(X*vs_x), <int>ceil(Y*vs_y), <int>ceil(Z*vs_z)
    if X1==int(X*vs_x): 
        X+=1
        v = np.pad(v, ((0,1),(0,0),(0,0)))
    if Y1==int(Y*vs_y): 
        Y+=1
        v = np.pad(v, ((0,0),(0,1),(0,0)))
    if Z1==int(Z*vs_z): 
        Z+=1
        v = np.pad(v, ((0,0),(0,0),(0,1)))

    cdef np.ndarray[DTYPE_t, ndim=3] interpolated = np.zeros((X1, Y1, Z1), dtype=DTYPE)

    _interp3D(&v[0,0,0], &interpolated[0,0,0], X, Y, Z, X1, Y1, Z1, vs_x, vs_y, vs_z)
    return interpolated

@cdivision(True)
cdef inline void _interp3D(DTYPE_t *v, DTYPE_t *result, 
               int X, int Y, int Z, int X1, int Y1, int Z1, DTYPE_t vs_x, DTYPE_t vs_y, DTYPE_t vs_z):

    cdef:
        int i, x0, x1, y0, y1, z0, z1, dim
        DTYPE_t x, y, z, xd, yd, zd, c00, c01, c10, c11, c0, c1, c

    dim = X*Y*Z
    dim1 = X1*Y1*Z1

    for i in range(dim1):
        x = i // (Y1*Z1)
        y = (i-Y1*Z1*x) // (Z1)
        z = i-Y1*Z1*x-Z1*y
        x = x+1
        y = y+1
        z = z+1#for matlab style index

        x0 = <int>floor(x/vs_x)
        x1 = x0 + 1
        y0 = <int>floor(y/vs_y)
        y1 = y0 + 1
        z0 = <int>floor(z/vs_z)
        z1 = z0 + 1

        xd = (x-x0*vs_x)/vs_x
        yd = (y-y0*vs_y)/vs_y
        zd = (z-z0*vs_z)/vs_z

        if x0 >= 1 and y0 >= 1 and z0 >= 1 and x1<=X and y1<=Y and z1<=Z:
            c00 = v[Y*Z*(x0-1)+Z*(y0-1)+(z0-1)]*(1-xd) + v[Y*Z*(x1-1)+Z*(y0-1)+(z0-1)]*xd
            c01 = v[Y*Z*(x0-1)+Z*(y0-1)+(z1-1)]*(1-xd) + v[Y*Z*(x1-1)+Z*(y0-1)+(z1-1)]*xd
            c10 = v[Y*Z*(x0-1)+Z*(y1-1)+(z0-1)]*(1-xd) + v[Y*Z*(x1-1)+Z*(y1-1)+(z0-1)]*xd
            c11 = v[Y*Z*(x0-1)+Z*(y1-1)+(z1-1)]*(1-xd) + v[Y*Z*(x1-1)+Z*(y1-1)+(z1-1)]*xd

            c0 = c00*(1-yd) + c10*yd
            c1 = c01*(1-yd) + c11*yd

            c = c0*(1-zd) + c1*zd
        else:
            c = 0
        
        if isnan(c):
            result[i] = 0
        else:
            result[i] = c 