import numpy as np
def midpoint(x, y, yp, f, h):
    ## return next y
    return yp - 2*f(x, y)*h
## needed fo start midpoint
def eulers(x, y, a, o, h):
    return y + h*f(x, y, a, o)
## actual solution
def x(t, a, b, c):
    return c*np.exp(-a*t)*np.cos(b*t)
def f(x, y, a ,o):
    return -2*a*x-o*o*y
## helper function
def impMid(x, y, a, o, h):
    k = f(x, y, a, o)
    return y + f(x+.5*h, y+.5*h*k, a, o)