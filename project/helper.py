## notes
## yp is previous (n-1)
## h is step size
## a is gamma
## o is omega
import numpy as np
def midp(x, y, yp, f, h):
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
    return y + f(x+.5*h, y+.5*h*k, a, o)*h
def midPoint(t, to, xo, xp0, steps, a, o):
    stepsize = (t - to)/steps
    y = [xo]
    for i in steps:
        y.append(yp - 2*f(x, y)*stepsize)
    return