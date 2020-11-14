## notes
## yp is previous (n-1)
## h is step size
## a is gamma
## o is omega
import numpy as np
import math
def midp(x, y, yp, h, a, o):
    ## return next y
    return yp - 2*f(x, y, a, o)*h
## needed fo start midpoint
def eulers(x, y, a, o, h):
    return y + h*f(x, y, a, o)
## actual solution
def x(t, a, b, x0):
    return x0*np.exp(-a*t)*np.cos(b*t)
def f(x, y, a ,o):
    return -2*a*x-o*o*y
## helper function
def impMid(x, y, a, o, h):
    k = f(x, y, a, o)
    return y + f(x+.5*h, y+.5*h*k, a, o)*h
def midPoint(t, to, xo, xp0, stepsize, a, o):
    y = [xo]
    print(eulers(to,xo,a,o,stepsize))
    print(x(stepsize, a, np.sqrt(3), xo))
    y.append(eulers(to,xo,a,o,stepsize))
    steps = int((t - to)/stepsize)
    i = 0
    for i in range(0, steps+1):
        y.append(y[i] + 2*f(to + stepsize*(i+1), y[i+1], a, o)*stepsize)
    print(y[-1])
    return
def implicitMidpoint(t, to, xo, xp0, stepsize, a, o):
    y = [xo]
    steps = int((t - to)/stepsize)
    for i in range(0, steps):
        k = f(to + i*stepsize, y[i], a, o)
        y.append(y[i] + f(to+.5*stepsize, y[i]+.5*stepsize*k, a, o)*stepsize)
    print(y[-1])
    return
if __name__ == "__main__":
    midPoint(5, 0, 1, 1, .0000001, 0, 2)
    b = np.sqrt(4)
    print(x(5, 0, b, 1))
    implicitMidpoint(5, 0, 1, 1, .0000001, 0, 2)