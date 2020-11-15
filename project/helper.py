## notes
## yp is previous (n-1)
## h is step size
## a is gamma
## o is omega
import numpy as np
import math
## needed fo start midpoint
def eulers(t, x, a, o, h):
    return x + h*f(t, x, a, o)
## actual solution
def x(t, a, b, x0):
    return x0*np.exp(-a*t)*np.cos(b*t)
def f(x, y, a ,o):
    return -2*a*x-o*o*y
def midPoint(t, to, xo, xp0, steps, a, o):
    y = [xo]
    h = (t - to)/steps
    y.append(eulers(to,xo,a,o,h))
    i = 0
    for i in range(0, steps-1):
        y.append(y[i] + 2*f(to + h*(i+1), y[i+1], a, o)*h)
    print(y[-1])
    return
def implicitMidpoint(t, to, xo, xp0, steps, a, o):
    h = (t - to)/steps
    y = np.zeros((steps+1,))
    t = np.arange(to, t+h, h)
    y[0] = xp0
    for i in range(1, steps+1):
        k = f(t[i-1], y[i-1], a, o)
        y[i] = y[i-1] + f(t[i-1]+.5*h, y[i-1]+.5*h*k, a, o)*h
    return

if __name__ == "__main__":
    # midPoint(1, 0, 1, 1, .001, 0, 2)
    b = np.sqrt(4-1)
    print(x(1, 1, b, 1))
    implicitMidpoint(1, 0, 1, 1, 1000, 1, 2)
