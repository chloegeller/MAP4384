## notes
## yp is previous (n-1)
## h is step size
## g is gamma
## o is omega
import numpy as np
import math
## needed fo start midpoint
def eulers(t, x, g, o, h):
    return x + h*f(t, x, g, o)
## actual solution
def x(t, g, b, x0):
    return x0*np.exp(-g*t)*np.cos(b*t)
def f(x, y, g ,o):
    return -2*g*x-o*o*y
def midPoint(t, t0, x0, xp0, steps, g, o):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros((steps+1,))
    t = np.arange(t0, t+h, h)
    y[0] = eulers(t0,x0,g,o,h)
    for i in range(1, steps):
        y[i+1] = y[i-1] + 2*f(t[i], y[i], g, o)*h
    print(y[-1])
    return
def implicitMidpoint(t, t0, x0, xp0, steps, g, o):
    h = (t - t0)/steps
    y = np.zeros((steps+1,))
    t = np.arange(t0, t+h, h)
    y[0] = xp0
    for i in range(1, steps+1):
        k = f(t[i-1], y[i-1], g, o)
        y[i] = y[i-1] + f(t[i-1]+.5*h, y[i-1]+.5*h*k, g, o)*h
    return

if __name__ == "__main__":
    # IV = (t0, x0, xp0, steps, g, o)
    # midPoint(1, 0, 1, 1, .001, 0, 2)
    b = np.sqrt(4-1)
    print(x(1, 1, b, 1))
    implicitMidpoint(1, 0, 1, 1, 1000, 1, 2)
