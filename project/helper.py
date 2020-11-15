## notes
## yp is previous (n-1)
## h is step size
## g is gamma
## o is omega
import numpy as np
import math
import matplotlib.pyplot as plt
## needed fo start midpoint
def eulers(t, x, xp0, g, o, h):
    return xp0 + h*f(t, x, g, o)
## actual solution
def x(t, g, b, x0):
    return x0*np.exp(-g*t)*np.cos(b*t)
def f(x, y, g ,o):
    return -2*g*x-o*o*y
def midPoint(t, IV):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros((steps+1,))
    t = np.arange(t0, t+h, h)
    y[0] = eulers(t0, x0, xp0, g, o, h)
    for i in range(1, steps):
        y[i+1] = y[i-1] + 2*f(t[i], y[i], g, o)*h
    return t, y
def implicitMidpoint(t, IV):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros((steps+1,))
    t = np.arange(t0, t+h, h)
    y[0] = xp0
    for i in range(1, (steps+1)):
        k = f(t[i-1], y[i-1], g, o)
        y[i] = y[i-1] + f(t[i-1]+.5*h, y[i-1]+.5*h*k, g, o)*h
    return t, y
def exact(t, IV):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros((steps+1,))
    t = np.arange(t0, t+h, h)
    b = np.sqrt((o*o)-(g*g))
    for i in range(0, (steps+1)):
        y[i] = x0*np.exp(-g*t[i])*np.cos(b*t[i])
    return t, y
if __name__ == "__main__":
    # IV = (t0, x0, xp0, steps, g, o)
    IV =(0, 1, 1, 2000, .1, .5)
    tm, ym = midPoint(1, IV)
    ti, yi = implicitMidpoint(3, IV)
    tx, yx = exact(3,IV)
    plt.plot(tx, yx, 'r', ti, yi, 'g')
    plt.show()
