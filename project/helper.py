## notes
## yp is previous (n-1)
## h is step size
## g is gamma
## o is omega
import numpy as np
import math
import matplotlib.pyplot as plt

## actual solution
def x(t, g, b, x0):
    return x0*np.exp(-g*t)*np.cos(b*t)

def midPoint(t, IV):
    t0, x0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros(steps+1)
    x = np.zeros(steps+1)
    t = np.arange(t0, t+h, h)
    xp0 = x0*g
    x[0] = x0
    y[0] = xp0
    for i in range(0, steps):
        if i == 0:
            x[i+1] = 2*h*y[i]
            y[i+1] = -4*g*h*y[i] - 2*o*o*h*x[i]
        else:
            x[i+1] = x[i-1] + 2*h*y[i]
            y[i+1] = y[i-1] - 4*g*h*y[i] - 2*o*o*h*x[i]
    return t, x

def implicitMidpoint(t, IV):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros(steps+1)
    t = np.arange(t0, t+h, h)
    y[0] = x0
    for i in range(0, (steps)):
        y[i+1] = (y[i]*(1-g*h)- o*o*(t[i+1]+t[i])/2)/(1+h*g)
    return t, y

def nsImpMidpoint(t, IV):
    t0, x0, xp0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros(steps+1)
    t = np.arange(t0, t+h, h)
    y[0] = x0
    for i in range(0, (steps)):
        y[i+1] = y[i]*np.exp(-g*h) - h*(o*o)*(t[i+1] + np.exp(-g*h)*t[i])*.5
    return t, y

def exact(t, IV):
    t0, x0, steps, g, o = IV
    h = (t - t0)/steps
    y = np.zeros(steps+1)
    t = np.arange(t0, t+h, h)
    b = np.sqrt((o*o)-(g*g))
    for i in range(0, (steps+1)):
        y[i] = x0*np.exp(-g*t[i])*np.cos(b*t[i])
    return t, y

def relativeError(exact, aprox, steps):
    rel = np.zeros((steps+1,))
    for i in range(0, steps+1):
        rel[i] = abs(exact[i]-aprox[i])/abs(exact[i])
    return rel

if __name__ == "__main__":
    # IV = (t0, x0, steps, g, o)
    IV =(0, 1, 100, .0001, 1)
    tm, ym = midPoint(1, IV)
    #ti, yi = implicitMidpoint(1, IV)
    tx, yx = exact(1,IV)
    #tn, yn = nsImpMidpoint(1, IV)
   # rel = relativeError(yx, yi, 10)
    #plt.plot(tm, ym, 'r', ti, yi, 'g', tx, yx, 'b')
    plt.plot(tm, ym, 'r', tx, yx, 'g')
    #plt.plot(tx, rel)
    plt.show()
