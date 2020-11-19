## notes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as fs

#####################################
 #                                 #
 #        Explicit Midpoint        #
 #                                 #
#####################################

def midPoint(t, IV):
    y = np.zeros(steps+1)
    xl = np.zeros(steps+1)
    t = np.arange(t0, t+h, h)
    xp0 = x0*gamma
    xl[0] = x0
    y[0] = xp0
    for i in range(0, steps):
        if i == 0:
            xl[i+1] = 2*h*y[i]
            y[i+1] = -4*gamma*h*y[i] - 2*omega**2*h*xl[i]
        else:
            xl[i+1] = xl[i-1] + 2*h*y[i]
            y[i+1] = y[i-1] - 4*gamma*h*y[i] - 2*omega**2*h*xl[i]
    return t, xl

#####################################
 #                                 #
 #        Implicit Midpoint        #
 #                                 #
#####################################

def IMP(p):
    X, Y = p
    deriv = np.array([0.,0.])
    deriv[0]= (Y+x[1])/2 - (X-x[0])/h
    deriv[1]= -2*gamma*(Y+x[1])/2 - omega**2*(X+x[0])/2 - (Y-x[1])/h
    return deriv

def implicitMidpoint():
    
    time=np.zeros(1)
    global x
    x = np.array([x0, xp0])
    traj=x

    for i in range(1, steps+1):
        t=i*h
        time=np.hstack([time,t])
        x_new = fs(IMP,x) 
        traj=np.vstack([traj,x_new])
        x=x_new
    
    return time, traj

#####################################
 #                                 #
 #  Non-standard Implicit Midpoint #
 #                                 #
#####################################

def NSIMP(p):
    X, Y = p
    deriv = np.array([0.,0.])
    deriv[0]= (Y+np.exp(-gamma*h)*xn[1])/2 - (X-np.exp(-gamma*h)*xn[0])/h
    deriv[1]= - omega**2*(X+np.exp(-gamma*h)*xn[0])/2 - (Y-np.exp(-gamma*h)*xn[1])/h
    return deriv

def nsImplicitMid():
    
    time=np.zeros(1)
    global xn
    xn = np.array([x0, xp0])
    traj=xn

    for i in range(1, steps+1):
        t=i*h
        time=np.hstack([time,t])
        x_new = fs(NSIMP,xn) 
        traj=np.vstack([traj,x_new])
        xn=x_new
    
    return time, traj

#####################################
 #                                 #
 #          Exact solution         #
 #                                 #
#####################################

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
    global t0
    t0 = 0
    global t
    t = 1
    global x0
    x0 = 1
    global steps
    steps = 10
    global h
    h = (t -t0)/steps
    global gamma
    gamma = .1
    global omega
    omega = 1
    global xp0
    xp0 = x0*gamma

    IV = (t0, x0, steps, gamma, omega)
    #IV =(0, 1, 100, .0001, 1)
    #tm, ym = midPoint(1, IV)
    #ti, xi = implicitMidpoint()
    #print(ti)
    #print(xi)
    tn, xn = nsImplicitMid()
    print(tn)
    print(xn)
    tx, yx = exact(1,IV)
    print(yx)
    #plt.plot(tm, ym, 'r', ti, yi, 'g', tx, yx, 'b')
    #plt.plot(ti, xi, 'r', tx, yx, 'g')
    #plt.plot(tx, rel)
    plt.show()
