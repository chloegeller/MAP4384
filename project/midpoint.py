## notes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as fs

#####################################
 #                                 #
 #        Explicit Midpoint        #
 #                                 #
#####################################

def midPoint():
    y = np.zeros(steps+1)
    xl = np.zeros(steps+1)
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
    return xl

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
    
    global x
    x = np.array([x0, xp0])
    traj=x

    for i in range(1, steps+1):
        x_new = fs(IMP,x) 
        traj=np.vstack([traj,x_new])
        x=x_new
    
    return traj

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
    
    global xn
    xn = np.array([x0, xp0])
    traj=xn

    for i in range(1, steps+1):
        x_new = fs(NSIMP,xn) 
        traj=np.vstack([traj,x_new])
        xn=x_new
    
    return traj

#####################################
 #                                 #
 #          Exact solution         #
 #                                 #
#####################################

def exact():
    y = np.zeros(steps+1)
    b = np.sqrt((omega**2)-(gamma**2))
    for i in range(0, (steps+1)):
        y[i] = x0*np.exp(-gamma*t[i])*np.cos(b*t[i])
    return y

#####################################
 #                                 #
 #          Relative error         #
 #                                 #
#####################################

def relativeError(exact, aprox, steps):
    rel = np.zeros((steps+1,))
    for i in range(0, steps+1):
        rel[i] = abs(exact[i]-aprox[i])/abs(exact[i])
    return rel

if __name__ == "__main__":
    global t0, x0, steps, h, gamma, omega, xp0, t
    t0 = 0
    tf = 1
    x0 = 1
    steps = 10
    h = (tf - t0)/steps
    gamma = .1
    omega = 1
    xp0 = x0*gamma
    t = np.arange(t0, tf + h, h)
    xm = midPoint()
    xi = implicitMidpoint()
    xn = nsImplicitMid()
    xe = exact()
    plt.plot(t, xe, 'g', t, xn[:,0], 'b', t, xi[:,0], 'r', t, xm, 'y')
    plt.show()
