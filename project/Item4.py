##############################################################################
## All three method are second order and they do not differ much in terms of 
## computational cost; demostrate this numerically. I expect several table with 
## the results.
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import time as tls
from scipy.optimize import fsolve as fs

# Initial parameters (global varibles)
a, b = 0, 5
omega = 5
gamma = 0.5
x0 = 100.0

# Exact solution function
def func(t):
    beta = np.sqrt(omega**2 - gamma**2)
    return x0*np.exp(-gamma*t)*np.cos(beta*t)

# Right side vectorial fuction 
def dxdt(t,x):
    deriv = np.array([0.,0.])
    deriv[0]=x[1]
    deriv[1]=-2*gamma*x[1]-omega**2*x[0]
    return deriv

# Exact values
Error = np.zeros(5)
N = np.zeros(5)
T = np.zeros(5)
H = np.zeros(5)
#-------------------------------------------------------------------------------
j=0
for n in [100, 250, 500, 1000, 2000]:
    start_seconds = tls.time()

    # step
    h=(b-a)/n

    def EMP(x,t):
        return x + h*dxdt(t + h/2, x + (h/2)*dxdt(t,x))
    
    # Exact values
    t_sol = np.linspace(a, b, n+1)
    x_sol = func(t_sol)
    
    # numerical solution
    Time=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj=x
    
    for i in range(n):
        t = i*h
        Time = np.hstack([Time,t])
        x_new = EMP(x,t)
        traj = np.vstack([traj,x_new])
        x = x_new
    
    Error[j] = np.abs((traj[n,0] - func(Time[n]))/func(Time[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1

print('By Explicit Midpoint Method')
results = [(N[i],H[i],T[i],Error[i]) for i in range(N.size)]
print(tabulate(results, headers=["n","h","Run time ","Rel. Error "], floatfmt=(".0f",".3f",".4f",".4e"),tablefmt="fancy_grid"))
#-------------------------------------------------------------------------------
j=0
for n in [100, 250, 500, 1000]:
    start_seconds = tls.time()

    # step
    h=(b-a)/n

    # Right side vectorial fuction 
    def IMP(p):
        X, Y = p
        zero = np.array([0.,0.])
        zero[0]= (Y+x[1])/2 - (X-x[0])/h
        zero[1]= -2*gamma*(Y+x[1])/2 - omega**2*(X+x[0])/2 - (Y-x[1])/h
        return zero
    
    # Exact values
    t_sol = np.linspace(a, b, n+1)
    x_sol = func(t_sol)
    
    # numerical solution
    Time=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj=x
    
    for i in range(t_sol.size):
        t=i*h
        Time=np.hstack([Time,t])
        x_new = fs(IMP,x) 
        traj=np.vstack([traj,x_new])
        x=x_new
        
    Error[j] = np.abs((traj[n,0] - func(Time[n]))/func(Time[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1


print('By Implicit Midpoint Method')
results = [(N[i],H[i],T[i],Error[i]) for i in range(N.size)]
print(tabulate(results, headers=["n","h","Run time ","Rel. Error "], floatfmt=(".0f",".3f",".4f",".4e"),tablefmt="fancy_grid"))
#-------------------------------------------------------------------------------
j=0
for n in [100, 250, 500, 1000]:
    start_seconds = tls.time()

    # step
    h=(b-a)/n

    # Right side vectorial fuction 
    def NSIMP(p):
        X, Y = p
        zero = np.array([0.,0.])
        zero[0]= (Y+np.exp(-gamma*h)*x[1])/2 - (X-np.exp(-gamma*h)*x[0])/h
        zero[1]= - omega**2*(X+np.exp(-gamma*h)*x[0])/2 - (Y-np.exp(-gamma*h)*x[1])/h
        return zero
    
    # Exact values
    t_sol = np.linspace(a, b, n+1)
    x_sol = func(t_sol)
    
    # numerical solution
    Time=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj=x
    
    for i in range(t_sol.size):
        t=i*h
        Time=np.hstack([Time,t])
        x_new = fs(NSIMP,x) 
        traj=np.vstack([traj,x_new])
        x=x_new
        
    Error[j] = np.abs((traj[n,0] - func(Time[n]))/func(Time[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1


print('By Non-Standard Implicit Midpoint Method')
results = [(N[i],H[i],T[i],Error[i]) for i in range(N.size)]
print(tabulate(results, headers=["n","h","Run time ","Rel. Error "], floatfmt=(".0f",".3f",".4f",".4e"),tablefmt="fancy_grid"))