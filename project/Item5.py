##############################################################################
## For gamma=0, the implicit methods are equivalent, and we expect they are superior 
## to the explicit method; check this numerically using long time simulations 
## with various h,omega,x0.
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import time as tls
from scipy.optimize import fsolve as fs

# Initial parameters (global varibles)
a, b = 0, 10
omega = 5.0
gamma = 0.0
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
Error = np.zeros(4)
N = np.zeros(4)
T = np.zeros(4)
H = np.zeros(4)
#-------------------------------------------------------------------------------
j=0
for n in [100, 500, 1000, 5000]:
    start_seconds = tls.time()

    # step
    h=(b-a)/n

    def EMP(x,t):
        return x + h*dxdt(t + h/2, x + (h/2)*dxdt(t,x))
    
    # Exact values
    t_sol = np.linspace(a, b, n+1)
    x_sol = func(t_sol)
    
    # numerical solution
    Time1=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj1=x
    
    for i in range(n):
        t = i*h
        Time1 = np.hstack([Time1,t])
        x_new = EMP(x,t)
        traj1 = np.vstack([traj1,x_new])
        x = x_new
    
    Error[j] = np.abs((traj1[n,0] - func(Time1[n]))/func(Time1[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1

    # Plot of the numerical solution
    plt.figure(1)
    plt.subplot(2,2,j)
    plt.plot(Time1,traj1[:,0],'mh-',label='EMP')

    # Plot of the exact solution
    plt.plot(t_sol, x_sol, 'k-', linewidth=2,label='Exact')
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.xlabel('t',fontsize=18)
    plt.ylabel('x(t)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Exact vs Numerical solution by EMP \n h = '+str(h)+' $\omega =$'+str(omega)+' and $\gamma =$'+str(gamma),fontsize=18)
    
plt.savefig("EMP5")   
# #-------------------------------------------------------------------------------
j=0
for n in [100, 500, 1000, 5000]:
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
    Time2=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj2=x
    
    for i in range(t_sol.size):
        t=i*h
        Time2=np.hstack([Time2,t])
        x_new = fs(IMP,x) 
        traj2=np.vstack([traj2,x_new])
        x=x_new
        
    Error[j] = np.abs((traj2[n,0] - func(Time2[n]))/func(Time2[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1

    # Plot of the numerical solution
    plt.figure(2)
    plt.subplot(2,2,j)
    plt.plot(Time2,traj2[:,0],'gh-',label='IMP')

    # Plot of the exact solution
    plt.plot(t_sol, x_sol, 'k-', linewidth=2,label='Exact')
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.xlabel('t',fontsize=18)
    plt.ylabel('x(t)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Exact vs Numerical solution by IMP \n h = '+str(h)+' $\omega =$'+str(omega)+' and $\gamma =$'+str(gamma),fontsize=18)
    
plt.savefig("IMP5")   

#-------------------------------------------------------------------------------
j=0
for n in [100, 500, 1000, 5000]:
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
    Time3=np.zeros(1)
    x=np.array([x0,gamma*x0])
    traj3=x
    
    for i in range(t_sol.size):
        t=i*h
        Time3=np.hstack([Time3,t])
        x_new = fs(NSIMP,x) 
        traj3=np.vstack([traj3,x_new])
        x=x_new
        
    Error[j] = np.abs((traj3[n,0] - func(Time3[n]))/func(Time3[n]))
    T[j] = tls.time() - start_seconds  
    N[j] = n 
    H[j] = h
    j=j+1

    # Plot of the numerical solution
    plt.figure(3)
    plt.subplot(2,2,j)
    plt.plot(Time3,traj3[:,0],'ch-',label='NSIMP')

    # Plot of the exact solution
    plt.plot(t_sol, x_sol, 'k-', linewidth=2,label='Exact')
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    plt.xlabel('t',fontsize=18)
    plt.ylabel('x(t)',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Exact vs Numerical solution by NSIMP \n h = '+str(h)+' $\omega =$'+str(omega)+' and $\gamma =$'+str(gamma),fontsize=18)
    
plt.savefig("NSIMP5") 
plt.show()