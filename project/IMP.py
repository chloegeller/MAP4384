##############################################################################
##
## Implement the Implicit Midpoint method (IMP)
##
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.optimize import fsolve as fs

 # Initial parameters
a, b = 0, 10
omega = 5.0
gamma = 0.3
x0 = 100.0
h=0.1
x=np.array([x0,gamma*x0])
traj=x

# Exact solution function
def func(t):
    beta = np.sqrt(omega**2 - gamma**2)
    return x0*np.exp(-gamma*t)*np.cos(beta*t)

# Right side vectorial fuction 
def IMP(p):
    X, Y = p
    deriv = np.array([0.,0.])
    deriv[0]= (Y+x[1])/2 - (X-x[0])/h
    deriv[1]= -2*gamma*(Y+x[1])/2 - omega**2*(X+x[0])/2 - (Y-x[1])/h
    return deriv

# Exact values
t_sol = np.linspace(a, b, (b-a)*10+1)
x_sol = func(t_sol)

# Appr solution
time=np.zeros(1)

for i in range(t_sol.size):
    t=i*h
    time=np.hstack([time,t])
    x_new = fs(IMP,x) 
    traj=np.vstack([traj,x_new])
    x=x_new

print('\n By Implicit Midpoint method: \n')
results = [(time[i],traj[i][0], traj[i][1]) for i in range(time.size)]
print(tabulate(results, headers=["time","x(t)", "x'(t)"], floatfmt=(".2f",".5e",".5e"),tablefmt="fancy_grid"))

plt.plot(time,traj[:,0],'yh-',label='By IMP')

# Plot of the exact solution
plt.plot(t_sol, x_sol, 'r', linewidth=2,label='Exact')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.title('Exact vs Numerical solution by IMP')
plt.show()