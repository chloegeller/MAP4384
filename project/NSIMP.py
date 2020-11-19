##############################################################################
##
## Implement the Non-Standard Implicit Midpoint method (NSIMP)
##
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.optimize import fsolve as fs

 # Initial parameters
a, b = 0, 1
omega = 1.0
gamma = 0.1
x0 = 1.0
h=0.1
x=np.array([x0,gamma*x0])
traj=x

# Exact solution function
def func(t):
    beta = np.sqrt(omega**2 - gamma**2)
    return x0*np.exp(-gamma*t)*np.cos(beta*t)

# Right side vectorial fuction 
def NSIMP(p):
    X, Y = p
    deriv = np.array([0.,0.])
    deriv[0]= (Y+np.exp(-gamma*h)*x[1])/2 - (X-np.exp(-gamma*h)*x[0])/h
    deriv[1]= - omega**2*(X+np.exp(-gamma*h)*x[0])/2 - (Y-np.exp(-gamma*h)*x[1])/h
    return deriv

# Exact values
t_sol = np.linspace(a, b, (b-a)*10+1)
x_sol = func(t_sol)

# Appr solution
time=np.zeros(1)

for i in range(1, t_sol.size):
    t=i*h
    time=np.hstack([time,t])
    x_new = fs(NSIMP,x) 
    traj=np.vstack([traj,x_new])
    x=x_new

print('\n By Non-standard Implicit Midpoint method: \n')
results = [(time[i],traj[i][0], traj[i][1]) for i in range(time.size)]
print(tabulate(results, headers=["time","x(t)", "x'(t)"], floatfmt=(".2f",".5e",".5e"),tablefmt="fancy_grid"))

plt.plot(time,traj[:,0],'gh-',label='By NSIMP')

# Plot of the exact solution
plt.plot(t_sol, x_sol, 'r-', linewidth=2,label='Exact')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.title('Exact vs Numerical solution by NSIMP')
plt.show()
