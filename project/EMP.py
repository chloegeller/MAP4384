##############################################################################
##
## Implement the Explicit Midpoint method (EMP)
##
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Initial parameters (global varibles)
a, b = 0, 10
omega = 5.0
gamma = 0.3
x0 = 100.0
h = 0.1

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

def EMP(x,t):
    return x + h*dxdt(t + h/2, x + (h/2)*dxdt(t,x))

# Exact values
t_sol = np.linspace(a, b, (b-a)*10+1)
x_sol = func(t_sol)

# Appr solution
time=np.zeros(1)
x=np.array([x0,gamma*x0])
traj=x

for i in range(t_sol.size):
    t=i*h
    time=np.hstack([time,t])
    x_new = EMP(x,t)
    traj=np.vstack([traj,x_new])
    x=x_new

print('\n By Explicit Midpoint method: \n')
results = [(time[i],traj[i][0], traj[i][1]) for i in range(time.size)]
print(tabulate(results, headers=["time","x(t)", "x'(t)"], floatfmt=(".2f",".5e",".5e"),tablefmt="fancy_grid"))

plt.plot(time,traj[:,0],'mh-',label='By EMP')
    
# Plot of the exact solution
plt.plot(t_sol, x_sol, 'k-', linewidth=2,label='Exact')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.title('Exact vs Numerical solution by EMP')
plt.show()