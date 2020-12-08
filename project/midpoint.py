## notes
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as fs
from tabulate import tabulate as table


#####################################
 #                                 #
 #        Explicit Midpoint        #
 #                                 #
#####################################

def mid_point():
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

def implicit_mid_point():
    
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

def ns_implicit_mid():
    
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

def relative_error(exact, aprox, steps):
    rel = np.zeros((steps+1,))
    for i in range(0, steps+1):
        rel[i] = abs(exact[i]-aprox[i])/abs(exact[i])
    return rel

def results(t, x_m, x_i, x_n, err1, err2, err3, x_ex):
    return list(zip(range(t.shape[0]), t, x_m, x_i, x_n, err1, err2, err3, x_ex))

def run_test():
    global t0, h, xp0, t

    t0 = 0
    tf = 1
    h = (tf - t0) / steps
    xp0 = x0 * gamma
    t = np.arange(t0, tf + h, h)

    x_m = mid_point()
    x_i = implicit_mid_point()
    x_n = ns_implicit_mid()
    x_ex = exact()

    error_mid = relative_error(x_ex, x_m, steps)
    error_imp = relative_error(x_ex, x_i[:,0], steps)
    error_nsimp = relative_error(x_ex, x_n[:,0], steps)
    error_exact = relative_error(x_ex, x_ex, steps)
    
    title1 = "Exact solution and methods: gamma=" + str(gamma) + ", omega=" + str(omega) + ", h=" + str(h)
    plt.figure(figsize=(8, 6))
    plt.plot(t, x_ex, "limegreen", label="Exact Solution", linewidth=5.5)
    plt.plot(t, x_m, "cornflowerblue", label="EMP")
    plt.plot(t, x_i[:,0], "blueviolet", marker="o", label="IMP")
    plt.plot(t, x_n[:,0], "darkorange", label="NSIMP")
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(title1)
    plt.legend(loc="upper right", fontsize="small")
    plt.savefig("./plots/"+title1+".png")
    plt.show()
    
    title2 = "Errors: gamma=" + str(gamma) + ", omega=" + str(omega) + ", h=" + str(h)
    plt.figure(figsize=(8,6))
    plt.plot(t, error_exact, "limegreen", label="Exact Error", linewidth=5.5)
    plt.plot(t, error_mid, "cornflowerblue", label="EMP Error")
    plt.plot(t, error_imp, "blueviolet", marker="o", label="IMP Error")
    plt.plot(t, error_nsimp, "darkorange", label="NSIMP Error")
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(title2)
    plt.legend(loc="upper right", fontsize="small")
    plt.savefig("./plots/"+title2+".png")
    plt.show()
    

    print()
    print(f"Testing using: steps = {steps}, x0 = {x0}, gamma = {gamma}, omega = {omega}, and h = {h}.", end="\n\n\n\n")
    table_results = results(t, x_m, x_i[:,0], x_n[:,0], error_mid, error_imp, error_nsimp, x_ex)
    print(table(table_results, headers=["n", "t", "EMP", "IMP", "NSIMP", "EMP Error", "IMP Error", "NSIMP Error", "Exact Solution"]))
    print()
    print()
        
    return

if __name__ == "__main__":
    global x0, steps, gamma, omega
    steps = 10
    x0 = 1
    gamma = 0.1
    omega = 1
    run_test()
