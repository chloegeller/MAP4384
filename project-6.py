# All libraries will be imported in this cell
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve as fs
from tabulate import tabulate as table

def mid_point():    
    y = np.zeros(steps+1)
    xl = np.zeros(steps+1)
    
    xp0 = x0 * gamma
    
    xl[0] = x0
    y[0] = xp0
    
    for i in range(0, steps):
        if i == 0:
            xl[i+1] = 2 * h * y[i]
            y[i+1] = -4 * gamma * h * y[i] - 2 * omega**2 * h * xl[i]
        else:
            xl[i+1] = xl[i-1] + 2*  h * y[i]
            y[i+1] = y[i-1] - 4 * gamma * h * y[i] - 2 * omega**2 * h * xl[i]
    return xl

def IMP(p):
    X, Y = p
    
    deriv = np.array([0., 0.])
    deriv[0] = (Y + x_imp[1]) / 2 - (X - x_imp[0]) / h
    deriv[1] = -2 * gamma * (Y + x_imp[1]) / 2 - omega**2 * (X + x_imp[0]) / 2 - (Y - x_imp[1]) / h
    
    return deriv

def implicit_mid_point():    
    global x_imp
    x_imp = np.array([x0, xp0])
    traj = x_imp

    for i in range(1, steps+1):
        x_new = fs(IMP, x_imp) 
        traj = np.vstack([traj, x_new])
        x_imp = x_new
    
    return traj

def NSIMP(p):
    X, Y = p
    deriv = np.array([0., 0.])
    deriv[0] = (Y + np.exp(-gamma * h) * x_nsimp[1]) / 2 - (X - np.exp(-gamma * h) * x_nsimp[0]) / h
    deriv[1] = -omega**2 * (X + np.exp(-gamma * h) * x_nsimp[0]) / 2 - (Y - np.exp(-gamma * h) * x_nsimp[1]) / h
    
    return deriv

def ns_implicit_mid(): 
    global x_nsimp
    x_nsimp = np.array([x0, xp0])
    traj = x_nsimp

    for i in range(1, steps+1):
        x_new = fs(NSIMP, x_nsimp) 
        traj = np.vstack([traj, x_new])
        x_nsimp = x_new
    
    return traj

def exact():
    y = np.zeros(steps+1)
    b = np.sqrt((omega**2) - (gamma**2))
    
    for i in range(0, (steps+1)):
        y[i] = x0 * np.exp(-gamma * t[i]) * np.cos(b * t[i])
    return y

def relative_error(exact, approx, steps):
    rel = np.zeros((steps+1,))
    
    for i in range(0, steps+1):
        rel[i] = abs(exact[i] - approx[i]) / abs(exact[i])
        
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
    
    fig, axs = plt.subplots(2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    title1 = "Exact solution and methods: gamma=" + str(gamma) + ", omega=" + str(omega) + ", h=" + str(h)
    axs[0].plot(t, x_ex, "limegreen", label="Exact Solution", linewidth=5.5)
    axs[0].plot(t, x_m, "cornflowerblue", label="EMP")
    axs[0].plot(t, x_i[:,0], "blueviolet", marker="o", label="IMP")
    axs[0].plot(t, x_n[:,0], "darkorange", label="NSIMP")
    axs[0].set(xlabel = "time", ylabel = "x(t)")
    axs[0].set_title(title1)
    axs[0].legend(loc="upper right", fontsize="small")
    
    title2 = "Errors: gamma=" + str(gamma) + ", omega=" + str(omega) + ", h=" + str(h)
    axs[1].plot(t, error_exact, "limegreen", label="Exact Error", linewidth=5.5)
    axs[1].plot(t, error_mid, "cornflowerblue", label="EMP Error")
    axs[1].plot(t, error_imp, "blueviolet", marker="o", label="IMP Error")
    axs[1].plot(t, error_nsimp, "darkorange", label="NSIMP Error")
    axs[1].set(xlabel = "time", ylabel = "Relative Error")
    axs[1].set_title(title2)
    axs[1].legend(loc="upper right", fontsize="small")
    plt.show()
    

    print()
    print(f"Testing using: steps = {steps}, x0 = {x0}, gamma = {gamma}, omega = {omega}, and h = {h}.", end="\n\n\n\n")
    table_results = results(t, x_m, x_i[:,0], x_n[:,0], error_mid, error_imp, error_nsimp, x_ex)
    print(table(table_results, headers=["n", "t", "EMP", "IMP", "NSIMP", "EMP Error", "IMP Error", "NSIMP Error", "Exact Solution"]))
    print()
    print()
        
    return

# This function is only used in the section when gamma = 0, to generate the plots using only IMP and NSIMP
def gamma_0():
    global t0, h, xp0, t
    t0 = 0
    tf = 1
    h = (tf - t0) / steps
    xp0 = x0 * gamma
    t = np.arange(t0, tf + h, h)


    x_i = implicit_mid_point()
    x_n = ns_implicit_mid()
    x_ex = exact()


    error_imp = relative_error(x_ex, x_i[:,0], steps)
    error_nsimp = relative_error(x_ex, x_n[:,0], steps)
    error_exact = relative_error(x_ex, x_ex, steps)

    title1 = "Exact solution and methods: gamma=" + str(gamma) + ", omega=" + str(omega) + ", h=" + str(h)
    plt.figure(figsize=(8,5))
    plt.plot(t, x_i[:,0], "navy", label="IMP", linewidth=8)
    plt.plot(t, x_n[:,0], "deeppink", label="NSIMP",linewidth=3)
    plt.plot(t, x_ex, "limegreen", marker='o', label="Exact Solution", linewidth=0.5)
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(title1)
    plt.legend(loc="lower left", fontsize="small")
    plt.show()

    return

if __name__ == "__main__":
    global x0, steps, gamma, omega
    
    # ================================ Initial Values ================================
    steps = 10
    x0 = 1
    gamma = 0.1
    omega = 1
    run_test()
    # ================================================================================

    # ============================== Computational Cost ==============================
    s = [10, 25, 50, 100]
    g = [10e-4, 10e-3, 10e-2, 10e-1]
    o = [1, 2, 5, 10]
    times = []

    for steps in s:
        for gamma in g:
            for omega in o:
                temp = []
                temp.append(steps)
                temp.append(gamma)
                temp.append(omega)
                start = time.time()
                mid_point()
                temp.append(time.time() - start)
                start = time.time()
                implicit_mid_point()
                temp.append(time.time() - start)
                start = time.time()
                ns_implicit_mid()
                temp.append(time.time() - start)
                times.append(temp)

    print(table(times, headers=["steps", "gamma", "omega", "EMP Time", "IMP Time", "NSIMP Time"]))
    # ================================================================================

    # ======================== Implicit Methods when gamma = 0 =======================
    steps = 10
    x0 = 1
    gamma = 0
    omega = 1

    # Generates the plots used in the report
    gamma_0()

    # To generate the tables used in the presention use run_test(), we did not use the plots from run_test() for this section
    # run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 100
    # run_test()

    # steps = 1000
    # run_test()

    steps = 10
    x0 = 1
    gamma = 0
    omega = 5

    # Generates the plots used in the report
    gamma_0()

    # To generate the tables used in the presention use run_test(), we did not use the plots from run_test() for this section
    # run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 100
    # run_test()

    # steps = 1000
    # run_test()

    steps = 10
    x0 = 3
    gamma = 0
    omega = 6

    # Generates the plots used in the report
    gamma_0()

    # To generate the tables used in the presention use run_test(), we did not use the plots from run_test() for this section
    # run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 100
    # run_test()

    # steps = 1000
    # run_test()
    # ================================================================================

    # =============== Long-time accuracy for small values of gamma > 0 ===============
    steps = 100
    x0 = 1
    gamma = 1e-14
    omega = 1
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()

    steps = 100
    x0 = 1
    gamma = 1e-5
    omega = 1
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()

    steps = 100
    x0 = 1
    gamma = 1e-14
    omega = 5
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()

    steps = 100
    x0 = 1
    gamma = 1e-14
    omega = 50
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()


    steps = 100
    x0 = 1
    gamma = 0.5
    omega = 1
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()

    steps = 100
    x0 = 1
    gamma = 0.9
    omega = 1
    run_test()

    # Uncomment the lines below to generate the tables used in the report
    # steps = 10
    # run_test()

    # steps = 1000
    # run_test()
    # ================================================================================

    # =============================== Breaking Points ================================
    ## ODD
    steps = 25
    x0 = 1
    gamma = 0.1
    omega = 1
    run_test()

    ## EVEN
    steps = 26
    x0 = 1
    gamma = 0.1
    omega = 1
    run_test()

    steps = 20
    x0 = 1
    gamma = 0.1
    omega = 3.14/2
    run_test()

    # Error diverging and converging
    steps = 100
    x0 = 1
    gamma = 0.1
    omega = 12.56
    run_test()