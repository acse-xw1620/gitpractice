import numpy as np
import matplotlib.pyplot as plt
from mpltools import annotation

# RK2 Scheme
def RK2_alpha(f, y0, t0, t_max, dt, alpha=0.5):


    assert alpha != 0, 'alpha could not be zero'
    y = np.array(y0)
    t = np.array(t0)
    y_all = [y0]
    t_all = [t0]

    while t < t_max:
        k1 = f(t, y)
        k2 = f(t + alpha*dt, y + alpha*dt*k1)
        y = y + ((2*alpha - 1)/(2*alpha))*dt*k1 + (1/(2*alpha))*dt*k2
        y_all.append(y)
        t = t + dt
        t_all.append(t)
    

    return np.array(y_all), np.array(t_all)


# Forward Euler Scheme
def forward_euler(f, y0, t0, t_max, dt):


    y = np.array(y0)
    t = np.array(t0)
    y_all = [y0]
    t_all = [t0]

    while t < t_max:
        y = y + dt*f(t, y)
        y_all.append(y)
        t = t + dt
        t_all.append(t)

    return np.array(y_all), np.array(t_all)


# Improved Euler Scheme
def improved_euler(f, y0, t0, t_max, dt):


    y = np.array(y0)
    t = np.array(t0)
    y_all = [y0]
    t_all = [t0]

    while t < t_max:
        ye = y + dt*f(t, y)
        y = y + 0.5*dt*( f(t, y) + f(t + dt, ye) )
        y_all.append(y)
        t = t + dt
        t_all.append(t)

    return np.array(y_all), np.array(t_all)


# Right hand side of the differential equation
def f(t, y): return y + t**3


# Excate solution in y(t)
def y_ex(t): return 7*np.exp(t) - t**3 - 3*t**2 - 6*t - 6

## confrim when alpha = 1, the scheme produces same results as improved euler
# set up parameters and initial conditions
y0 = 1.; t0 = 0; dt = 0.1; t_max = 3.; alpha = 1
# calculate rk2 results
y_RK2_a1, t_RK2_a1 = RK2_alpha(f, y0, t0, t_max, dt, alpha)
# calculate IE results
y_IE, t_IE = improved_euler(f, y0, t0, t_max, dt)
# Test whether IE agrees with RK2 when alpha = 1
if np.allclose(y_IE, y_RK2_a1) == True:
    print('When alpha=1, RK2 method agrees with improved Euler method')
else:
    print('When alpha=1, RK2 method does not agrees with improved Euler method')


# Error Metric

N = 10
dts = 0.5**np.arange(0, N)

errors_last_FE = np.zeros(N)
errors_norm_FE = np.zeros(N)
errors_last_IE = np.zeros(N)
errors_norm_IE = np.zeros(N)
for i, dt in enumerate(dts):
    tex = np.arange(t0, t_max+dt, dt)
    yex = y_ex(tex)
    y_FE, t_FE = forward_euler(f, y0, t0, t_max, dt)
    y_IE, t_IE = improved_euler(f, y0, t0, t_max, dt)
    errors_last_FE[i] = np.abs(y_FE[-1] - yex[-1])
    errors_norm_FE[i] = np.abs(np.linalg.norm(y_FE - yex, ord = 1)/len(t_FE))
    errors_last_IE[i] = np.abs(y_IE[-1] - yex[-1])
    errors_norm_IE[i] = np.abs(np.linalg.norm(y_IE - yex, ord = 1)/len(t_IE))


fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111)
ax.loglog(dts,errors_last_FE, 'r.-', label = 'errors_last_FE')
ax.loglog(dts,errors_norm_FE, 'g.-', label = 'errors_norm_FE')
ax.loglog(dts,errors_last_IE, 'y.-', label = 'errors_last_IE')
ax.loglog(dts,errors_norm_IE, 'k.-', label = 'errors_norm_IE')
ax.legend()
ax.grid()
plt.show()