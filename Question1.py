import numpy as np
import matplotlib.pyplot as plt

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


# Right hand side of the differential equation
def f(t, y): return y + t**3

# Excate solution in y(t)
def y_ex(t): return 7*np.exp(t) - t**3 - 3*t**2 - 6*t - 6


# set up parameters and initial conditions
y0 = 1.; t0 = 0; dt = 0.1; t_max = 3.; alpha = 1

# calculate rk2 results
y_RK2, t_RK2 = RK2_alpha(f, y0, t0, t_max, dt, alpha)


# plot
# tfine = np.arange(t0, t_max+dt/10, dt/10)
# yex = y_ex(tfine)
# fig = plt.figure(figsize = (12, 8))
# ax = fig.add_subplot(111)
# ax.plot(t_RK2, y_RK2, 'b', label = 'RK2 wiht $alpha$ = %.1f' %(alpha))
# ax.plot(tfine, yex, 'r', label = 'Exact Solution')
# ax.legend(fontsize = 14)
# ax.grid()
# plt.show()

# confrim when alpha = 1, the scheme produces same results as improved euler
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

# calculate IE results
y_IE, t_IE = improved_euler(f, y0, t0, t_max, dt)

# Test whether IE agrees with RK2 when alpha = 1
if np.allclose(y_IE, y_RK2) == True:
    print('When alpha=1, RK2 method agrees with improved Euler method')
else:
    print('When alpha=1, RK2 method does not agrees with improved Euler method')