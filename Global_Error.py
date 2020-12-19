import numpy as np
import matplotlib.pyplot as plt

# Defining the first order in dy/dt
def f1(t): return 10. + t
def y1_ex(t, y0): return y0 + 10*t + t**2/2

# Defining the first order in dy/dt
def f2(t): return (10. + t)**2
def y2_ex(t, y0): return y0 + 100.*t + 20.*t**2/2 + t**3/3

# initial conditions and parameters
y0 = 1.; t0 = 0.; tf = 2; N = 10

# arrays to store errors
error1 = np.zeros(N)
error2 = np.zeros(N)

# calculate
dts = .5**np.arange(0, N)
for i, dt in enumerate(dts):
    y = y0; t = t0
    while t < tf:
        y += dt*f1(t)
        t += dt
    error1[i] = np.abs(y - y1_ex(t, y0))
    y = y0; t = t0
    while t < tf:
        y += dt*f2(t)
        t += dt
    error2[i] = np.abs(y - y2_ex(t, y0))


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
ax.loglog(dts, error1, 'b.-', label = 'first order in f')
ax.loglog(dts, error2, 'r.-', label = 'second order in f')
ax.plot([10**-2, 10**-3], [10**-1, 10**-2], 'k.-')
ax.grid()
ax.legend()
plt.show()