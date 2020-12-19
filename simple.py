import numpy as np
import matplotlib.pyplot as plt

# Defining the first order in dy/dt
def f1(t): return 10. + t
def f1_ex(t, y0): return y0 + 10*t + t**2/2

# Defining the first order in dy/dt
def f2(t): return (10. + t)**2
def f2_ex(t, y0): return y0 + 100.*t + 20.*t**2/2 + t**3/3

# initial conditions and parameters
y0 = 1.; t0 = 0.; N = 10

# arrays to store errors
error1 = np.zeros(N)
error2 = np.zeros(N)

# calculate
dts = .5**np.arange(0, N)
for i, dt in enumerate(dts):
    y1 = y0 + dt*f1(t0)
    error1[i] = np.abs(f1_ex(t0 + dt, y0) - y1)
    y1 = y0 + dt*f2(t0)
    error2[i] = np.abs(f2_ex(t0 + dt, y0) - y1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
ax.loglog(dts, error1, 'b.-', label = 'first order in f')
ax.loglog(dts, error2, 'r.-', label = 'second order in f')
ax.plot([10**-2, 10**-1], [10**-5, 10**-3], 'k.-')
ax.grid()
ax.legend()
plt.show()