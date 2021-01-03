import A_Matrix as a
import numpy as np
import matplotlib.pyplot as plt

Pe = 100
L = 1
U = 1
CE = 0
N = 500

dt = 0.0002; tend = 1
dx = L/N

A, x = a.Amatrix(Pe, L, U, N,CE, Periodic = True) ###

t = np.arange(0, tend, dt)
C = np.empty((len(x), len(t)))
C[:, 0] = np.exp(-((x-0.2)/0.05)**2)#CE*x/L # initial condition when t = 0

# loop over time
I = np.eye(len(x))
A_FTCS = (I + A*dt)
A_BTCS = np.linalg.inv((I - A*dt))
A_CN = (I +0.5*dt*A)@np.linalg.inv(I-0.5*dt*A)
for i in range(len(t)-1):
    C[:, i+1] = A_CN@C[:, i]



# Construct the exact solution to the steady state
# xf = np.linspace(0, L, 1000)
# Cex = CE*(np.exp(Pe*xf/L) - 1)/(np.exp(Pe) - 1)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111)

for i in np.arange(0, len(t), int(len(t)/10)):
    ax.plot(x, C[:, i], '.-', label = 'time = %.0f s' %(i))

# ax.plot(xf, Cex, 'k', lw = 3, label = 'Exact solution steady state')
ax.set_xlim(0, 1)
ax.grid()
ax.legend()
plt.show()