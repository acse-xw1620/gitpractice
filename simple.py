import numpy as np
import matplotlib.pyplot as plt


a = np.arange(0, 10, 0.1)
y = np.exp(-a)

fig = plt.figure(figsize = (12, 4))
ax = fig.add_subplot(111)
ax.plot(a, y, 'ro', label = 'simple exp function')
ax.grid()
ax.legend(loc = 'best', fontsize = 15)
ax.set_title('My title')
plt.show()