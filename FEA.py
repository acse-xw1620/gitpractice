import numpy as np


x = np.linspace(0, 3, 4)
print(x[0:])
dx = np.diff(x)
M = np.zeros((len(x), len(x)))
# M[1:-1, 1:-1] = 
print(np.diag(dx[:-1]/3 + dx[1:]/3, k = 0))