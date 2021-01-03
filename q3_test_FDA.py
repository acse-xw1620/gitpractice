import numpy as np
import matplotlib.pyplot as plt

# physics parameters
kappa = 0.01
U = 0.2
# mesh information
Lx = 1
N_nodes = 101
N_elements = N_nodes - 1
x_nodes = np.linspace(0, Lx, N_nodes)
dx = np.diff(x_nodes)[0]

def initial_condition(x): return np.exp(-(x - 0.5)**2 / 0.005)
u_ic = initial_condition(x_nodes)

def assemble_adv_diff_disc_matrix_central(U, kappa, L, N):

    # define spatial mesh
    dx = L / N
    x = np.linspace(-dx / 2, dx / 2 + L, N + 2)
    # define first the parameters we defined above
    r_diff = kappa / dx**2
    r_adv = 0.5 * U / dx

    # and use them to create the B matrix - recall zeros on the first and last rows
    B = np.zeros((N + 2, N + 2))
    for i in range(1, N + 1):
        B[i, i - 1] = r_diff + r_adv
        B[i, i] = -2 * r_diff
        B[i, i + 1] = r_diff - r_adv

    # create M matrix - start from the identity
    M = np.eye(N + 2)
    # and fix the first and last rows
    M[0,(0,1)] = [0.5, 0.5]
    M[-1,(-2,-1)] = [0.5, 0.5]   

    # find A matrix
    A = np.linalg.inv(M) @ B
    return A, x

# physical parameters


# set the RHS boundary value to zero as well as the left
CE = 0


# define number of points in spatial mesh (N+2 including ghose nodes)
N = 100

# use the function we just wrote to form the spatial mesh and the discretisation matrix 
A, x = assemble_adv_diff_disc_matrix_central(U, kappa, Lx, N_nodes)

# define a time step size
dt = 0.001

# and compute and print some key non-dimensional parameters
dx = Lx / N
print('Pe_c: {0:.5f}'.format(U*dx/kappa))
print('CFL: {0:.5f}'.format(U*dt/dx))
print('r: {0:.5f}'.format(kappa*dt/(dx**2)))

# define the end time and hence some storage for all solution levels
tend = dt*1000
# assume a constant dt and so can define all the t's in advance
t = np.arange(0, tend, dt)
# and we can also set up a matrix to store the discrete solution in space-time
# with our a priori knowledge of the size required.
C = np.empty((len(x),len(t)))

# define an initial condition - a "blob" in the shape of a Gaussian
# and place it in the first column of the C matrix which stores all solution levels
C[:, 0] = initial_condition(x)

# now let's do the time-stepping via a for loop
# we will need the identity matrix so define it once outside the loop
I = np.eye(len(x))
for n in range(len(t)-1):
    C[:,n+1] = (I + A * dt) @ C[:, n]
    

# set up figure
fig = plt.figure(figsize=(7, 7))
ax1 = plt.subplot(111)
ax1.set_xlabel('$x/L$', fontsize=16)
ax1.set_ylabel('$C/C_E$', fontsize=16)
ax1.set_title('Adv-Diff time-dependent PDE solve', fontsize=16)

# let's plot every 500th time level
for i in np.arange(0,len(t),500):
    ax1.plot(x, C[:,i], '.-')


# to exclude ghost points just restrict the x limit of the plot
ax1.set_xlim(0, 1)
plt.show()