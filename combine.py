import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt

kappa = 0.01
U = 0.2
# mesh information
Lx = 1
N_nodes = 101
N_elements = N_nodes - 1
x_nodes = np.linspace(0, Lx, N_nodes)
dx = np.diff(x_nodes)

# Number of odes per element
N_loc = 2

# the connectivity matrix give us mapping from local to global numbering
connectivity_matrix = np.zeros((N_loc, N_elements), dtype=int)
for element in range(N_elements):
    connectivity_matrix[0, element] = element
    connectivity_matrix[1, element] = element + 1
# Define initial condition
def initial_condition(x): return np.exp(-(x - 0.5)**2 / 0.005)
# time stepping
dt = 0.001
t = 0
t_end = dt*1000
theta = 0
u_ic = initial_condition(x_nodes)


# construct Matrix M, K

# initialise M,K to a zero array
M = np.zeros((N_nodes, N_nodes))
K = np.zeros((N_nodes, N_nodes))
P = np.zeros((N_nodes, N_nodes))

# loop over all elements
for element in range(N_elements):
    # loop over local nodes
    for i_local in range(N_loc):
        i_global = connectivity_matrix[i_local, element]

        for j_local in range(N_loc):
            j_global = connectivity_matrix[j_local, element]
            if(i_local == 0):
                if(j_local == 0):      
                    # again, these integrals will be explained properly in the next cell
                    integrand = lambda xi: 0.5*(1-xi) * 0.5*(1-xi)
                else:
                    integrand = lambda xi: 0.5*(1-xi) * 0.5*(1+xi)
            else:
                if(j_local == 0):           
                    integrand = lambda xi: 0.5*(1+xi) * 0.5*(1-xi)
                else:
                    integrand = lambda xi: 0.5*(1+xi) * 0.5*(1+xi)
            # add in the local contribution to the global mass matrix
            M[i_global,j_global] += 0.5*dx[element] * si.quad(integrand, -1, 1)[0]

        for j_local in range(N_loc):
            j_global = connectivity_matrix[j_local, element]
            if(i_local == 0):
                if(j_local == 0):      
                    # again, these integrals will be explained properly in the next cell
                    integrand = lambda xi: (-1/dx[element]) * (-1/dx[element])
                else:
                    integrand = lambda xi: (-1/dx[element]) * (1/dx[element])
            else:
                if(j_local == 0):           
                    integrand = lambda xi: (1/dx[element]) * (-1/dx[element])
                else:
                    integrand = lambda xi: (1/dx[element]) * (1/dx[element])
            # add in the local contribution to the global mass matrix
            K[i_global,j_global] += 0.5*dx[element] * si.quad(integrand, -1, 1)[0]

        for j_local in range(N_loc):
            j_global = connectivity_matrix[j_local, element]
            if(i_local == 0):
                if(j_local == 0):      
                    # again, these integrals will be explained properly in the next cell
                    integrand = lambda xi: (1/dx[element]) * 0.5*(1-xi)
                else:
                    integrand = lambda xi: (1/dx[element]) * 0.5*(1+xi)
            else:
                if(j_local == 0):           
                    integrand = lambda xi: (-1/dx[element]) * 0.5*(1-xi)
                else:
                    integrand = lambda xi: (-1/dx[element]) * 0.5*(1+xi)
            # add in the local contribution to the global mass matrix
            P[i_global,j_global] += 0.5*dx[element] * si.quad(integrand, -1, 1)[0]


def apply_bcs(A, b, lbc, rbc, bc_option=0):
    """Apply BCs using a big spring method.
    
    bc_option==0 Homogeneous Neumann
    bc_option==1 inhomogeneous Dirichlet
    """
    if(bc_option==0):
        # for homogeneous Neumann conditions, for this problem, we have to **do nothing**!
        pass
    elif(bc_option==1):
        big_spring = 1.0e10
        A[0,0] = big_spring            
        b[0]   = big_spring * lbc
        A[-1,-1] = big_spring            
        b[-1]   = big_spring * rbc         
    else:
        raise Exception('bc option not implemented')

A = M + dt*theta*(kappa*K)
RHS_matrix = M - dt*(1-theta)*(kappa*K) - dt*U*P

# and finally time step
u_old = np.copy(u_ic)
while t<t_end:
    b = RHS_matrix @ u_old.T 
    apply_bcs(A, b, 0, 0, bc_option=1)
    u = np.linalg.solve(A, b)
    u_old = np.copy(u)
    t += dt

fig = plt.figure(figsize  =(12, 12))
ax = fig.add_subplot(111)
ax.plot(x_nodes, u, 'b', label = 'element')
ax.plot(x_nodes, u_ic, 'r', label = 'exact_element')


#################################


# physics parameters
# kappa = 0.01
# U = 0.2
# # mesh information
# Lx = 1
# N_nodes = 101
# N_elements = N_nodes - 1
# x_nodes = np.linspace(0, Lx, N_nodes)
dx = dx[0]

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

# use the function we just wrote to form the spatial mesh and the discretisation matrix 
A, x = assemble_adv_diff_disc_matrix_central(U, kappa, Lx, N_nodes)
# define a time step size
dt = 0.001
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

ax.plot(x, C[:,0], label = 'exact_diff')
ax.plot(x, C[:, -1], label = 'diff')
ax.legend()
ax.set_xlim(0, 1)
plt.show()