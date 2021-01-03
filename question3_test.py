import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt

kappa = 0.01
U = 0.5
# mesh information
Lx = 0.8
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
    apply_bcs(A, b, 0, 1, bc_option=1)
    u = np.linalg.solve(A, b)
    u_old = np.copy(u)
    t += dt

plt.plot(u)
plt.show()
# hhhhh