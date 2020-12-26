import numpy as np
import scipy.linalg as la # scipy version 1.5.4


def adv_diff_ana_sol(U, kappa, x0, x, t):
    return np.exp( -((x-x0) - U*t)**2/(4.*kappa*t)) / np.sqrt(4. * np.pi*kappa*t)