from casadi import *
import numpy as np
from Dynamic_system import simple_CSTR as System#specifications, DAE_system, integrator_model
from utilities import MPC
import matplotlib.pyplot as plt
T = 1.  # Time horizon
N = 12  # number of control intervals


# nk, tf, x0, Lsolver, c_code = specifications()
#
# xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, \
# algebraics, inputs, nd, na, nu, nmp, modparval, ng, gfcn, Obj_M, Obj_L= DAE_system()
#



# Objective term

# Formulate discrete time dynamics


# Fixed step Runge-Kutta 4 integrator

MPC_ = MPC(System, 100, penalize_u=True, collocation_degree=8)
Sys = System()
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()
his_x = np.zeros([1000, len(x0)])
his_u= np.zeros([1000, 1])
t    = 0.
u_apply = np.array([0.])

for i in range(1000):
    his_x[i] = x0
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, u=u_apply, ref = 0.4, t=t)

    u_apply  = np.array(u_opt)[:,0]
    his_u[i] = u_apply
    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt


plt.plot(his_x)

print('2')