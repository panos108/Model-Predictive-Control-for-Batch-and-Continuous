from casadi import *
import numpy as np
from Dynamic_system import Bio_reactor as System#specifications, DAE_system, integrator_model
from utilities import MPC
import matplotlib.pyplot as plt
T = 1.  # Time horizon
N = 6  # number of control intervals


# nk, tf, x0, Lsolver, c_code = specifications()
#
# xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, \
# algebraics, inputs, nd, na, nu, nmp, modparval, ng, gfcn, Obj_M, Obj_L= DAE_system()
#



# Objective term

# Formulate discrete time dynamics


# Fixed step Runge-Kutta 4 integrator
M = 4  # RK4 steps per interval


MPC_ = MPC(System, 12, penalize_u=False)

u_opt, x_opt, w_opt = MPC_.solve_MPC(np.array([1, 150,0]), t=0.)
# solver = nlpsol('solver', 'ipopt', prob)#'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
#
# # Solve the NLP
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p =  np.array([1.,100.,0.]))
Sys = System()
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()
his_x = np.zeros([1000, len(x0)])
his_u= np.zeros([1000, 2])
t    = 0.
for i in range(6):
    his_x[i] = x0
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, t=t)

    u_apply  = np.array(u_opt)[:,0]
    his_u[i] = u_apply
    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt
