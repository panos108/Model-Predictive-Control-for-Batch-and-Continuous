from casadi import *
import numpy as np
from Dynamic_system import Semi_Batch as System#specifications, DAE_system, integrator_model
from utilities import MPC
import matplotlib.pyplot as plt

MPC_ = MPC(System, 20, penalize_u=True, collocation_degree=8)

# solver = nlpsol('solver', 'ipopt', prob)#'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
#
# # Solve the NLP
# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p =  np.array([1.,100.,0.]))
Sys = System()
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()
his_x = np.zeros([1000, len(x0)])
his_u= np.zeros([1000, 2])
u_apply = np.array([0.,0.])

t    = 0.
for i in range(20):
    his_x[i] = x0
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, u=u_apply, t=t)

    u_apply  = np.array(u_opt)[:,0]
    his_u[i] = u_apply
    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt


plt.plot(his_x[:20,0])

plt.show()
print(2)