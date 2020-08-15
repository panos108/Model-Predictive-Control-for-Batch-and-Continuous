from casadi import *
import numpy as np
from Dynamic_system import polymer_CSTR as System#specifications, DAE_system, integrator_model
from utilities import MPC
import matplotlib.pyplot as plt
import time
MPC_ = MPC(System, 1, penalize_u=False, collocation_degree=8)
Sys = System()
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()
his_x = np.zeros([1000, len(x0)])
his_u= np.zeros([1000, 1])
t    = 0.
u_apply = np.array([0.1675])
his_t = np.zeros([1000])
for i in range(1000):
    his_x[i] = x0
    start = time.time()
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, u=u_apply,  ref = [3.0780178085943333, 0.14874773146466175,
                                                                  0.019293038350734066*100, 292.54885691405192/100], t=t) #
    his_t[i] = time.time()-start
    u_apply  = np.array(u_opt)[:,0]
    his_u[i] = u_apply
    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt


plt.plot(his_x)

print('2')