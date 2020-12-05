from casadi import *
import numpy as np
from utilities import MPC
import matplotlib.pyplot as plt
from Dynamic_system_small import Bio_reactor as System#specifications, DAE_system, integrator_model

# System     = Bio_reactor
# System_unc = Bio_reactor_unc
N =6 # This is the total steps
#Initialize system
Sys = System()
F = Sys.integrator_model()
# Construct MPC
MPC_ = MPC(System, horizon=N, penalize_u=False)
# Define system
dt, x0, _, _, _ = Sys.specifications()

# Initialize Matrices
N_mc  = 1
his_x = np.zeros([N_mc, 1000, len(x0)])
his_u = np.zeros([N_mc, 1000, 2])
u_apply = np.array([0.])
u1 = np.array([265.15893977, 269.47831514, 271.16360361, 220.6355221 ,
       301.72600227, 120.00073468]).reshape((1,-1))
u2 = np.array([25.23242466, 20.31618808, 39.68463853, 39.999973  , 39.99999722,
       39.99999775]).reshape((1,-1))*0.8
uu = np.vstack((u1,u2))
for kk in range(N_mc):
    Sys = System()
    F = Sys.integrator_model()
# Construct MPC
    MPC_ = MPC(System, horizon=6, penalize_u=False)
    dt, x0, _, _, _ = Sys.specifications()
    t     = 0.

    for i in range(N):
        # his_x[kk,i,:] = x0
        u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, t=t) # Deterministic Model
        #print(np.array(u_opt)[0,0])
        u_apply  = uu[:,i]#np.array([ np.array(u_opt)[:,0]])#np.array([[200, np.array(u_opt)[0,0]]])

        his_u[kk,i,:] = u_apply
        x1 = F(x0=x0, p=u_apply) #p=vertcat(u_apply, dp))           # Real System
        x0 = np.array(x1['xf']).reshape((-1,))# + np.random.multivariate_normal([0.]*3,np.array([400.,1e5,1e-2]*diag(np.ones(3))*1e-6))
        his_x[kk,i,:] = x0

        t += dt
print(his_u.shape)
plt.plot(his_x[0,:N,2])
