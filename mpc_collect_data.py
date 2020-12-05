from casadi import *
import numpy as np
from Dynamic_system import simple_CSTR as System#specifications, DAE_system, integrator_model
from utilities import MPC, cosntract_history
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
N_training = 6000
his_x   = np.zeros([N_training, len(x0)])
his_u   = np.zeros([N_training, 1])
his_obj = np.zeros([N_training, 1])
his_feature = []
stp = 0.2
hist_states = cosntract_history(Sys, 5, store_u=True, set_point0=np.array([stp]))
t    = 0.
u_apply = np.array([0.])
k = 0
for i in range(N_training):
    his_x[i] = x0
    if abs(x0-stp)<=1e-4:#np.mod(i,100)==0 and i>=100:
        k+=1
    if k==10:
        k  = 0
        stp = np.random.rand()*0.3+0.1
    Feature  = hist_states.append_history(x0, u_apply, x0-stp)
    his_feature.append(Feature.copy())
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, u=u_apply, ref = stp, t=t)

    u_apply    = np.array(u_opt)[:,0]
    his_u[i]   = u_apply
    his_obj[i] = MPC_.obj

    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt


from simpleTorch.train_ann import *

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_size, action_size, action_range= None):
        super().__init__()
        if action_range==None:
            # self.action_low, self.action_high =\
            #     torch.from_numpy(np.array([0.]*action_size)), torch.from_numpy(np.array([1.]*action_size))#, tordokimase to ch.from_numpy(np.array([1.]*action_size))
            self.range_available = False
        else:
            self.range_available = True
            self.action_low, self.action_high = torch.from_numpy(np.array(action_range))
        self.layer1 = nn.Linear(state_size, 40)
        self.layer2 = nn.Linear(40, 40)
        self.layer3 = nn.Linear(40, 40)
        self.action = nn.Linear(40, action_size)

    def forward(self, state):
        m      = torch.nn.LeakyReLU(0.1)#0.01)
        layer1 = m(self.layer1(state))
        layer2 =m(self.layer2(layer1))
        layer3 = m(self.layer3(layer2))
        action = torch.sigmoid(self.action(layer3))
        if self.range_available:
            return self.action_low + (self.action_high - self.action_low) * (action)
        else:
            return (action)
#

his_f = np.array(his_feature)
model_u = Model(his_f.shape[1],his_u.shape[1])

ANN_u = train_ann(model_u, his_f, his_u, normalize_y=(0,1), epoch=200, plot=True, print_val=False)
model_obj = Model(his_f.shape[1],his_obj.shape[1])

ANN_obj = train_ann(model_obj, his_f, his_obj, normalize_y=(0,1), epoch=200, plot=True)

plt.plot(his_obj[:i+1])
plt.plot(ANN_u.predict(his_f),'r--')
plt.plot(ANN_obj.predict(his_f),'r--')
plt.plot(his_u[:i+1])
print('2')
