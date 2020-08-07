from casadi import *
import numpy as np
from Dynamic_system import different_CSTR as System#specifications, DAE_system, integrator_model
from OrthogonalCollocation import perform_orthogonal_collocation, construct_polynomials_basis
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
M = 4  # RK4 steps per interval


MPC_ = MPC(System, 100, penalize_u=True)
Sys = System()
dt, x0, _, _, _ = Sys.specifications()
F = Sys.integrator_model()
his_x = np.zeros([1000, len(x0)])
his_u= np.zeros([1000, 1])
t    = 0.
u_apply = np.array([0.])

for i in range(200):
    his_x[i] = x0
    u_opt, x_opt, w_opt = MPC_.solve_MPC(x=x0, u=u_apply, ref = 0.2, t=t)

    u_apply  = np.array(u_opt)[:,0]
    his_u[i] = u_apply
    x1 = F(x0=x0, p=u_apply)
    x0 = np.array(x1['xf']).reshape((-1,))
    t += dt

F1 = integrator_model()


# Evaluate at a test point
#Fk = F1(x0=[0.2, 0.3], p=[0.4, 0], DT=T / N)

#print(Fk['xf'])
hisx11 = np.zeros([100,3, N+1])
hisx12 = np.zeros([100, N+1])
hisu11 = np.zeros([100, N+1])
hisu12 = np.zeros([100, N+1])
N = 12
ind = 0
opt = [280.20281982, 286.00323486, 284.53033447, 286.36618042,
      284.40078735, 277.26065063, 263.92999268, 125.36869049,
      120.22846222, 120.0776062 , 120.03261566, 120.01446533]


#[280.20281982, 286.00323486, 284.53033447, 286.36618042,
#       284.40078735, 277.26065063, 263.92999268, 125.36869049,
#       120.22846222, 120.0776062 , 120.03261566, 120.01446533]
dc = 4
C, D, B = construct_polynomials_basis(dc, 'radau')
for kkk in range(1):
    N1 = N + 1
    dun = np.array(modparval)
    x0              = np.array([1.,150.,0.])

    dist = (np.random.multivariate_normal(np.zeros(dun.shape),
                                          np.diag(dun * 0.00)))
    x0 += np.random.multivariate_normal([0, 0, 0], np.diagflat([1e-3 * np.array([1., 150. ** 2, 0.])]))

    Fk11 = x0
    x11 = np.zeros(N+1)
    x12 = np.zeros(N+1)
    u11 = np.zeros(N)
    u12 = np.zeros(N)
    x_plot = []
    y_plot = []
    u_plot = []
    c3_opt = [0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.00760708, 0.03529533, 0.0653219 ,
       0.09717915, 0.1304817 , 0.16487212]
    c1_opt = [ 1.        ,  1.28125778,  1.81404381,  2.65369506,  3.92985956,
        5.81200785,  8.40601442, 11.17627753, 12.79172547, 14.1643698 ,
       15.49017448, 16.78072497, 18.03987993]
    c2_opt = [ 150.        ,  796.80474151, 1312.6204907 , 1666.78915072,
       1790.1964449 , 1592.10599838, 1012.28694149,  315.42981172,
        179.16100503,  150.63485842,  132.14373783,  118.24336272,
        107.3211653 ]
    y1_opt = [1.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 1, 1, 1 ,
       1, 1]
    y2_opt = [0        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 1, 1, 1 ,
       1, 1]
    for i in range(N):
        w   = []
        w0  = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        discrete = []
        t = 0
# "Lift" initial conditions
        Xk = SX.sym('X0', 3)
        w += [Xk]
        ff = np.array(Fk11).squeeze()
        lbw += [*ff]
        ubw += [*ff]
        w0 += [*ff]
        discrete += [False] * 3
        x_plot = []
        x_plot += [Xk]
        y_plot = []
        u_plot = []


        dt_last = 0#0.05*i
        x11[0] = 1.
        x12[0] = 0.
# Formulate the NLP
        N1 = N1 - 1
        for k in range(N1):
# New NLP variable for the control

            yk = SX.sym('y_' + str(k), 2)
            w += [yk]
            lbw += [1., 1.]
            ubw += [1., 1.]
            w0 += [1., 1.]
            discrete += [False] * 2
            y_plot += [yk]

            # g += [Xk[1]-500. - 100000*(1-yk[0])]
            # lbg += [-10000000.]
            # ubg += [0.]
            # g += [-Xk[0] + 10. - 1000 * (1 - yk[1])]
            # lbg += [-100000.]
            # ubg += [0.]
            #
            # g += [Xk[1] - 500. + 100000 * (yk[0])]
            # lbg += [0.]
            # ubg += [10000000.]
            # g += [-Xk[0] + 10. + 1000 * (yk[1])]
            # lbg += [0.]
            # ubg += [1000000.]

            Uk = SX.sym('U_' + str(k), 2)
            w += [Uk]
            lbw += [120, 0.]
            ubw += [400, 40.]
            w0 += [240, 40.]
            discrete += [False] * 2
            u_plot += [Uk]

# Integrate till the end of the interval
            w, lbw, ubw, w0, g, lbg, ubg, Xk, discrete = perform_orthogonal_collocation(dc, 3, w, lbw, ubw, w0, [0,0,0],
                                                                             [inf,inf,inf], D, Xk, k, yk, C, f1, Uk, 240/12, g, lbg, ubg, x_plot, discrete)#F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)
            #Fk = F1(x0=Xk, p=Uk, y=yk)

            #Xk_end = Fk['xf']
    #    J = J + Fk['qf']

    # New NLP variable for state at end of interval
            #Xk = SX.sym('X_' + str(k + 1), 3)
            #w += [Xk]
            #lbw += [-inf, -inf, -inf]
            #ubw += [inf, inf, inf]
            #w0 += [*ff]
            #x_plot += [Xk]
            #discrete += [False] * 3

# Add equality constraint
            #g += [Xk_end - Xk]
            #lbg += [0, 0, 0]
            #ubg += [0, 0, 0]

#        if k > 0:
#            g += [DTk - 0.05]
#            lbg += [0]
#            ubg += [0]
#            dt_last = DTk
# Create an NLP solver
#        g += [t - T-0.05*i]  # [sum(w[2::3])]
#        lbg += [0]
#        ubg += [0]
            g += [Xk[1] - 800.]
            lbg += [-inf]
            ubg += [0]
            g += [Xk[2] - 0.011 * Xk[0]]
            lbg += [-inf]
            ubg += [0]
            if k > 0 :
                J += (w[11*k+2] - w[11*k-9]).T @ np.diagflat([3.125e-8, 3.125e-006]) @ (w[11*k+2] - w[11*k-9])
#(w[11*k+2] - w[11*k-9]).T @ np.diagflat([3.125e-8, 3.125e-006]) @ (w[11*k+2] - w[11*k-9])

        J +=  -Xk[2]
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*y_plot)], ['w'], ['x','u','y'])
        solver = nlpsol('solver', 'ipopt', prob)#'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#

# Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        x_opt, u_opt, y_opt =  trajectories(sol['x'])
# Plot the solution
        x1_opt = w_opt[0::7]
        x2_opt = w_opt[1::7]
        u1_opt = w_opt[5::7]
        u2_opt = w_opt[6::7]
        y_ex = [0., 0.]
        if Fk11[1]<= 500:
            y_ex[0] = 1.
        if Fk11[0]>= 10:
            y_ex[1] = 1.
        Fk1 = F1(x0=Fk11, p=vertcat(np.array(u_opt)[:,0], [1,1]))
        hisx11[kkk, :, i] = np.array(Fk11).reshape(3,)
        #, np.zeros(modparval.shape)))#p=np.array(u_opt)[:,0], y=y_ex) #I deleted the [0][0]
 #       Fk11 = stochastic_plant(Fk11, [u1_opt[0], u2_opt[0]], 0.05)
        Fk11 = Fk1['xf']# + np.random.multivariate_normal(np.zeros(3),
                        #                     [400.,1e5,1e-2]*diag(np.ones(nd))*1e-5)\
              # + np.array([400. * np.sin(t), 1e5 * np.sin(t), 1e-2 * np.sin(t)]) * 1e-5

        x12[i+1] = Fk11[1]
        x11[i+1] = Fk11[0]
       # #11[i] = u1_opt[0]
        #u12[i] = u2_opt[0]

        hisx11[kkk, :, i+1] = np.array(Fk11).reshape(3,)
        hisu11[kkk, i] = np.array(u_opt)[0,0]
        hisu12[kkk, i] = np.array(u_opt)[1,0]
        ind += 1
#plt.figure(1)



# plt.plot(ts, u1_opt, '-')
# plt.plot(ts, u2_opt, '-')
import mmap

ss = 0
for i in range(ind):
    with open('error_on_fail' + str(i) + '.txt', 'rb', 0) as file, \
            mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(b'Optimal Solution Found') != -1:
            ss += 1

print('converged:'+str(int(ss/ind*100))+'%')
sm12_mean = np.zeros(N+1)
sm11_mean = np.zeros(N+1)
sm12_std = np.zeros(N+1)
sm11_std = np.zeros(N+1)

for i in range(N+1):
    sm12_mean[i] = hisx12[:,i].mean()
    sm11_mean[i] = hisx11[:,i].mean()
    sm11_std[i] = hisx11[:,i].std()
    sm12_std[i] = hisx12[:,i].std()
plt.plot(ts, np.array(sm11_mean),'r--')
plt.fill_between(ts, np.array(sm11_mean) - np.array(sm11_std),
                 np.array(sm11_mean) + np.array(sm11_std),
                 label=r'$\sigma$')
plt.plot(ts, np.array(sm12_mean),'r--')
plt.fill_between(ts, np.array(sm12_mean) - np.array(sm12_std),
                 np.array(sm12_mean) + np.array(sm12_std),
                 label=r'$\sigma$')
#s = plt.imread('Figure_1-20.png')
#plt.imshow(s)
plt.show()
import pickle
import datetime
now = datetime.datetime.now()

pickle.dump( [hisx11, hisu11, hisu12], open('MPC-bio2'+str(now.date())+str(now.hour)+str(now.minute)+'.p', "wb" ))
print(Fk11)
