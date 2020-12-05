from casadi import *
import numpy as np
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
from OrthogonalCollocation import construct_polynomials_basis

class MPC:

    def __init__(self, Model_Def, horizon, collocation_degree = 4, penalize_u=False):
        self.Model_def = Model_Def()           # Take class of the dynamic system
        self.dc        = collocation_degree    # Define the degree of collocation
        self.N         = horizon               # Define the Horizon of the problem
        # FIXME  Add different horizon for control and prediction
        dt, x0, Lsolver, c_code, self.shrinking_horizon = self.Model_def.specifications()
        self.dt        = dt


        xd, _, u, _, ODEeq, _, self.u_min, self.u_max, self.x_min, self.x_max, _, \
        _, _, self.nd, _, self.nu, self.n_ref, _, _, self.ng, self.gfcn, \
        self.Obj_M, self.Obj_L, self.Obj_D, self.R = self.Model_def.DAE_system() # Define the System
        self.penalize_u = penalize_u

        self.f = Function('f1', [xd, u], [vertcat(*ODEeq)])
        # Define options for solver
        opts = {}
        opts["expand"] = True
        opts["ipopt.print_level"] = 0
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.tol"] = 1e-8
        opts["calc_lam_p"] = False
        opts["calc_multipliers"] = False
        opts["ipopt.print_timing_statistics"] = "no"
        opts["print_time"] = False
        self.opts = opts

        self.MPC_construct()




    def MPC_construct(self):
        """
        ODEeq: Is the system of ODE
        gcn  : Is the inequality constraints
        ObjM : Is the mayer term of DO
        ObjL : Is the Lagrange term of DO
        Obj_D: The a discretized objective for each time
        :return:
        """
        N  = self.N
        dc = self.dc
        dt = self.dt


        C, D, B = construct_polynomials_basis(dc, 'radau')
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        t = 0
        # "Lift" initial conditions
        shrink  = SX.sym('p_shrink', self.N)
        x_ref   = SX.sym('p_ref', self.n_ref)
        X_0     = SX.sym('p_x', self.nd)  #This is a parameter that defines the Initial Conditions
        Xk      = SX.sym('X0', self.nd)
        w      += [Xk]
        lbw    += [*self.x_min]
        ubw    += [*self.x_max]
        w0     += [*self.x_min]
        g      += [Xk - X_0]
        lbg    += [*np.zeros([self.nd])]
        ubg    += [*np.zeros([self.nd])]
        x_plot  = []
        x_plot += [Xk]
        u_plot  = []
        if self.penalize_u:
            U_past  = SX.sym('p_u', self.nu)  #This is a parameter that defines the Initial Conditions
            prev    = U_past
        for i in range(N):
        # Formulate the NLP
        # New NLP variable for the control

            Uk   = SX.sym('U_' + str(i), self.nu)
            if self.penalize_u:
                J += (Uk-prev).T @ self.R @ (Uk - prev) * shrink[i]
                prev = Uk
            w += [Uk]
            lbw += [*self.u_min]
            ubw += [*self.u_max]
            w0 += [*self.u_min]
            u_plot += [Uk]

# Integrate till the end of the interval
            w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, J = self.perform_orthogonal_collocation(dc, self.nd, w, lbw, ubw, w0,
                                                 self.x_min, self.x_max,
                                                 D, Xk, i, C, self.f, Uk, dt,
                                                 g, lbg, ubg, shrink[i], x_plot, B, J, x_ref)#F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)
            for ig in range(self.ng):
                g   += [self.gfcn(Xk, x_ref,    Uk)[ig]*shrink[i]]
                lbg += [-inf]
                ubg += [0.]
            J+= self.Obj_D(Xk, x_ref,  Uk) * shrink[i]
        J +=  self.Obj_M(Xk, x_ref,  Uk)
        if self.penalize_u:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [U_past]
            p += [shrink]
            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [shrink]
            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}

        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot)], ['w'], ['x','u'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg

    def solve_MPC(self, x, ref=None, u=None, t=0.):

        if self.n_ref>0:
            p0 = np.concatenate((x, np.array([ref]).reshape((-1,))))
        else:
            p0 = x

        if self.shrinking_horizon:
            if t==0.:
                shrink = np.ones([self.N])
                self.steps = self.N
            else:
                shrink = np.concatenate((np.ones([self.steps]), np.zeros([self.N-self.steps])))
        else:
            shrink = np.ones([self.N])

        if self.penalize_u:
            p0 = np.concatenate((p0,u))


        p0 = np.concatenate((p0, shrink))

        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                     p=p0)
        w_opt = sol['x'].full().flatten()
        x_opt, u_opt = self. trajectories(sol['x'])
        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print('Opt failed')
        if self.shrinking_horizon:
            self.steps += - 1
        self.obj = sol['f'].full().flatten()
        return u_opt, x_opt, w_opt

    def perform_orthogonal_collocation(self,d, nx, w, lbw, ubw, w0, lbx, ubx, D, Xk, s, C, f, Uk,
                                       h, g, lbg, ubg, shrink, x_plot, B, J, x_ref):
        """

        :return:
        """
        Xc = []

        for j in range(d):
            Xkj = SX.sym('X_' + str(s) + '_' + str(j), nx)
            Xc += [Xkj]
            w += [Xkj]
            lbw.extend(lbx)
            ubw.extend(ubx)
            w0.extend([1.] * nx)
            x_plot+= [Xkj]
        # Loop over collocation points
        Xk_end = D[0] * Xk

        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj = f(Xc[j - 1], Uk) * shrink  #
            g += [(h * fj - xp)]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)

            for ig in range(self.ng):
                g   += [self.gfcn(Xc[j-1], x_ref, Uk)[ig]*shrink]
                lbg += [-inf]
                ubg += [0.]
            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]
        #            if int(j1) < np.shape(t_meas)[0]:
        #                if np.real(k * T / N) == t_meas[j1]:
        #                    count[k] = 1
        #                    j1 += 1
        # Add contribution to quadrature function
            qj = self.Obj_L(Xc[j - 1], x_ref,Uk) * shrink  #

            J += B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(s + 1), nx)
        w += [Xk]
        lbw.extend(lbx)
        ubw.extend(ubx)
        w0.extend([0] * nx)

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg.extend([0.] * nx)
        ubg.extend([0.] * nx)

        return w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, J


class cosntract_history:
    def __init__(self, model, N, store_u = True, set_point0 = 0.):
        #Define self vars
        dt, x0, Lsolver, c_code, specifications = model.specifications()

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states, \
        algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, \
        Obj_L, Obj_D, R = model.DAE_system()

        self.model   = model           # The model defined in terms of casadi
        self.N       = N               # Number of past data
        self.store_u = store_u
        self.nx      = nd
        self.nu      = nu
        self.nsp     = len(set_point0)
        self.u_min   = u_min
        self.u_max   = u_max
        state_0, e_sp0 = x0, x0-set_point0#model.reset(set_point0)
        # initialize history
        history_x = np.array([*state_0]*N).reshape((-1,1))
        history_sp = np.array([*e_sp0]*N).reshape((-1,1))

        if store_u:                  # If u are stored as history (simple RNN structure)
            history_u = np.array([0]*N*self.nu).reshape((-1,1))
            self.history = np.vstack((history_x,history_sp,history_u))
            self.size_states = N * (self.nu + self.nx + self.nsp)
        else:
            self.history = np.vstack((history_x,history_sp))
            self.size_states = N * (self.nx+self.nsp)

        self.history = self.history.reshape((-1,))
        # start counting the past values
        self.past = 1


    def append_history(self, new_state, u, e_sp):

        if self.store_u:
            n = self.nx+self.nu + self.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp, u))
            self.history[:n] = aug_states

        else:
            n = self.nx+ self.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp))

            self.history[:n] = aug_states
        self.past +=1

        return self.history