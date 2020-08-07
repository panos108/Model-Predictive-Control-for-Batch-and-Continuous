# GP NMPC problem setup
import numpy as np
from casadi import *


class Model_definition:

    def specifications(self):
        ''' Specify Problem parameters '''
        tf              = 240.      # final time
        nk              = 12        # sampling points
        dt              = tf/nk
        x0              = np.array([1.,150.,0.])
        Lsolver         = 'mumps'  #'ma97'  # Linear solver
        c_code          = False    # c_code
        shrinking_horizon = True

        return dt, x0, Lsolver, c_code, shrinking_horizon

    def DAE_system(self):
        # Define vectors with names of states
        states     = ['x','n','q']
        nd         = len(states)
        xd         = SX.sym('xd',nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = []
        na         = len(algebraics)
        xa         = SX.sym('xa',na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        set_points = []
        n_ref      = len(set_points)
        x_ref      = SX.sym('x_ref',n_ref)
        for i in range(n_ref):
            globals()[set_points[i]] = x_ref[i]



        # Define vectors with banes of input variables
        inputs     = ['L','Fn']
        nu         = len(inputs)
        u          = SX.sym("u",nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Define model parameter names and values
        modpar    = ['u_m', 'k_s', 'k_i', 'K_N', 'u_d', 'Y_nx', 'k_m', 'k_sq',
        'k_iq', 'k_d', 'K_Np']
        modparval = [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
        2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]

        nmp       = len(modpar)
        uncertainty = []#SX.sym('uncp', nmp)
        for i in range(nmp):
            globals()[modpar[i]] = SX(modparval[i])# + uncertainty[i])

        # Additive measurement noise
    #    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Additive disturbance noise
    #    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

        # Initial additive disturbance noise
    #    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3
        # Declare ODE equations (use notation as defined above)

        dx   = u_m * L/(L+k_s+L**2./k_i) * x * n/(n+K_N) - u_d*x
        dn   = - Y_nx*u_m* L/(L+k_s+L**2./k_i) * x * n/(n+K_N)+ Fn
        dq   = (k_m * L/(L+k_sq+L**2./k_iq) * x - k_d * q/(n+K_Np))# * (sign(499.9999 - n)+1)/2 * (sign(x - 10.0001)+1)/2

              # * (tanh(1000*(500. - n))+1)/2 * (tanh(1000*(x - 10.000))+1)/2# * (sign(499.9999 - n)+1)/2 * (sign(x - 10.0001)+1)/2

        ODEeq =  [dx, dn, dq]

        # Declare algebraic equations
        Aeq = []

        # Define objective to be minimized
        t     = SX.sym('t')
        Obj_M = Function('mayer', [xd, x_ref, u], [-q])  # Mayer term
        Obj_L = Function('lagrange', [xd, x_ref, u], [0.])  # Lagrange term
        Obj_D = Function('Discrete', [xd, x_ref, u], [0.])  # Lagrange term

        R           = np.diagflat([3.125e-8, 3.125e-006])                         # Weighting of control penality

        # Define control bounds
        u_min = np.array([120., 0.])
        u_max = np.array([400., 40.])
        x_min = np.array([0.]*nd)
        x_max = np.array([np.inf]*nd)
        # Define constraint functions g(x) <= 0
        gequation = vertcat(n - 800., q - 0.011 * x)
        ng = SX.size(gequation)[0]
        gfcn = Function('gfcn', [xd, xa, u], [gequation])

        return xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states,\
               algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R


    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: NaN
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states, \
        algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R = self.DAE_system()

        dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u, uncertainty),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
        opts = {'tf': 240/12}  # interval length
        F = integrator('F', 'idas', dae, opts)

        return F




class simple_CSTR:

    def specifications(self):
        ''' Specify Problem parameters '''
        dt                = 0.01      # final time
        x0                = np.array([0.])
        Lsolver           = 'mumps'  #'ma97'  # Linear solver
        c_code            = False    # c_code
        shrinking_horizon = False
        return dt, x0, Lsolver, c_code, shrinking_horizon

    def DAE_system(self):
        # Define vectors with names of states
        states     = ['x']
        nd         = len(states)
        xd         = SX.sym('xd',nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = []
        na         = len(algebraics)
        xa         = SX.sym('xa',na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        set_points = ['x_ref']
        n_ref      = len(set_points)
        x_ref      = SX.sym('x_ref',n_ref)
        for i in range(n_ref):
            globals()[set_points[i]] = x_ref[i]

        # Define vectors with banes of input variables
        inputs     = ['u']
        nu         = len(inputs)
        u          = SX.sym("u",nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Define model parameter names and values
        modpar    = ['u_scale', 'V', 'k', 'C0']
        modparval = [3000., 5000., 2., 1.]

        nmp       = len(modpar)
        uncertainty = []#SX.sym('uncp', nmp)
        for i in range(nmp):
            globals()[modpar[i]] = SX(modparval[i])# + uncertainty[i])

        dx   = u_scale * u / V * (C0 -x) - k * x **3

        ODEeq =  [dx]

        # Declare algebraic equations
        Aeq = []

        # Define objective to be minimized
        t     = SX.sym('t')
        Obj_M = Function('mayer', [xd, x_ref, u], [0.])  # Mayer term
        Obj_L = Function('lagrange', [xd, x_ref, u], [(x-x_ref)**2])  # Lagrange term
        Obj_D = Function('Discrete', [xd, x_ref, u], [0.])  # Lagrange term

        R           = np.eye(nu)#np.diagflat([3.125e-8, 3.125e-006])                         # Weighting of control penality

        # Define control bounds
        u_min = np.array([0.])
        u_max = np.array([1.])
        x_min = np.array([0.]*nd)
        x_max = np.array([np.inf]*nd)
        # Define constraint functions g(x) <= 0
        gequation = vertcat(x - 1.)
        ng = SX.size(gequation)[0]
        gfcn = Function('gfcn', [xd, x_ref, u], [gequation])

        return xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states,\
               algebraics, inputs, nd, na, nu,n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R



    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: NaN
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """
        dt, x0, Lsolver, c_code, specifications = self.specifications()

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states, \
        algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R = self.DAE_system()

        dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u, uncertainty),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
        opts = {'tf': dt}  # interval length
        F = integrator('F', 'idas', dae, opts)

        return F


class different_CSTR:

    def specifications(self):
        ''' Specify Problem parameters '''
        dt                = 0.01      # final time
        x0                = np.array([0., 0])
        Lsolver           = 'mumps'  #'ma97'  # Linear solver
        c_code            = False    # c_code
        shrinking_horizon = False

        return dt, x0, Lsolver, c_code, shrinking_horizon

    def DAE_system(self):
        # Define vectors with names of states
        states     = ['x1', 'x2']
        nd         = len(states)
        xd         = SX.sym('xd',nd)
        for i in range(nd):
            globals()[states[i]] = xd[i]

        # Define vectors with names of algebraic variables
        algebraics = []
        na         = len(algebraics)
        xa         = SX.sym('xa',na)
        for i in range(na):
            globals()[algebraics[i]] = xa[i]

        set_points = ['x_ref']
        n_ref      = len(set_points)
        x_ref      = SX.sym('x_ref',n_ref)
        for i in range(n_ref):
            globals()[set_points[i]] = x_ref[i]

        # Define vectors with banes of input variables
        inputs     = ['u']
        nu         = len(inputs)
        u          = SX.sym("u",nu)
        for i in range(nu):
            globals()[inputs[i]] = u[i]

        # Define model parameter names and values
        modpar    = ['u_scale', 'V', 'k', 'C0']
        modparval = [3000., 5000., 2., 1.]

        nmp       = len(modpar)
        uncertainty = []#SX.sym('uncp', nmp)
        for i in range(nmp):
            globals()[modpar[i]] = SX(modparval[i])# + uncertainty[i])

        dx1   = u_scale * u / V * (C0 -x1) - k * x1
        dx2  = -u_scale * u/  V * x2        + k * x1
        ODEeq =  [dx1, dx2]

        # Declare algebraic equations
        Aeq = []

        # Define objective to be minimized
        t     = SX.sym('t')
        Obj_M = Function('mayer', [xd, x_ref, u], [0.])  # Mayer term
        Obj_L = Function('lagrange', [xd, x_ref, u], [(x1-x_ref)**2])  # Lagrange term
        Obj_D = Function('Discrete', [xd, x_ref, u], [0.])  # Lagrange term

        R           = 0.1*np.eye(nu)#np.diagflat([3.125e-8, 3.125e-006])                         # Weighting of control penality

        # Define control bounds
        u_min = np.array([0.])
        u_max = np.array([1.])
        x_min = np.array([0.]*nd)
        x_max = np.array([np.inf]*nd)
        # Define constraint functions g(x) <= 0
        gequation = vertcat(x1 - 1.)
        ng = SX.size(gequation)[0]
        gfcn = Function('gfcn', [xd, x_ref, u], [gequation])

        return xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states,\
               algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R



    def integrator_model(self):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: NaN
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """
        dt, x0, Lsolver, c_code, _ = self.specifications()

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states, \
        algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, Obj_L, Obj_D, R = self.DAE_system()

        dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u, uncertainty),
               'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
        opts = {'tf': dt}  # interval length
        F = integrator('F', 'idas', dae, opts)

        return F
