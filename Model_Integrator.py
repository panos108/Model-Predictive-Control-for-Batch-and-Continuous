import numpy as np
import scipy.integrate as scp
import sdeint

''' In here we simulate the real model'''


def model_integration(params, initial_state, controls, dt):


    '''
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    dtime: float of duration of time step
    '''

    U1_u = controls['U1_u']
    U2_u = controls['U2_u']
    a_p = params['a_p']
    b_p = params['b_p']
    dtime = dt

    def SimpleMod(t, initial_state):

        #state vector
        y1_s = initial_state[0]
        y2_s = initial_state[1]

        # differential equations
        dev_y1 = -(U1_u+U1_u**2*a_p)*y1_s + U2_u
        dev_y2 = U1_u*y1_s*b_p - U2_u*y1_s# + np.exp(-.5*t)*np.sin(y1_s)*np.random.normal(0.0)*0.01
        return [dev_y1, dev_y2]

    ode = scp.ode(SimpleMod)
    ode.set_integrator('lsoda', nsteps=3000)
    # initial time always 0
    ode.set_initial_value(initial_state, 0.0)
    new_state = ode.integrate(ode.t + dtime)
    return new_state


def plant(params, initial_state, controls, dt):
    '''
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    dtime: float of duration of time step
    '''

    U1_u = controls['U1_u']
    U2_u = controls['U2_u']
    a_p = params['a_p']
    b_p = params['b_p']
    dtime = dt
   # a = (np.random.normal(0.0)*0.01+0.2)
    def SimpleMod(t, initial_state):
        # state vector
        y1_s = initial_state[0]
        y2_s = initial_state[1]

        # differential equations
        dev_y1 = -(U1_u + U1_u ** 2 * a_p) * y1_s + 0.5*U2_u*y2_s / (y1_s + y2_s)
        dev_y2 = U1_u * y1_s * b_p - 0.7 * U2_u * y1_s

        return np.array([dev_y1, dev_y2], dtype='float64')

    ode = scp.ode(SimpleMod)
    ode.set_integrator('lsoda', nsteps=3000)
    # initial time always 0
    ode.set_initial_value(initial_state, 0.0)
    new_state = list(ode.integrate(ode.t + dtime))
    return np.array(new_state)


def stochastic_plant(initial_state, controls, dt):
    '''
    params: dictionary of parameters passed to a model
    initial_state: numpy array of initial state
    controls: numpy array of control actions for this time step
    dtime: float of duration of time step
    '''

    U1_u = controls[0]
    U2_u = controls[1]
    a_p = 0.5
    b_p = 1
    dtime = dt
   # a = (np.random.normal(0.0)*0.01+0.2)
    def SimpleMod(initial_state, t):
        # state vector
        y1_s = initial_state[0]
        y2_s = initial_state[1]

        # differential equations
        dev_y1 = -(U1_u + U1_u ** 2 * a_p) * y1_s + 0.5*U2_u*y2_s / (y1_s + y2_s)
        dev_y2 = U1_u * y1_s * b_p - 0.7 * U2_u * y1_s

        return np.array([dev_y1, dev_y2], dtype='float64')

    def Stochastic(initial_state, t):

        return np.diag([0., abs(0.1*initial_state[0])**0.5])

    tspan = np.linspace(0.0, dtime, 300)
    new_states = sdeint.itoint(SimpleMod, Stochastic, np.array(initial_state), tspan)
    new_state = new_states[-1]

    return np.array(new_state)
