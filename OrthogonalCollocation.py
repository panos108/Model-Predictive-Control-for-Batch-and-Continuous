from casadi import *


def construct_polynomials_basis(d, poly_type):

    # Get collocation points
    tau_root = np.append(0, collocation_points(d, poly_type))

    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))

    # Coefficients of the continuity equation
    D = np.zeros(d + 1)

    # Coefficients of the quadrature function
    B = np.zeros(d + 1)

    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity
        # equation
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return C, D, B


def perform_orthogonal_collocation(d, nx, w, lbw, ubw, w0, lbx, ubx, D, Xk, s, C, f, Uk,
                                   h, g, lbg, ubg, shrink, x_plot):
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
        w0.extend([0] * nx)

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
        lbg.extend([0.] * nx)
        ubg.extend([0.] * nx)


        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]
    #            if int(j1) < np.shape(t_meas)[0]:
    #                if np.real(k * T / N) == t_meas[j1]:
    #                    count[k] = 1
    #                    j1 += 1
    # Add contribution to quadrature function
    #      J = J + B[j]*qj*h

    # New NLP variable for state at end of interval
    Xk = SX.sym('X_' + str(s + 1), nx)
    w += [Xk]
    lbw.extend(lbx)
    ubw.extend(ubx)
    w0.extend([0] * nx)

    x_plot += [Xk]
    # Add equality constraint
    g += [Xk_end - Xk]
    lbg.extend([0.] * nx)
    ubg.extend([0.] * nx)

    return w, lbw, ubw, w0, g, lbg, ubg, Xk
