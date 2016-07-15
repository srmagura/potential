import numpy as np
import itertools as it
from scipy.interpolate import CloughTocher2DInterpolator

import matlab
import matlab.engine

from chebyshev import get_chebyshev_roots

def my_print(x):
    print('(polarfd) ' + str(x))

def get_z_interp(problem, R1, chebyshev_functions):
    """
    Return an interpolating function that approximates z

    R1 -- value of r up to which the interpolating function must be
        accurate
    chebyshev_manager --
    """
    k = problem.k
    a = problem.a
    nu = problem.nu

    R0 = .05
    match_r = .20

    N = 16
    Nlam = 10

    dim0 = (N+1)**2
    dim = dim0 + Nlam

    my_print('N = {}'.format(N))
    my_print('R0 = {}'.format(R0))
    my_print('R1 = {}'.format(R1))
    my_print('match_r = {}'.format(match_r))

    hr = (R1-R0)/N
    hth = (2*np.pi - a) / N

    def get_r(m):
        return R0 + m*hr

    def get_th(l):
        return a + l*hth

    def get_index(m, l):
        return m*(N+1) + l

    def get_index_b(J):
        return dim0 + J

    eval_g_inv = chebyshev_functions['eval_g_inv']
    eval_g = chebyshev_functions['eval_g']
    eval_B = chebyshev_functions['eval_B']

    # Calculate Chebyshev coefficients of true boundary data
    #t_roots = get_chebyshev_roots(1024)
    #boundary_data = np.zeros(len(t_roots))

    #for i in range(len(t_roots)):
    #    th = eval_g(t_roots[i])
    #    boundary_data[i] = eval_u(R1, th)

    #true_lam = np.polynomial.chebyshev.chebfit(t_roots, boundary_data, Nlam)
    #my_print('First chebyshev coef not included:', true_lam[-1])

    #true_lam = true_lam[:-1]

    rhs = []

    row = 0

    data = []
    row_ind = []
    col_ind = []

    # Set BC's on wedge
    # Be careful not to have any overlap when setting
    # boundary and initial conditions
    for m in range(N+1):
        r = get_r(m)

        i = get_index(m, 0)

        data.append(1)
        row_ind.append(row)
        col_ind.append(i)

        rhs.append(problem.eval_bc(r, 2))
        row += 1

        i = get_index(m, N)
        data.append(1)
        row_ind.append(row)
        col_ind.append(i)

        rhs.append(problem.eval_bc(r, 1))
        row += 1

    # Match Chebyshev series to data on wedge
    for l in (0, N):
        data.append(1)
        row_ind.append(row)
        col_ind.append(get_index(N, l))

        th = get_th(l)

        for J in range(Nlam):
            data.append(-eval_B(J, th))
            row_ind.append(row)
            col_ind.append(get_index_b(J))

        rhs.append(0)
        row += 1

    # Solution = 0 near tip
    for m in range(1, N):
        if get_r(m) > match_r and m > 3:
            break

        for l in range(1, N):
            data.append(1)
            row_ind.append(row)
            col_ind.append(get_index(m, l))

            rhs.append(0)
            row += 1


    for l in range(1, N):
        # Set BC near origin
        i = get_index(0, l)

        data.append(1)
        row_ind.append(row)
        col_ind.append(i)

        rhs.append(0)
        row += 1

        # Set BC at arc
        data.append(1)
        row_ind.append(row)
        col_ind.append(get_index(N, l))

        th = get_th(l)
        for J in range(Nlam):
            data.append(-eval_B(J, th))
            row_ind.append(row)
            col_ind.append(get_index_b(J))

        rhs.append(0)
        row += 1

    # Finite difference scheme
    for m in range(1, N):
        for l in range(1, N):
            r_2 = get_r(m-1)
            r_1 = get_r(m-1/2)
            r = get_r(m)
            r1 = get_r(m+1/2)
            r2 = get_r(m+1)

            local = np.zeros([3, 3])

            # 4th order
            local[-1, -1] += (
                -hr/(24*hth**2*r*r_2**2) + 1/(12*hth**2*r_2**2)
                + r_1/(12*hr**2*r)
            )

            local[0, -1] += (
                hr**2/(12*hth**2*r**4)
                + k**2/12 + 5/(6*hth**2*r**2) - r1/(12*hr**2*r)
                - r_1/(12*hr**2*r)
            )

            local[1, -1] += (
                hr/(24*hth**2*r*r2**2) + 1/(12*hth**2*r2**2)
                + r1/(12*hr**2*r)
            )


            local[-1, 0] += (
                -hr*k**2/(24*r) - hr/(12*r**3)
                + hr/(12*hth**2*r*r_2**2) + k**2/12 - 1/(6*hth**2*r_2**2)
                + 5*r_1/(6*hr**2*r)
            )

            local[0, 0] += (hr**2*k**2/(12*r**2)
                - hr**2/(6*hth**2*r**4) + 2*k**2/3
                - 5/(3*hth**2*r**2) - 5*r1/(6*hr**2*r)
                - 5*r_1/(6*hr**2*r)
            )

            local[1, 0] += (
                hr*k**2/(24*r) + hr/(12*r**3)
                - hr/(12*hth**2*r*r2**2) + k**2/12 - 1/(6*hth**2*r2**2)
                + 5*r1/(6*hr**2*r)
            )

            local[-1, 1] += (
                -hr/(24*hth**2*r*r_2**2)
                + 1/(12*hth**2*r_2**2) + r_1/(12*hr**2*r)
            )

            local[0, 1] += (hr**2/(12*hth**2*r**4)
                + k**2/12 + 5/(6*hth**2*r**2) - r1/(12*hr**2*r)
                - r_1/(12*hr**2*r)
            )

            local[1, 1] += (
                hr/(24*hth**2*r*r2**2)
                + 1/(12*hth**2*r2**2) + r1/(12*hr**2*r)
            )

            th = get_th(l)
            f = problem.eval_f_polar(r, th)
            d_f_r = problem.eval_d_f_r(r, th)
            d2_f_r = problem.eval_d2_f_r(r, th)
            d2_f_th = problem.eval_d2_f_th(r, th)

            # 4th order
            rhs.append(d2_f_r*hr**2/12
                + d2_f_th*hth**2/12
                + d_f_r*hr**2/(12*r)
                + f*hr**2/(12*r**2)
                + f
            )

            for dm, dl in it.product((-1, 0, 1), repeat=2):
                m1 = m + dm
                l1 = l + dl

                data.append(local[dm, dl])
                row_ind.append(row)
                col_ind.append(get_index(m1, l1))

            row += 1

    eng = matlab.engine.start_matlab()

    eng.workspace['row_ind'] = matlab.double([i+1 for i in row_ind])
    eng.workspace['col_ind'] = matlab.double([j+1 for j in col_ind])
    eng.workspace['data'] = matlab.double(data)
    eng.workspace['rhs'] = matlab.double(rhs)

    u = eng.eval('sparse(row_ind, col_ind, data) \ rhs.\'')
    u = np.array(u._data)

    eng.quit()

    my_print('System shape: {} x {}'.format(max(row_ind), max(col_ind)))

    # Measure error on arc
    error = []
    for m in range(N+1):
        for l in range(N+1):
            exp = problem.eval_expected_polar(get_r(m), get_th(l))
            error.append(abs(u[get_index(m, l)] - exp))

    my_print('Error on arc: {}'.format(np.max(error)))

    import sys; sys.exit(0)

    # Create interpolating function
    points = np.zeros((dim0, 2))
    data = np.zeros(dim0)

    for m in range(N+1):
        for l in range(N+1):
            i = get_index(m, l)
            points[i, 0] = get_r(m)
            points[i, 1] = get_th(l)
            data[i] = u[i]

    interpolater = CloughTocher2DInterpolator(points, data)

    # Measure error on perturbed boundary
    interpolater_boundary = []
    u_boundary = []

    for th in np.arange(a, 2*np.pi, .01):
        r = problem.boundary.eval_r(th)
        interpolater_boundary.append(interpolater(r, th))
        u_boundary.append(eval_u(r, th))

    error = np.max(np.abs(np.array(interpolater_boundary) - u_boundary))
    my_print('Error on perturbed boundary: {}'.format(error))

    return interpolater
