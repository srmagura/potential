from numpy import pi, sqrt, cos, arccos as acos, sin

def eval_weight(t):
    return 1 / sqrt(1 - t**2)

def eval_T(J, t):
    return cos(J * acos(t))

def eval_d_T_t(J, t):
    return J*sin(J*acos(t))/sqrt(-t**2 + 1)

def eval_d2_T_t(J, t):
    return J*(J*cos(J*acos(t))/(t**2 - 1) + t*sin(J*acos(t))/(-t**2 + 1)**(3/2))

def eval_d3_T_t(J, t):
    return J*(-J**2*sin(J*acos(t))/(-t**2 + 1)**(3/2) - 3*J*t*cos(J*acos(t))/(t**2 - 1)**2 + 3*t**2*sin(J*acos(t))/(-t**2 + 1)**(5/2) + sin(J*acos(t))/(-t**2 + 1)**(3/2))

def eval_d4_T_t(J, t):
    return J*(J**3*cos(J*acos(t))/(t**2 - 1)**2 - 4*J**2*t*sin(J*acos(t))/(sqrt(-t**2 + 1)*(t**2 - 1)**2) + J**2*t*sin(J*acos(t))/((-t**2 + 1)**(3/2)*(t**2 - 1)) - J**2*t*sin(J*acos(t))/(-t**2 + 1)**(5/2) + 15*J*t**2*cos(J*acos(t))/(t**2 - 1)**3 - 4*J*cos(J*acos(t))/(t**2 - 1)**2 + 15*t**3*sin(J*acos(t))/(-t**2 + 1)**(7/2) + 9*t*sin(J*acos(t))/(-t**2 + 1)**(5/2))

def eval_dn_T_t(n, J, t):
    derivs = (
            eval_T, eval_d_T_t, eval_d2_T_t, eval_d3_T_t, eval_d4_T_t
    )

    return derivs[n](J, t)

def get_chebyshev_roots(J):
    return [cos(pi/2* (2*l-1)/J) for l in range(J, 0, -1)]
