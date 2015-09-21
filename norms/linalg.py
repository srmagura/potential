import numpy as np

def eval_norm(ip_array, v):
    """
    Evaluate the norm of v, where the norm is defined by `ip_array`.
    """
    return np.sqrt(np.vdot(v, ip_array.dot(v)))

def solve_var(A, b, ip_array):
    """
    Find the optimal solution to the overdetermined linear system
    Ax = b.

    This function returns the x that minimizes the residual in the
    sense of the norm defined by `ip_array`.

    A -- 2D numpy array, left-hand side
    b -- 1D numpy array, right-hand side
    ip_array - inner product array that defines the norm
    """
    A = np.matrix(A)
    b = np.matrix(b).T

    evals, U = np.linalg.eig(ip_array)
    sqrt_lambda = np.matrix(np.diag(np.sqrt(evals)))
    U = np.matrix(U)
    B = U * sqrt_lambda * U.getH()

    x = np.linalg.lstsq(B*A, B*b)[0]
    return np.ravel(x)
