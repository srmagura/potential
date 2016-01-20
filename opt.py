import numpy as np

def constrained_lstsq(A, b, C, d):
    """
    Solve Ax = b in the least squares sense, subject to the equality
    constraint Cx = d.

    Reference: http://stanford.edu/class/ee103/lectures/
        constrained-least-squares/constrained-least-squares_slides.pdf
    """
    n = A.shape[1]
    p = C.shape[0]
    M = np.zeros((n+p, n+p), dtype=complex)
    M[:n, :n] = 2*np.transpose(A).dot(A)
    M[n:, :n] = C
    M[:n, n:] = np.transpose(C)

    f = np.zeros((n+p), dtype=complex)
    f[:n] = 2*np.transpose(A).dot(b)
    f[n:] = d

    sol = np.linalg.solve(M, f)
    return sol[:n]
