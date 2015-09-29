import numpy as np

def _get_ip_array(h, weights):
    r"""
    Get the SPD matrix M as that satisfies

    .. math:: \Vert v \Vert**2 = v^* M v,

    where the norm is the l2 norm.

    M is actually returned as an ndarray for convenience. The name
    ip_array comes from "inner product array".
    """
    # Scale the ip_array so that the average of the entries is h
    scaling_factor = len(weights) / sum(weights)
    return h * scaling_factor * np.diag(weights)

def get_ip_array__unweighted(h, nodes):
    return _get_ip_array(h, np.ones(len(nodes)))
