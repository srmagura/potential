import numpy as np

def get_ip_array__weighted(h, weights):
    r"""
    Get the SPD matrix M that satisfies

    .. math:: \Vert v \Vert**2 = v^* M v,

    where the norm is the l2 norm.

    M is actually returned as an ndarray for convenience. The name
    ip_array comes from "inner product array".
    """
    # Scale the ip_array so that the average of the entries is h
    scaling_factor = len(weights) / sum(weights)
    return h * scaling_factor * np.diag(weights)

def get_ip_array__unweighted(h, n_nodes):
    return get_ip_array__weighted(h, np.ones(n_nodes))

def get_ip_array__radial(h, radii, weight_func):
    """
    Returns ip_array for a weighted l2 norm where the weights are
    a function of the polar radius only.

    radii -- array containing r / R for each node. Here r denotes the
        polar radius of the node, and R denotes the radius of the
        circle (domain).

    weight_func -- function mapping polar radius to weight
    """
    weights = [weight_func(r) for r in radii]
    return get_ip_array__weighted(h, weights)
