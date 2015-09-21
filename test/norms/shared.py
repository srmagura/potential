def get_random_v(self, nodes):
    """
    Return a randomly-generated complex-valued function defined on
    `nodes`.
    """
    v_real = np.random.uniform(-1, 1, len(nodes))
    v_imag = np.random.uniform(-1, 1, len(nodes))
    return v_real + 1j*v_imag

def test_spd(self):
    """
    Verify that ip_array is Hermitian and positive definite.
    """
    nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
    ip_array = sobolev.get_ip_array(self.h, nodes, self.sa)
    ip_matrix = np.matrix(ip_array)

    self.assertTrue(np.array_equal(ip_matrix.getH(), ip_matrix))

    for i in range(25):
        v = self.get_random_v(nodes)
        if np.array_equal(v, np.zeros(len(v))):
            continue

        norm = np.vdot(v, ip_array.dot(v))
        self.assertTrue(norm > 0)

def test_norm(self):
    """
    Test that the two methods of evaluating the Sobolev norm
    eval_norm() and sobolev.get_ip_array() return the same results.
    """
    nodes = ((-1, 0), (0, 0), (1, 0), (0, 1))
    ip_array = sobolev.get_ip_array(self.h, nodes, self.sa)

    for i in range(10):
        v = self.get_random_v(nodes)
        norm1 = eval_norm(self.h, nodes, self.sa, v)
        norm2 = np.vdot(v, ip_array.dot(v))
        self.assertTrue(abs(norm1 - norm2) < 1e-13)
