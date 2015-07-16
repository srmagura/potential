"""
Test the Multivalue data structure defined in ps.multivalue.
"""
import unittest
import numpy as np

from ps.multivalue import Multivalue

def is_equal(mv1, mv2):
    """
    Compares this Multivalue with another, checking the data for
    equality up to a certain tolerance. 'elen' is not compared.
    """
    tol = 1e-13

    if mv1.nodes != mv2.nodes:
        return False

    for node in mv1._data:
        for i in range(len(mv1[node])):
            dict1 = mv1[node][i]
            dict2 = mv2[node][i]

            if (dict1['setype'] != dict2['setype'] or
                abs(dict1['value'] - dict2['value']) > tol):
                return False

    return True

class TestMultivalue(unittest.TestCase):

    def setUp(self):
        """ Create Multivalue mv1, which is used in multiple tests. """
        self.nodes = [(0, 0), (0, 1)]
        self.mv1 = Multivalue(self.nodes)

        self.mv1[(0, 0)].extend([
            {
                'elen': 1,
                'setype': 0,
                'value': 0.5,
            },
            {
                'elen': 1.5,
                'setype': 1,
                'value': 0.75,
            },
        ])

        self.mv1[(0, 1)].append({
            'elen': 1,
            'setype': 0,
            'value': 2,
        })

    def test_getitem(self):
        """ Test Multivalue.__getitem__() """
        self.assertEqual(self.mv1[(0, 0)][0]['setype'], 0)

        with self.assertRaises(KeyError):
            self.mv1[(2, 2)]

    def test_add(self):
        """ Test Multivalue.__add__() """
        mv2 = Multivalue(self.nodes)

        mv2[(0, 0)].extend([
            {
                'elen': 1,
                'setype': 0,
                'value': 3.5,
            },
            {
                'elen': 1,
                'setype': 1,
                'value': 4,
            },
        ])
        mv2[(0, 1)].append({
            'elen': 1,
            'setype': 0,
            'value': -3,
        })

        expected = Multivalue(self.nodes)
        expected[(0, 0)].extend([
            {
                'setype': 0,
                'value': 4,
            },
            {
                'setype': 1,
                'value': 4.75,
            },
        ])
        expected[(0, 1)].append({
            'setype': 0,
            'value': -1,
        })

        self.assertTrue(is_equal(expected, self.mv1 + mv2))

        # Addition should be commutative
        self.assertTrue(is_equal(expected, mv2 + self.mv1))

    def test_reduce_mv1(self):
        """ Compare self.mv1.reduce() to expected result. """
        expected = np.array([0.6, 2])
        self.assertTrue(np.allclose(expected, self.mv1.reduce()))

    def test_reduce0(self):
        """
        Test reduce() when two values at the same node have elen=0.
        """
        mv = Multivalue([(0, 0)])
        mv[(0, 0)].extend([
            {
                'elen': 0,
                'setype': 0,
                'value': 2,
            },
            {
                'elen': 0,
                'setype': 1,
                'value': 4.75,
            },
        ])

        reduced_value = mv.reduce()[0]
        self.assertEqual(reduced_value, (2+4.75)/2)
