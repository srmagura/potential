import numpy as np

class Multivalue:
    """
    A grid function defined on the discrete boundary gamma that may have
    multiple values at a single node.

    The underlying data structure is the dictionary `_data`. It should be
    considered private.

    For each node in
    nodes, _data contains an (initially empty) list. The list
    should contain items that are dictionaries with keys 'value',
    'setype', and 'elen'. The current algorithm computes at most 2
    extension values at a given node, so reduce() with throw an
    AssertionError if there are 3 or more items in the list.
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self._data = {}

        for node in nodes:
            self._data[node] = []

    def __getitem__(self, node):
        return self._data[node]

    def __add__(self, mv2):
        """
        Return the sum of this multivalue grid function with another.

        mv2 --- the other Multivalue object

        The two multivalue objects are assumed to
         - be defined at the same nodes,
         - have the same number of values at a given node, and
         - the corresponding values of in two lists must have the
           same setype.
        """
        result = Multivalue(self.nodes)

        for node in self._data:
            for i in range(len(self[node])):
                dict1 = self[node][i]
                dict2 = mv2[node][i]

                assert dict1['setype'] == dict2['setype']

                dict_result = {
                    'elen': dict1['elen'],
                    'value': dict1['value'] + dict2['value'],
                    'setype': dict1['setype']
                }

                result[node].append(dict_result)

        return result

    def reduce(self):
        """
        Reduce the multiple-valued grid function to a single-valued
        grid function. Returns the single-valued grid function.

        At nodes where there are two values, a weighted average
        based on extension length (elen) is used.
        """
        # The single-valued grid function
        ext = np.zeros(len(self.nodes), dtype=complex)

        for l in range(len(self.nodes)):
            node = self.nodes[l]
            list_ = self._data[node]

            if len(list_) == 1:
                ext[l] = list_[0]['value']
            elif len(list_) == 2:
                elen0 = list_[0]['elen']
                value0 = list_[0]['value']

                elen1 = list_[1]['elen']
                value1 = list_[1]['value']

                # To prevent divide-by-zero
                if elen0 == 0 and elen1 == 0:
                    elen0 = elen1 = 1

                # Weighted average
                ext[l] = (elen1*value0 + elen0*value1) / (elen0 + elen1)

            else:
                # We should never have 3 or more values at one node
                assert False

        return ext

    def calc_rel_convergence(self, u1, u2):
        """
        Calculate the relative convergence for the sequence (self, u1, u2).
        See Solver.calc_rel_convergence for more information.

        This function is only used for debugging the extension test
        for PizzaSolver with relative convergence enabled.
        """
        diff12 = []
        diff01 = []

        for i, j in self.nodes:
            i1, j1 = i//2, j//2
            i2, j2 = i//4, j//4

            if i % 4 == 0 and j % 4 == 0:
                for l in range(len(u2[i2, j2])):
                    dict1 = None
                    dict2 = u2[i2, j2][l]

                    for ll in range(len(u1[i1, j1])):
                        _dict1 = u1[i1, j1][ll]
                        if _dict1['setype'] == dict2['setype']:
                            dict1 = _dict1
                            break

                    if dict1:
                        diff12.append(abs(dict1['value'] - dict2['value']))

            if i % 2 == 0 and j % 2 == 0:
                for l in range(len(u1[i1, j1])):
                    dict0 = None
                    dict1 = u1[i1, j1][l]

                    for ll in range(len(self[i, j])):
                        _dict0 = self[i, j][ll]
                        if _dict0['setype'] == dict1['setype']:
                            dict0 = _dict0
                            break

                    if dict0:
                        diff01.append(abs(dict0['value'] - dict1['value']))

        return np.log2(max(diff12) / max(diff01))
