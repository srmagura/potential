from ps.multivalue import Multivalue

mv1 = Multivalue(None)
mv2 = Multivalue(None)

mv1.set((0, 0), 1, .2)
mv1.set((0, 1), 1, .3)
mv1.set((1, 0), 1, .4)

mv2.set((0, 1), 2, -.3)
mv2.set((1, 0), 1, .9)
mv2.set((-1, 0), 2, -.1)

result = mv1.add_multivalue(mv2)

assert result.get((0, 0), 1) == .2
assert result.get((0, 1), 1) == .3
assert result.get((0, 1), 2) == -.3
assert result.get((1, 0), 1) == 1.3
assert result.get((-1, 0), 2) == -.1

print('Passed.')
