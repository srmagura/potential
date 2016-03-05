"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .sing_h import H_Hat, H_Parabola, H_Sine8, H_SineRange, H_LineSine

problem_dict = OrderedDict((
    ('h-sine8', H_Sine8),
    ('h-sine-range', H_SineRange),
    ('h-hat', H_Hat),
    ('h-parabola', H_Parabola),
    ('h-line-sine', H_LineSine),
))

for name, problem in problem_dict.items():
    problem.name = name
