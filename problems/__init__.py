"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .sing_h import H_Hat, H_Parabola, H_Sine8, H_SineRange

problem_dict = OrderedDict((
    ('h-sine8', SingH_Sine8),
    ('h-sine-range', SingH_SineRange),
    ('h-hat', SingH_Hat),
    ('h-parabola', SingH_Parabola),
))

for name, problem in problem_dict.items():
    problem.name = name
