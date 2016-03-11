"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .smooth import Smooth_H
from .sing_h import H_Hat, H_Parabola, H_Sine8, H_SineRange, H_LineSine

problem_dict = OrderedDict((
    ('smooth-h', Smooth_H),
    ('h-sine8', H_Sine8),
    ('h-sine-range', H_SineRange),
    ('h-hat', H_Hat),
    ('h-parabola', H_Parabola),
    ('h-line-sine', H_LineSine),
))

for _name, _problem in problem_dict.items():
    _problem.name = _name
