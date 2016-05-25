"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .smooth import SmoothH_Sine, SmoothH_E
from .sing_h_trace import Trace_Hat, Trace_Parabola,\
    Trace_Sine8, Trace_SineRange, Trace_LineSine
from .sing_h_shift import Shift_Hat, Shift_Parabola, Shift_LineSine

problem_dict = OrderedDict((
    ('smooth-h-sine', SmoothH_Sine),
    ('smooth-h-e', SmoothH_E),
    ('trace-sine8', Trace_Sine8),
    ('trace-sine-range', Trace_SineRange),
    ('trace-hat', Trace_Hat),
    ('trace-parabola', Trace_Parabola),
    ('trace-line-sine', Trace_LineSine),
    ('shift-hat', Shift_Hat),
    ('shift-parabola', Shift_Parabola),
    ('shift-line-sine', Shift_LineSine),
))

for _name, _problem in problem_dict.items():
    _problem.name = _name
