"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .smooth import Smooth_Sine, Smooth_E, Smooth_YCos, Smooth_YCos2,\
    Smooth_One, Smooth_Two, Smooth_F0, Smooth_F1
from .sing_h_trace import Trace_Hat, Trace_Parabola,\
    Trace_Sine8, Trace_SineRange, Trace_LineSine
from .sing_h_shift import Shift_Hat, Shift_Parabola, Shift_LineSine
from .sing_ih import I_Bessel
from .sing_ihz import IZ_Bessel, IHZ_Bessel_Line, IHZ_Bessel_Quadratic

problem_dict = OrderedDict((
    ('smooth-sine', Smooth_Sine),
    ('smooth-e', Smooth_E),
    ('smooth-ycos', Smooth_YCos),
    ('smooth-ycos2', Smooth_YCos2),
    ('smooth-one', Smooth_One),
    ('smooth-two', Smooth_Two),
    ('smooth-f0', Smooth_F0),
    ('smooth-f1', Smooth_F1),
    ('trace-sine8', Trace_Sine8),
    ('trace-sine-range', Trace_SineRange),
    ('trace-hat', Trace_Hat),
    ('trace-parabola', Trace_Parabola),
    ('trace-line-sine', Trace_LineSine),
    ('shift-hat', Shift_Hat),
    ('shift-parabola', Shift_Parabola),
    ('shift-line-sine', Shift_LineSine),
    ('i-bessel', I_Bessel),
    ('iz-bessel', IZ_Bessel),
    ('ihz-bessel-line', IHZ_Bessel_Line),
    ('ihz-bessel-quadratic', IHZ_Bessel_Quadratic),
))

for _name, _problem in problem_dict.items():
    _problem.name = _name
