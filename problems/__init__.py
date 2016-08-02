"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .smooth import Smooth_Sine, Smooth_E, Smooth_YCos
from .sing_h_trace import Trace_Hat, Trace_Parabola,\
    Trace_Sine8, Trace_SineRange, Trace_LineSine
from .sing_h_shift import Shift_Hat, Shift_Parabola, Shift_LineSine
from .sing_ih import I_Bessel #, I_Bessel3, IH_Bessel_Line  #IH_Bessel_Quadratic,
from .sing_ihz import IZ_Bessel, IHZ_Bessel_Line

problem_dict = OrderedDict((
    ('smooth-sine', Smooth_Sine),
    ('smooth-e', Smooth_E),
    ('smooth-ycos', Smooth_YCos),
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
    #('ih-bessel-line', IH_Bessel_Line),
    #('ih-bessel-quadratic', IH_Bessel_Quadratic),
))

for _name, _problem in problem_dict.items():
    _problem.name = _name
