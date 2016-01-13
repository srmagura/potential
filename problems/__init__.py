"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict

from .sine import Sine
from .ycosine import YCosine
from .sing_i import SingI_NoCorrection, SingI
from .sing_h import SingH_Hat, SingH_Parabola, SingH_Sine8, SingH_Sine4,\
    SingH_SineRange
from .sing_ih import SingIH_Sine8, SingIH_Hat, SingIH_Parabola, SingIH_Line
from .fbterm import FbTerm

problem_dict = OrderedDict((
    ('sine', Sine),
    ('ycosine', YCosine),

    ('sing-i-nc', SingI_NoCorrection),
    ('sing-i', SingI),

    ('sing-h-sine4', SingH_Sine4),
    ('sing-h-sine8', SingH_Sine8),
    ('sing-h-sine-range', SingH_SineRange),
    ('sing-h-hat', SingH_Hat),
    ('sing-h-parabola', SingH_Parabola),

    ('sing-ih-sine8', SingIH_Sine8),
    ('sing-ih-hat', SingIH_Hat),
    ('sing-ih-parabola', SingIH_Parabola),
    ('sing-ih-line', SingIH_Line),

    ('fbterm', FbTerm),
))

for name, problem in problem_dict.items():
    problem.name = name
