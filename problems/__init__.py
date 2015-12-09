"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from collections import OrderedDict
from enum import Enum

RegularizeBc = Enum('RegularizeBc', 'none fft known')

from .sine import Sine
from .ycosine import YCosine
from .sing_i import SingINoCorrection, SingI
from .sing_h import SingH_Hat, SingH_Parabola, SingH_Sine8, SingH_Sine4,\
    SingH_SineRange
#from .sing_ih import SingIH_FFT_Sine, SingIH_FFT_Hat, SingIH_FFT_Parabola,\
#    SingIH_FFT_Line
from .fbterm import FbTerm

problem_dict = OrderedDict((
    ('sine', Sine),
    ('ycosine', YCosine),

    ('sing-i-nc', SingINoCorrection),
    ('sing-i', SingI),

    ('sing-h-sine4', SingH_Sine4),
    ('sing-h-sine8', SingH_Sine8),
    ('sing-h-sine-range', SingH_SineRange),
    ('sing-h-hat', SingH_Hat),
    ('sing-h-parabola', SingH_Parabola),

    #('sing-ih-fft-sine', SingIH_FFT_Sine),
    #('sing-ih-fft-hat', SingIH_FFT_Hat),
    #('sing-ih-fft-parabola', SingIH_FFT_Parabola),
    #('sing-ih-fft-line', SingIH_FFT_Line),

    ('fbterm', FbTerm),
))
