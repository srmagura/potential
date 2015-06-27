"""
The `problems` package houses classes that define the boundary value problems
that the numerical algorithm can (hopefully) solve.

This module defines choices for the problem command-line argument.
"""
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .sing_i import SingINoCorrection, SingI
from .sing_h import SingH_FFT_Hat, SingH_FFT_Parabola, SingH_FFT_Sine
from .sing_ih import SingIH_FFT_Sine, SingIH_FFT_Hat, SingIH_FFT_Parabola,\
    SingIH_FFT_Line

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'sing-i-nc': SingINoCorrection,
    'sing-i': SingI,
    'sing-h-fft-sine': SingH_FFT_Sine,
    'sing-h-fft-hat': SingH_FFT_Hat,
    'sing-h-fft-parabola': SingH_FFT_Parabola,
    'sing-ih-fft-sine': SingIH_FFT_Sine,
    'sing-ih-fft-hat': SingIH_FFT_Hat,
    'sing-ih-fft-parabola': SingIH_FFT_Parabola,
    'sing-ih-fft-line': SingIH_FFT_Line,
}
