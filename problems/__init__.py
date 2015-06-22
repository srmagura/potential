'''
Defines choices for the problem command-line argument.
'''
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .sing_i import SingINoCorrection, SingI
from .sing_h import SingHHat, SingHParabola, SingHSine
from .sing_ih import SingIH_FFT_Line, SingIH_FFT_Hat

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'sing-i-nc': SingINoCorrection,
    'sing-i': SingI,
    'sing-h-hat': SingHHat,
    'sing-h-parabola': SingHParabola,
    'sing-h-sine': SingHSine,
    'sing-ih-fft-line': SingIH_FFT_Line,
    'sing-ih-fft-hat': SingIH_FFT_Hat,
}
