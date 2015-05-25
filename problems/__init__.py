'''
Defines choices for the problem command-line argument.
'''
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .sing_i import SingINoCorrection, SingIReg, SingI
from .sing_ih import SingIHKnownLine, SingIHKnownHat
from .sing_h import SingHHat, SingHParabola

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'sing-i-nc': SingINoCorrection,
    'sing-i-reg': SingIReg,
    'sing-i': SingI,
    'sing-h-hat': SingHHat,
    'sing-h-parabola': SingHParabola,
    'sing-ih-known-line': SingIHKnownLine,
    'sing-ih-known-hat': SingIHKnownHat,
}
