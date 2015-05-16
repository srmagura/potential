'''
Defines choices for the problem command-line argument.
'''
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .sing_i import SingINoCorrection, SingIReg, SingI
from .sing_ih import ShcBesselKnown, ShcBessel
from .sing_h import FourierBesselHat, FourierBesselParabola

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'sing-i-nc': SingINoCorrection,
    'sing-i-reg': SingIReg,
    'sing-i': SingI,
    'shc-bessel-known': ShcBesselKnown,
    'shc-bessel': ShcBessel,
    'fourier-bessel-hat': FourierBesselHat,
    'fourier-bessel-parabola': FourierBesselParabola,
}
