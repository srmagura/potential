'''
Defines choices for the problem command-line argument.
'''
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .bessel import BesselNoCorrection, BesselReg, Bessel
from .shc_bessel import ShcBesselKnown, ShcBessel
from .fourier_bessel import FourierBessel
from .fourier_bessel_mpmath import FourierBesselMpMath

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'bessel-nc': BesselNoCorrection,
    'bessel-reg': BesselReg,
    'bessel': Bessel,
    'shc-bessel-known': ShcBesselKnown,
    'shc-bessel': ShcBessel,
    'fourier-bessel': FourierBessel,
    'fourier-bessel-mpmath': FourierBesselMpMath,
}
