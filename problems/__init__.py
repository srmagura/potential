'''
Defines choices for the -p (problem) command-line argument.
'''
from .sine import Sine, SinePizza
from .wave import Wave, WavePizza
from .ycosine import YCosine, YCosinePizza
from .bessel import BesselNoCorrection, BesselReg, Bessel
from .shc_bessel import ShcBesselKnown, ShcBessel

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'bessel-nc': BesselNoCorrection,
    'bessel-reg': BesselReg,
    'bessel': Bessel,
    'shc-bessel-known': ShcBesselKnown,
    'shc-bessel': ShcBessel,
}
