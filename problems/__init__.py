'''
Defines choices for the -p (problem) command-line argument.
'''
from .sine import Sine, SinePizza
from .ycosine import YCosine, YCosinePizza
from .bessel import BesselNoCorrection, BesselReg, Bessel
#from .shc_bessel import ShcBesselKnown, ShcBessel

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'bessel-nc': BesselNoCorrection,
    'bessel-reg': BesselReg,
    'bessel': Bessel,
#    'shc-bessel-known': ShcBesselKnown,
#    'shc-bessel': ShcBessel,
}
