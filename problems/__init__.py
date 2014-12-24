from .sine import Sine, SinePizza
from .wave import Wave, WavePizza
from .ycosine import YCosine, YCosinePizza
from .bessel import BesselNoCorrection, BesselReg, Bessel

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'bessel-nc': BesselNoCorrection,
    'bessel-reg': BesselReg,
    'bessel': Bessel
}
