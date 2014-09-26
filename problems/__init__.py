from .sine import Sine, SinePizza
from .wave import Wave, WavePizza
from .ycosine import YCosine, YCosinePizza
from .jump import JumpNoCorrection, JumpReg0
from .bessel import BesselNoCorrection, BesselReg, Bessel

problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'jump-nc': JumpNoCorrection,
    'jump-reg0': JumpReg0,
    'bessel-nc': BesselNoCorrection,
    'bessel-reg': BesselReg,
    'bessel': Bessel
}
