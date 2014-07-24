from .mountain import Mountain
from .sine import Sine, SinePizza
from .wave import Wave, WavePizza
from .ripple import Ripple, RipplePizza
from .ycosine import YCosine, YCosinePizza
from .jump import JumpNoCorrection, JumpReg, u04TestProblem


problem_dict = {
    'mountain': Mountain,
    'sine': Sine,
    'sine-pizza': SinePizza,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'ripple': Ripple,
    'ripple-pizza': RipplePizza,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'jump-nc': JumpNoCorrection,
    'jump-reg': JumpReg,
    'u04-test': u04TestProblem,
}
