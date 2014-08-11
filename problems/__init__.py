from .sine import Sine, SinePizza
from .wave import Wave, WavePizza
from .ripple import Ripple, RipplePizza, RipplePizzaSympy
from .ycosine import YCosine, YCosinePizza
from .jump import JumpNoCorrection, JumpReg0


problem_dict = {
    'sine': Sine,
    'sine-pizza': SinePizza,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'ripple': Ripple,
    'ripple-pizza': RipplePizza,
    'ripple-pizza-sympy': RipplePizzaSympy,
    'ycosine': YCosine,
    'ycosine-pizza': YCosinePizza,
    'jump-nc': JumpNoCorrection,
    'jump-reg0': JumpReg0,
}
