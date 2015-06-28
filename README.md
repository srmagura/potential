# potential - numerical solvers for the Helmholtz equation
Author: Sam Magura (srmagura AT ncsu.edu)
License: GPL 

This software package contains solvers for the Helmholtz equation in 2D, using finite differences and the method of difference potentials. Currently, there are solvers for circular domains and the "pizza domain", i.e. a disk with a wedge removed.

`potential` requires Python 3.x as well as the Python packages listed in requirements.txt.

If you don't understand the mathematical algorithm that this code implements, you won't understand the code! If you want to run this code or modify it for use in a project of your own, you'll need to do some reading first.

## References
* M. Medvinsky, S. Tsynkov, E. Turkel. "The Method of Difference Potentials for the Helmholtz Equation Using Compact High Order Schemes." Journal of Scientific Computation (2012) 53.
* S. Magura. ["Solving the Helmholtz equation via difference potentials on a domain with a reentrant corner."](http://www4.ncsu.edu/~srmagura/media/ma491_paper.pdf)
