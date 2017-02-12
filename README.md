# Numerical solver for the Helmholtz equation on a domain with a reentrant corner
Author: Sam Magura (śȑmȧgũŗa@ncsu.edu without the accents)
License: GPL

Python implementation of a numerical solver for the Helmholtz equation over an irregular two-dimensional domain. The solution typically becomes singular at the corner, disrupting the convergence. To restore the numerical accuracy, we subtract the singularity before solving. The resulting regularized problem is then solved with high-order accuracy via the method of difference potentials, which uses a finite difference scheme as the core discretization methodology.

An unofficial draft of our paper, which describes the problem and the solution algorithm in full detail, is available [here](https://drive.google.com/file/d/0B2uOumn4y0KyOWxpdlM5Tnhwckk/view?usp=sharing). The paper is going to be submitted to the Journal of Computational Physics.

## Code organization

* `ps` - main source code. Contains the regularization algorithm and the method of difference potentials
* `problems` - definitions of boundary value problems that the algorithm can solve
* `scripts` - various executable scripts for debugging, plotting, and studying the algorithm
* `test` - unit tests

## How to run
Requires Python 3.x as well as the Python packages listed in requirements.txt. To install the packages:
$ pip install -r requirements.txt

Command-line interface help:
$ python3 run.py -h
$ python3 runz.py -h

## Additional references
[R1] M. Medvinsky, S. Tsynkov, E. Turkel. "The Method of Difference Potentials for the Helmholtz Equation Using Compact High Order Schemes." Journal of Scientific Computation 53 (2012).   
[R2] S. Magura. ["Solving the Helmholtz equation via difference potentials on a domain with a reentrant corner."](http://www4.ncsu.edu/~srmagura/media/ma491_paper.pdf)   
[R3] D. S. Britt, S. V. Tsynkov, E. Turkel. "A high-order numerical method for the Helmholtz equation with nonstandard boundary conditions." SIAM Journal of Scientific Computation 35:5 (2013).   
[R4] S. Britt, S. Petropavlovsky, S. Tsynkov, E. Turkel. "Computation of singular solutions to the Helmholtz equation with high order accuracy." Applied Numerical Mathematics 93 (July 2015).

Comments in the code may reference these papers.

## Testing
The primary way to test this software is to ensure that all problems are solved with the expected convergence rate. (The help for the command-line interface lists all defined problems.)

There is also some unit testing. Unit tests are in the test directory.

To run them:
$ python3 -m unittest
from the project's base directory. Python's automatic test discovery is used.

## Naming conventions
There will undoubtedly be differences in variable names when comparing this software package to our research papers. At the very least, we attempt to be internally consistent. The variable names match up most closely with [R2].

Here are some naming conventions that are used throughout the project:

Segment ID's -- 0 for outer boundary, 1 for lower radius, 2 for upper radius
n_<something> -- shorthand notation for "number of <something>'s"
eval_f(...) -- evaluate the (mathematical) function f
d_f_x -- partial derivative of f with respect to x
d2_f_x -- second partial derivative of f with respect to x
d2_f_x_y -- partial derivative of f with respect to x and y (mixed partial)
th -- shorthand for theta
