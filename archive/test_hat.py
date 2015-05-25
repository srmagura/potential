import numpy as np

a = np.pi/6

def x(th):
    return (2*th-(2*np.pi+a))/(2*np.pi-a)

print(x(a))
print(x((2*np.pi+a)/2))
print(x(2*np.pi))
