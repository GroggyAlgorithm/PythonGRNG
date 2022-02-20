# PythonGRNG
Standalone version of my random number generator for Python

The perlin octave noise is currently struggling with performance
in ways the c# version does not. I imagine this id due to me looping through
the full resolution size on each axis as opposed to the cube root in the 3D
octave noise or the square root for the 2D.


Currently available algorithms for general selection:

```
 seed:int - The starting seed for the psuedo random number generator.\n
 algo:str - The random algorithm to use.\n
 Current algorithm choices (case insensitive):\n

 default and incorrect args - An edited and and Adapted lehmer 32 bit algorithm.

 'x32' - Xor shift 32 bit algorithm

 'x64' - Xor shift 64 bit algorithm

 'x128' - Xor shift 128 bit algorithm

 'wy64' - The w.y. hash 64 bit algorithm

 'l64' - The lehmer 64 bit algorithm

```


Example use:

```
# Selects the wy64 algorithm for each random function with a starting seed of 10
g = GRNG(seed=10, algo='wy64') 

```
