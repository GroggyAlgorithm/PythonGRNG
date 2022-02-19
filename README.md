# PythonGRNG
Standalone version of my random number generator for Python

Currently available algorithms for general selection:

```
# algo:str=None - The random algorithm to use\n
# Current choices (case insensitive):\n
#
# default and incorrect args - Adapted lehmer 32 algorithm
#
# 'x32' - Xor shift 32 algorithm
#
# 'x64' - Xor shift 64 algorithm
#
# 'x128' - Xor shift 128 algorithm
#
# 'wy64' - The w.y. hash 64 algorithm
#
# 'l64' - The lehmer 64 algorithm

```


Example use:

```
# Selects the wy64 algorithm for each random function with a starting seed of 10
g = GRNG(seed=10, algo='wy64') 

```
