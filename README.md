# PythonGRNG
Standalone version of my random number generator for Python. Requires python 3.10+.

The perlin octave noise is currently struggling with performance in ways the c# version does not. 

I imagine this id due to me looping through the full resolution size on each axis as opposed to the cube root in the 3D
octave noise or the square root for the 2D.

Imports

```
from numbers import Number    
import time
import math
import sys
```

Arguments
```
seed:Number=None - The seed for the random number generator. It will default to math.trunc(time.time)
    
algo:str=None - The random algorithm to use(case insensitive)

```


Class wide Algorithm selection. When running some methods, they will default to using this to generate the numbers. Individual algorithms can also just be called on their own.

```
 Current algorithm choices (case insensitive):\n

 default and incorrect args - An edited and and Adapted lehmer 32 bit algorithm.

 'x32' - Xor shift 32 bit algorithm

 'x64' - Xor shift 64 bit algorithm

 'x128' - Xor shift 128 bit algorithm

 'wy64' - The w.y. hash 64 bit algorithm

 'l64' - The lehmer 64 bit algorithm

```

Properties/Class variables

```
__MAX_VALUE: The maximum value for the values. Current set to sys.maxsize-1    

__MIN_VALUE: The minimum value for the values. Current set to -__MAX_VALUE    

self.__randAlgo = lambda x:....: lambda for the selected random algorithm function.    

self.seed: The random number generators seed.
    
self.permutationTable: The permutation table for perlin noise. Gets set to values 0-255 inclusive. 
    
```

Private / Internal use Functions

```
Math functions for the class only. Added here in order to let the class exist on its own.
    
def __CbLerp(self, a:Number,b:Number,t:Number) -> Number:

def __Fade(self, t:Number) -> Number:

def __Clamp(self, value:Number,min:Number,max:Number) -> Number:

def __Clamp01(self, value:Number) -> Number:

def __InverseLerp(self, a:Number,b:Number,value:Number ) -> Number:

def __InverseLerpClamped(self, a:Number,b:Number,value:Number) -> Number:

def __Lerp(self, a:Number,b:Number,t:Number) -> Number:

def __LerpClamped(self, a:Number,b:Number,t:Number) -> Number:

def __NormalizeInRange(self, value:Number, minPossibleValue:Number, maxPossibleValue:Number) -> Number:


```


Functions
 
```
    def xorshift32(value:int) -> int: Performs the xorshift 32 algorithm
    
    def xorshift64(value:int) -> int: Performs the xorshift 64 algorithm
    
    def reverse17(val:int) -> int: Xor shift variation: (val >> 17) ^ (val >> 34) ^ (val >> 51)
    
    def reverse23(val:int) -> int: Xor shift variation: (val << 23) ^ (val << 46))
    
    def xorshift128(value0:int, value1:int=None) -> int: Performs the xorshift 128 algorithm
    
    def AdaptedLehmer32(value:int) -> int: A version of the Lehmer algorithm
    
    def wyHash64(value:int) -> int: A version of the wyhash 64-bit hash algorithm by Wang Yi
    
    def Lehmer64(value:int) -> int: A version of the Lehmer 64 algorithm
    
    def Next(self, maximum:Number=None)-> float: Returns a random value between 0 and 1. If a value is passed for maximum, the value will be normalized with the range of that
    
    def Range(self, a:Number, b:Number) -> Number: Returns a random value within the range passed, inclusive
    
    def NextBool(self) -> bool: Returns a random boolean value
    
    def NextInt(self) -> int: Returns the next integer value
    
    def RangeInt(self, a:int, b:int) -> int: Returns a random int value within the range passed, inclusive
    
    def NextPercentage(self) -> Number: Returns a random percentage value between 0 and 100
    
    def NextSign(self) -> int: Returns a random sign value of either negative or positive
    
    def NextElement(self, l:list): Returns a random value within the passed list
    
    def Shuffle(self, l:list) -> None: Shuffles the list passed
    
    def GrayScale(self): Returns a random gray scale value
    
    def LowerBiasValue(self, strength:float) -> float: Creates a value biased towards 0 based on the strength

    def UpperBiasValue(self, strength:float) -> float: Creates a value biased towards 1 based on the strength

    def ExtremesBiasValue(self, strength:float) -> float: Returns a value biased towards the extremes

    def CenterBiasValue(self,strength:float) -> float: Returns a random value biased towards the center
    
    def NextGaussian(self) -> float: Creates a random value based on averaging around a point instead of an even distibution
    
    def Gaussian(self, center:Number, deviation:Number) -> float: Creates a random value based on averaging around a point instead of an even distibution
    
    def SmallestRandom(self, iterations:int) -> float: Returns the smallest value in x iterations

    def LargestRandom(self, iterations:int) -> float: Returns the largest value in x iterations

    def CenteredRandom(self, iterations:int) -> float: Returns the most centered of values in a range up to x iterations

    def Gradient1D(self, hash:int, x:Number):
        Returns gradient value for perlin noise

    def Gradient2D(self, hash:int, x:Number, y:Number):
        Returns gradient value for perlin noise

    def Gradient3D(self, hash:int, x:Number, y:Number, z:Number):
        Returns gradient value, from Ken Perlins inmproved noise

    def CreatePermutationTable(self): Creates a permutation table of values from 0 - 255 inclusive
    
    def Perlin1D(self, x:Number) -> float:
        "Creates 1D perlin noise
    
    def Perlin2D(self, x:Number, y:Number) -> float:
        Creates 2D perlin noise"
    
    def Perlin3D(self, x:Number, y:Number, z:Number) -> float:
        Creates 3D perlin noise"
    
    
    def ImprovedNoise(self, x:Number, y:Number,  z:Number=0.25) -> float: Ken perlins improved noise
    
    
    
    def SimplePerlin2D(self, xIteration:Number,  yIteration:Number, noiseScale:Number, frequency:Number, 
                      offsetX:Number=0, offsetY:Number=0, centerX:Number=0, centerY:Number=0) -> Number:
    
        Creates simple 2D perlin noise
    
    
    
    def SimplePerlin3D(self, xIteration:Number, yIteration:Number, zIteration:Number, noiseScale:Number, frequency:Number,  
                      offsetX:Number=0, offsetY:Number=0, offsetZ:Number=0, centerX:Number=0, centerY:Number=0, centerZ:Number=0) -> Number:
    
        Creates simple 3D perlin noise
    
    
   def PerlinOctaves2D(self, octaveAmount:int, resolution:int, persistance:Number, lacunarity:Number, 
                        scale:Number, offsetX:Number, offsetY:Number, normalizeHeights:bool, normalizeHeightGlobally:bool=True, roughness:Number=10_000):
        
        Creates a list of perlin layered octave noise
    

    def PerlinOctaves3D(self, octaveAmount:int, resolution:int, persistance:Number, lacunarity:Number, 
                        scale:Number, offsetX:Number, offsetY:Number, offsetZ:Number, normalizeHeights:bool, normalizeHeightGlobally:bool=True, roughness:Number=10_000):
        
        Creates a list of perlin layered octave noise


```





Example use:

```
from GRNG import GRNG

# Selects the wy64 algorithm for each random function with a starting seed of 10
g = GRNG(seed=10, algo='wy64')
print(GRNG.xorshift32(5))
print(GRNG.xorshift64(5))
print(GRNG.reverse17(100))
print(GRNG.reverse23(100))
print(GRNG.xorshift128(5))
print(GRNG.AdaptedLehmer32(5))
print(GRNG.wyHash64(5))
print(GRNG.Lehmer64(5))
print(g.Next())
print(g.Range(-9,100))
print(g.NextBool())
print(g.NextInt())
print(g.RangeInt(-9,100))
print(g.NextPercentage())
print(g.NextSign())
a = [1,2,3]
print(g.NextElement(a))
g.Shuffle(a)
print(a)
print(g.GrayScale())
print(g.LowerBiasValue(0.1))
print(g.UpperBiasValue(0.5))
print(g.ExtremesBiasValue(0.5))
print(g.CenterBiasValue(0.5))
print(g.NextGaussian())
print(g.Gaussian(center=5,deviation=4))

for i in range(0,5):
    print(g.SmallestRandom(5))
    print(g.LargestRandom(5))
    print(g.CenteredRandom(5))

print(g.ImprovedNoise(1,2))
print(g.SimplePerlin2D(1,2,1,2,1,2,4,4))
print(g.SimplePerlin3D(1,2,3,1,2,1,2,3,1,2,3))
print(g.CreatePermutationTable())
print(g.PerlinOctaves3D(2,5,1,2,1,1,2,3,True,10_000))
print(g.PerlinOctaves2D(5,5,1,2,1,1,2,True,10_000))
print(g.Gradient(100,1,2,3))

```
