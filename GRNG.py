from numbers import Number
import time
import math
import sys


class GRNG:
    """Standalone version of The Groggy Random Number Generator.
    
    Requirements and imports:
    -----------------------------
    
    Python 3.10+
    
    from numbers import Number
    
    import time
    
    import math
    
    import sys
    
    from warnings import warn
    
    Arguments:
    -----------------------------
    
    seed:Number=None - The seed for the random number generator.
    It will default to math.trunc(time.time)
    
    
    algo:str=None - The random algorithm to use(case insensitive)
    
    
    Class wide Algorithm selection:
    ---------------------------------
    
    Current algorithm choices (case insensitive):
    
    default and incorrect args - Adapted lehmer 32 algorithm
    
    'x32' - Xor shift 32 algorithm
    
    'x64' - Xor shift 64 algorithm
    
    'x128' - Xor shift 128 algorithm
    
    'wy64' - The w.y. hash 64 algorithm
    
    'l64' - The lehmer 64 algorithm
    
    
    Properties/Class variables:
    ---------------------------------
    
    __MAX_VALUE: The maximum value for the values. Current set to sys.maxsize-1
    
    __MIN_VALUE: The minimum value for the values. Current set to -__MAX_VALUE
    
    self.__randAlgo = lambda x:....: lambda for the selected random algorithm function.
    
    self.seed: The random number generators seed.
    
    self.permutationTable: The permutation table for perlin noise. Gets set to values 0-255 inclusive. 
    
    
    Private/ Internal use Functions:
    ------------------------
    
    Math functions for the class only. Added here in order to let the class exist on its own.
    
    
    \ndef __CbLerp(self, a:Number,b:Number,t:Number) -> Number:
    \ndef __Fade(self, t:Number) -> Number:
    \ndef __Clamp(self, value:Number,min:Number,max:Number) -> Number:
    \ndef __Clamp01(self, value:Number) -> Number:
    \ndef __InverseLerp(self, a:Number,b:Number,value:Number ) -> Number:
    \ndef __InverseLerpClamped(self, a:Number,b:Number,value:Number) -> Number:
    \ndef __Lerp(self, a:Number,b:Number,t:Number) -> Number:
    \ndef __LerpClamped(self, a:Number,b:Number,t:Number) -> Number:
    \ndef __NormalizeInRange(self, value:Number, minPossibleValue:Number, maxPossibleValue:Number) -> Number:
    \n
    
    
    Functions:
    ------------------------------
    
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

    
    
    
    """
    
    __MAX_VALUE = sys.maxsize-1
    __MIN_VALUE = -__MAX_VALUE
    
    def __init__(self,seed:int=None, algo:str=None, usePremadePermutation:bool=True) -> None:
        
        self.permutationTable = [151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        151] if usePremadePermutation == True else self.CreatePermutationTable()
        
        self.seed = seed if seed is not None else math.trunc(time.time())
        __tmpalgo = algo.lower() if algo is not None else ''
        match __tmpalgo:
            case 'xor32':
                self.__randAlgo = lambda x: GRNG.xorshift32(x)
            case 'xor64':
                self.__randAlgo = lambda x: GRNG.xorshift64(x)
            case 'xor128':
                self.__randAlgo = lambda x: GRNG.xorshift128(x)
            case 'wy64':
                self.__randAlgo = lambda x: GRNG.wyHash64(x)
            case 'l64':
                self.__randAlgo = lambda x: GRNG.Lehmer64(x)
            case _:
                self.__randAlgo = lambda x: GRNG.AdaptedLehmer32(x)
        
        pass

    def __CbLerp(self, a:Number,b:Number,t:Number) -> Number:
        return ( b - a ) * t + a
    

    def __Fade(self, t:Number) -> Number:
        return (t*t*t*(t*(t*6-15)+10))
        
    def __Clamp(self, value:Number,min:Number,max:Number) -> Number:
        newV = value
        if( newV < min ): newV = min
        if( newV > max ): newV = max
        return newV
    

    def __Clamp01(self, value:Number) -> Number: 
        return self.__Clamp(value,0,1)

    def __InverseLerp(self, a:Number,b:Number,value:Number ) -> Number: ( value - a ) / ( b - a )
    def __InverseLerpClamped(self, a:Number,b:Number,value:Number) -> Number: self.__Clamp01( ( value - a ) / ( b - a ) )
    def __Lerp(self, a:Number,b:Number,t:Number) -> Number: ( 1 - t ) * a + t * b
    def __LerpClamped(self, a:Number,b:Number,t:Number) -> Number:
        t = self.__Clamp01( t )
        return ( 1 - t ) * a + t * b
    
    def __NormalizeInRange(self, value:Number, minPossibleValue:Number, maxPossibleValue:Number) -> Number:
        checkedmin = min(minPossibleValue,maxPossibleValue)
        checkedmax = max(minPossibleValue,maxPossibleValue)
        
        if(value <= checkedmin or (checkedmax - checkedmin == 0)):
            normalizedValue = 0
        elif(value >= checkedmax or checkedmin == checkedmax):
            normalizedValue = 1
        else:
            normalizedValue = ((value - checkedmin) / (checkedmax - checkedmin));

        return normalizedValue
        

    def xorshift32(value:int) -> int:
        """Performs the xorshift 32 algorithm
        
        Arguments
        -------------------
        value:int - The value to perform xor shift on
        
        Returns
        ------------------
        -> int: - The new value
        
        """
        
        rand = value
        rand ^= value << 13
        rand ^= value >> 17
        rand ^= value << 5
        return rand & GRNG.__MAX_VALUE



    def xorshift64(value:int) -> int:
        """Performs the xorshift 64 algorithm
        
        Arguments
        -------------------
        value:int - The value to perform xor shift on
        
        Returns
        ------------------
        -> int: - xorshifted psuedo random value
        
        """
        rand = value
        rand ^= value << 13
        rand ^= value >> 7
        rand ^= value << 17

        return rand & GRNG.__MAX_VALUE
    
    def reverse17(val:int) -> int:
        """
        Xor shift variation: (val >> 17) ^ (val >> 34) ^ (val >> 51)
        """
        return (val ^ (val >> 17) ^ (val >> 34) ^ (val >> 51)) & GRNG.__MAX_VALUE


    def reverse23(val:int) -> int:
        """
        Xor shift variation: (val << 23) ^ (val << 46))
        """
        return (val ^ (val << 23) ^ (val << 46)) & GRNG.__MAX_VALUE
    

    def xorshift128(value0:int, value1:int=None) -> int:
        """Performs the xorshift 128 algorithm
        
        Arguments
        -------------------
        value0:int - The first value to perform xor shift with
        value1:int=None - The second value to perform xor shift with
        Defaults to None and sets itself at reverse23(reverse17(value0))
        
        Returns
        ------------------
        -> int: - xorshifted psuedo random value
        
        """
        if value1 is None:
            value1 = GRNG.reverse23(GRNG.reverse17(value0))
        v0 = value1
        v1 = value0
        v1 ^= (v1 << 23)
        v1 ^= (v1 >> 17)
        v1 ^= v0
        v1 ^= (v0 >> 26)
        return v1 & GRNG.__MAX_VALUE
    
    
    
    def AdaptedLehmer32(value:int) -> int:
        """A version of the Lehmer algorithm
        
        Arguments
        ----------------------
        value:int - The value to run through the lehmer algorithm
        
        Returns
        ----------------------
        -> int: - The psuedo random number created
        
        """
        value += int(0xe120fc15)
        tmp = value * int(0x4a39b70d)
        m1 = ((tmp >> 32) ^ tmp)
        tmp = m1 * int(0x12fad5c9)
        return ((tmp >> 32)^tmp) & GRNG.__MAX_VALUE
    
    
    def wyHash64(value:int) -> int:
        """A version of the wyhash 64-bit hash algorithm by Wang Yi
        
        
        Arguments
        --------------------------
        value:int - The value to put through the algorithm
        
        
        Returns
        ----------------------
        -> int: - The psuedo random number created
        
        """
        
        value += int(0x60bee2bee120fc15)
        tmp = value * int(0x60bee2bee120fc15)
        m1 = ((tmp >> 64) ^ tmp)
        tmp = m1 * int(0x1b03738712fad5c9)
        m2 = (tmp >> 64) ^ tmp
        return m2 & GRNG.__MAX_VALUE
    
    
    def Lehmer64(value:int) -> int:
        """A version of the Lehmer 64 algorithm
        
        Arguments
        --------------------------
        value:int - The value to put through the algorithm
        
        
        Returns
        ----------------------
        -> int: - The psuedo random number created
        
        """
        value *= int(0xda942042e4dd58b5)
        return (value >> 64) & GRNG.__MAX_VALUE


    def Next(self, maximum:Number=None)-> float:
        """Returns a random value between 0 and 1. If a value is passed for maximum, the value will be normalized with the range of that
        """
        randVal = self.__randAlgo(self.seed)
        self.seed = randVal
        if(maximum is None):
            randVal = (self.__NormalizeInRange(value=randVal,minPossibleValue=GRNG.__MIN_VALUE, maxPossibleValue=GRNG.__MAX_VALUE))
        else:
            randVal = (self.__NormalizeInRange(value=randVal,minPossibleValue=GRNG.__MIN_VALUE, maxPossibleValue=maximum))
        return randVal
    
    def Range(self, a:Number, b:Number) -> Number:
        """
        Returns a random value within the range passed, inclusive
        """
        return self.__LerpClamped(a,b,self.Next())

    def NextBool(self) -> bool: 
        """Returns a random boolean value
        """
        nB = True if self.Next() > 0.5 else False
        return nB

    def NextInt(self) -> int:
        """Returns the next integer value"""
        randVal = self.__randAlgo(self.seed)
        self.seed = randVal
        return randVal
    
    def RangeInt(self, a:int, b:int) -> int:
        """
        Returns a random int value within the range passed, inclusive
        """
        randVal = self.Range(a,b)
        return math.trunc(randVal)
    
    def NextPercentage(self) -> Number:
        """Returns a random percentage value between 0 and 100"""
        return self.Range(0,100)
    
    def NextSign(self) -> int:
        """Returns a random sign value of either negative or positive"""
        val = 1 if self.NextBool() is True else -1
        return val

    def NextElement(self, l:list):
        """Returns a random value within the passed list"""
        return l[self.RangeInt(0,len(l)-1)]

    def Shuffle(self, l:list) -> None:
        """Shuffles the list passed"""
        for i in range(0, len(l)):
            j = self.RangeInt(0,len(l)-1)
            tmp = l[j]
            l[j] = l[i]
            l[i] = tmp
            
    def GrayScale(self):
        """Returns a random gray scale value"""
        randVal = self.RangeInt(0,256)
        return randVal,randVal,randVal
    
    def LowerBiasValue(self, strength:float) -> float:
        """Creates a value biased towards 0 based on the strength"""
        t = self.Next()
        if (abs(strength) >= 1): rVal = 0 #To avoid dividing by 0
        else:
            k=self.__Clamp01(1 - abs(strength))
            k = k * k * k - 1
            rVal = self.__Clamp01 ((t + t * k) / (t * k + 1))
        return rVal
    
    def UpperBiasValue(self, strength:float) -> float:
        """Creates a value biased towards 1 based on the strength"""
        return (1 - self.LowerBiasValue(strength))

    def ExtremesBiasValue(self, strength:float) -> float:
        """Returns a value biased towards the extremes"""
        t = self.LowerBiasValue(strength)
        rVal = t if self.NextBool() is False else 1-t
        return rVal
    def CenterBiasValue(self,strength:float) -> float:
        """Returns a random value biased towards the center"""
        t = self.LowerBiasValue(strength)
        return (0.5 + t * 0.5 * self.NextSign())

    def NextGaussian(self) -> float:
        """Creates a random value based on averaging around a point instead of an even distibution"""
        value1 = 0
        value2 = 0
        s = 0
        while(s >= 1.0 or s == 0):
            value1 = 2.0 * self.Range(0, 1) - 1.0
            value2 = 2.0 * self.Range(0, 1) - 1.0
            s = (value1 * value1) + (value2 * value2)
        s = math.sqrt((-2 * math.log(s)) / s)
        return value1 * s

    def Gaussian(self, center:Number, deviation:Number) -> float:
        """Creates a random value based on averaging around a point instead of an even distibution"""
        return (center + self.NextGaussian() * deviation)

    def SmallestRandom(self, iterations:int) -> float:
        """Returns the smallest value in x iterations"""
        rVal = self.__MAX_VALUE
        for i in range(0, iterations):
            rVal = min(rVal,self.Next())
        return rVal

    def LargestRandom(self, iterations:int) -> float:
        """Returns the largest value in x iterations"""
        rVal = self.__MIN_VALUE
        for i in range(0, iterations):
            rVal = max(rVal,self.Next())
        return rVal

    def CenteredRandom(self, iterations:int) -> float:
        """Returns the most centered of values in a range up to x iterations"""
        rVal = 1
        for i in range(0, iterations):
            rand = self.Next()
            if (abs(rand) - 0.5) < (rVal - 0.5): rVal = rand
        return rVal



    def Gradient1D(self, hash:int, x:Number):
        """Returns gradient value for perlin noise"""
        rV = x if (hash & 1) == 0 else -x
        return (rV)


    def Gradient2D(self, hash:int, x:Number, y:Number):
        """Returns gradient value for perlin noise"""
        rU = y if (hash & 2) == 0 else -y
        rV = x if (hash & 1) == 0 else -x
        return (rV+rU)

    def Gradient3D(self, hash:int, x:Number, y:Number, z:Number):
        """Returns gradient value, from Ken Perlins inmproved noise"""
        h = hash & 15
        u = x if h<8 else y
        v = y if h<4 else x if h==12 or h==14 else z
        rU = u if h&1 == 0 else -u
        rV = v if h&2 == 0 else -v
        return (rU+rV)
    
    def CreatePermutationTable(self):
        """Creates a permutation table of values from 0 - 255 inclusive"""
        newPermutation = []
        for i in range(0, 256):
            newPermutation.append(i)
        
        self.Shuffle(newPermutation)

        for i in range(256, 512):
            newPermutation.append(i-256)

        
        self.Shuffle(newPermutation)

        return newPermutation
    
    def Perlin1D(self, x:Number) -> float:
        """Creates 1D perlin noise"""
        newX = math.floor(x) & 0xff
        x-= math.floor(x)
        fadedX = self.__Fade(x)
        perm = self.permutationTable
        return self.__CbLerp(fadedX,self.Gradient1D(perm[newX],x),self.Gradient1D(perm[newX+1],x-1)) * 2 *GRNG.__MAX_VALUE
    
    def Perlin2D(self, x:Number, y:Number) -> float:
        """Creates 2D perlin noise"""
        perm = self.permutationTable
        newX = math.floor(x) & 0xff
        newY = math.floor(y) & 0xff
        x-= math.floor(x)
        y-= math.floor(y)
        fadedX = self.__Fade(x)
        fadedY = self.__Fade(y)
        A = (perm[newX] + newY) & 0xff
        B = (perm[newX + 1] + newY) & 0xff
        return self.__CbLerp(fadedY, self.__CbLerp(fadedX, self.Gradient2D(perm[A],x,y),self.Gradient2D(perm[B], x-1,y)),
            self.__CbLerp(fadedX, self.Gradient2D(perm[A+1],x,y-1), self.Gradient2D(perm[B+1], x-1, y-1))) *GRNG.__MAX_VALUE
    
    def Perlin3D(self, x:Number, y:Number, z:Number) -> float:
        """Creates 3D perlin noise"""
        perm = self.permutationTable
        newX = math.floor(x) & 0xff
        newY = math.floor(y) & 0xff
        newZ = math.floor(z) & 0xff
        x-= math.floor(x)
        y-= math.floor(y)
        z-= math.floor(z)
        fadedX = self.__Fade(x)
        fadedY = self.__Fade(y)
        fadedZ = self.__Fade(z)
        
        A = (perm[newX] + newY) & 0xff
        B = (perm[newX + 1] + newY) & 0xff
        AA = (perm[A] + newZ) & 0xff
        BA = (perm[B] + newZ) & 0xff
        AB = (perm[A+1] + newZ) & 0xff
        BB = (perm[B+1] + newZ) & 0xff
        
        return self.__CbLerp(fadedZ, 
            self.__CbLerp(fadedY, 
                self.__CbLerp(fadedX, self.Gradient(perm[AA], x, y, z), self.Gradient(perm[BA], x-1, y, z)),
                    self.__CbLerp(fadedX, self.Gradient(perm[AB], x, y-1, z), self.Gradient(perm[BB], x-1, y-1, z))),
                       self.__CbLerp(fadedY, self.__CbLerp(fadedX, self.Gradient(perm[AA+1], x, y  , z-1), self.Gradient(perm[BA+1], x-1, y  , z-1)),
                            self.__CbLerp(fadedX, self.Gradient(perm[AB+1], x, y-1, z-1), self.Gradient(perm[BB+1], x-1, y-1, z-1)))) *GRNG.__MAX_VALUE
        
    
    def ImprovedNoise(self, x:Number, y:Number,  z:Number=0) -> float:
        """Based on Ken perlins improved noise
        Arguments
        ---------------------------
        x: x axis value
        y: y axis value
        z: z axis value
        
        Returns
        ---------------------------
         -> Number: A Perlin value
        """
        if self.permutationTable is None:
            self.permutationTable = self.CreatePermutationTable()
        
        p = self.permutationTable

        x+=self.Next()
        y+=self.Next()
        z+=self.Next()

        #FIND UNIT CUBE THAT CONTAINS POINT.
        X = math.floor(x) & 255
        Y = math.floor(y) & 255
        Z = math.floor(z) & 255
        x -= math.floor(x)                                # FIND RELATIVE X,Y,Z
        y -= math.floor(y)                                # OF POINT IN CUBE.
        z -= math.floor(z)
        u = self.__Fade(x)                                # COMPUTE __Fade CURVES
        v = self.__Fade(y)                                # FOR EACH OF X,Y,Z.
        w = self.__Fade(z)
                
        #  HASH COORDINATES OF THE 8 CUBE CORNERS,
        A = p[X]+Y 
        AA = p[A]+Z
        AB = p[A+1]+Z
        B = p[X+1]+Y 
        BA = p[B]+Z 
        BB = p[B+1]+Z
        
        #  AND ADD BLENDED RESULTS FROM  8 CORNERS OF CUBE
        rVal = self.__CbLerp(w, self.__CbLerp(v, 
                    self.__CbLerp(u, self.Gradient(p[AA], x,y,z), 
                    self.Gradient(p[BA], x-1, y  , z   )),
                    self.__CbLerp(u, self.Gradient(p[AB], x  , y-1, z   ),
                    self.Gradient(p[BB], x-1, y-1, z   ))),
                    self.__CbLerp(v, 
                    self.__CbLerp(u, 
                    self.Gradient(p[AA+1], x  , y  , z-1 ),
                    self.Gradient(p[BA+1], x-1, y  , z-1 )), 
                    self.__CbLerp(u, self.Gradient(p[AB+1], x  , y-1, z-1 ),
                    self.Gradient(p[BB+1], x-1, y-1, z-1 ))))

        return (rVal*GRNG.__MAX_VALUE)
    
    
    
    def SimplePerlin2D(self, xIteration:Number,  yIteration:Number, noiseScale:Number, frequency:Number, 
                      offsetX:Number=0, offsetY:Number=0, centerX:Number=0, centerY:Number=0) -> Number:
    
        """Creates simple 2D perlin noise
        
        Arguments
        --------------------------
        xIteration:Number - Iteration/Location on the x axis
        yIteration:Number - Iteration/Location on the y axis
        noiseScale:Number - Scale of the noise
        frequency:Number - frequency value for the noise
        offsetX:Number=0 - Offset for the noise on the x axis. Defaults to 0
        offsetY:Number=0 - Offset for the noise on the y axis. Defaults to 0
        centerX:Number=0 - center location on the x axis. Defaults to 0
        centerY:Number=0 - center location on the y axis. Defaults to 0
        
        Returns
        -------------------------
        -> Number: The perlin noise created
        
        """
    
        perlinValue = 0
        divisor = noiseScale * frequency if noiseScale and frequency != 0 else 1
        newX = (xIteration - centerX + offsetX) / divisor
        newY = (yIteration - centerY - offsetY) / divisor

        perlinValue = self.Perlin2D(x=newX, y=newY) * 2 - 1

        return perlinValue

    def SimplePerlin3D(self, xIteration:Number, yIteration:Number, zIteration:Number, noiseScale:Number, frequency:Number,  
                      offsetX:Number=0, offsetY:Number=0, offsetZ:Number=0, centerX:Number=0, centerY:Number=0, centerZ:Number=0) -> Number:
    
        """Creates simple 3D perlin noise
        
        Arguments
        --------------------------
        xIteration:Number - Iteration/Location on the x axis
        yIteration:Number - Iteration/Location on the y axis
        zIteration:Number - Iteration/Location on the z axis
        noiseScale:Number - Scale of the noise
        frequency:Number - frequency value for the noise
        offsetX:Number=0 - Offset for the noise on the x axis. Defaults to 0
        offsetY:Number=0 - Offset for the noise on the y axis. Defaults to 0
        offsetZ:Number=0 - Offset for the noise on the z axis. Defaults to 0
        centerX:Number=0 - center location on the x axis. Defaults to 0
        centerY:Number=0 - center location on the y axis. Defaults to 0
        centerZ:Number=0 - center location on the y axis. Defaults to 0
        
        Returns
        -------------------------
        -> Number: The perlin noise created
        
        """
    
        perlinValue = 0
        divisor = noiseScale * frequency if noiseScale != 0 and frequency != 0 else 1
        newx = (xIteration - centerX + offsetX) / divisor
        newy = (yIteration - centerY - offsetY) / divisor
        newz = (zIteration - centerZ + offsetZ) / divisor

        perlinValue = (self.Perlin3D(x=newx, y=newy, z=newz) * 2 - 1)

        return perlinValue


    def PerlinOctaves2D(self, octaveAmount:int, resolution:int, persistance:Number, lacunarity:Number, 
                        scale:Number, offsetX:Number, offsetY:Number, normalizeHeights:bool, normalizeHeightGlobally:bool=True, roughness:Number=10_000):
        
        """Creates a list of perlin layered octave noise
        
        Arguments
        -----------------------------------
        octaveAmount:int- The amount of octaves/layers for the noise. 
        resolution:int- The size of the area.
        persistance:Number- The persistance value for how quickly the noise wave should drop on the y axis.
        lacunarity:Number- The lacunarity value for the x axis of the noise wave.
        scale:Number- The scaling for the noise. Controls the zooming in on the noise.
        offsets X and Y:Number- The offsets for the noise.
        normalizeHeights:bool- if the height should be normalized
        normalizeHeightGlobally:bool- If the values should be normalized globally. Defaults to true
        roughness:Number- Roughness value to add to the noise. Defaults to 10_000-
        
        
        Returns
        ----------------------------------
        A list of perlin values
        
        """
        
        if(octaveAmount < 1): octaveAmount = 1
        if(persistance <= 0): persistance = 0.001
        if(lacunarity < 0.01): lacunarity = 0.01
        if(resolution < 1): resolution = 1
        noiseMap = []
        octaveOffsets = [[0,0]]
        
        
        centerLocation = resolution/2
        amplitude = 1
        frequency = 1
        maximumHeight = 0
        minimumHeight = 0
        
        # GetOctaves
        for i in range(0, octaveAmount):
            x = self.Range(-roughness, roughness) + offsetX + centerLocation
            y = self.Range(-roughness, roughness) - offsetY - centerLocation
            octaveOffsets.append([x,y])
            maximumHeight += amplitude
            minimumHeight -= amplitude
            amplitude *= persistance
        
        y=0
        x=0

        for w in range(0, math.floor(math.pow(resolution,2))):
                
            amplitude = 1
            frequency = 1
            noiseValue = 0
            
            for i in octaveOffsets:
                noise = self.SimplePerlin2D(xIteration=x,yIteration=y, noiseScale=scale,frequency=frequency,offsetX=i[0],
                                            offsetY=i[1], centerX=centerLocation,centerY=centerLocation)
                noiseValue += noise * amplitude
                amplitude *= persistance
                frequency *= lacunarity
            
            if (noiseValue < minimumHeight):
                minimumHeight = noiseValue
            elif (noiseValue > maximumHeight):
                maximumHeight = noiseValue


            if (normalizeHeightGlobally and normalizeHeights):
                normalizedHeight = (noiseValue + 1) / (maximumHeight / 0.9)
                noiseValue = self.__Clamp(normalizedHeight, 0, sys.maxsize)
            elif(normalizeHeights):
                noiseValue = self.__Clamp(noiseValue,minimumHeight, maximumHeight)
                
            noiseMap.append(noiseValue)

            y+=1
            
            if(y >= resolution):
                x+=1
                y=0

            if(x >= resolution):
                break



        return noiseMap



    def PerlinOctaves3D(self, octaveAmount:int, resolution:int, persistance:Number, lacunarity:Number, 
                        scale:Number, offsetX:Number, offsetY:Number, offsetZ:Number, normalizeHeights:bool, normalizeHeightGlobally:bool=True, roughness:Number=10_000):
        
        """Creates a list of perlin layered octave noise
        
        Arguments
        -----------------------------------
        octaveAmount:int- The amount of octaves/layers for the noise. 
        resolution:int- The size of the area.
        persistance:Number- The persistance value for how quickly the noise wave should drop on the y axis.
        lacunarity:Number- The lacunarity value for the x axis of the noise wave.
        scale:Number- The scaling for the noise. Controls the zooming in on the noise.
        offsets X and Y:Number- The offsets for the noise.
        normalizeHeights:bool- if the height should be normalized
        normalizeHeightGlobally:bool- If the values should be normalized globally. Defaults to true
        roughness:Number- Roughness value to add to the noise. Defaults to 10_000-
        
        
        Returns
        ----------------------------------
        A list of perlin values
        
        """
        
        if(octaveAmount < 1): octaveAmount = 1
        if(persistance <= 0): persistance = 0.001
        if(lacunarity < 0.01): lacunarity = 0.01
        if(resolution < 1): resolution = 1
        
        noiseMap = []
        octaveOffsets = [[0,0,0]]
        
        

        centerLocation = resolution/2
        amplitude = 1
        frequency = 1
        maximumHeight = 0
        minimumHeight = 0
        
        # GetOctaves
        for i in range(0, octaveAmount):
            x = self.Range(-roughness, roughness) + offsetX + centerLocation
            y = self.Range(-roughness, roughness) - offsetY - centerLocation
            z = self.Range(-roughness, roughness) - offsetZ - centerLocation
            octaveOffsets.append([x,y,z])
            maximumHeight += amplitude
            minimumHeight -= amplitude
            amplitude *= persistance

        y=0
        x=0
        z=0

        for w in range(0, math.floor(math.pow(resolution,3))):
            
            amplitude = 1
            frequency = 1
            noiseValue = 0
            
            for i in octaveOffsets:
                noise = self.SimplePerlin3D(xIteration=x,yIteration=y, zIteration=z, noiseScale=scale,frequency=frequency,offsetX=i[0],
                                            offsetY=i[1],offsetZ=i[2], centerX=centerLocation,centerY=centerLocation, centerZ=centerLocation)
                noiseValue += noise * amplitude
                amplitude *= persistance
                frequency *= lacunarity
            
            if (noiseValue < minimumHeight):
                minimumHeight = noiseValue
            elif (noiseValue > maximumHeight):
                maximumHeight = noiseValue


            if (normalizeHeightGlobally and normalizeHeights):
                normalizedHeight = (noiseValue + 1) / (maximumHeight / 0.9)
                noiseValue = self.__Clamp(normalizedHeight, 0, sys.maxsize)
            elif(normalizeHeights):
                noiseValue = self.__InverseLerpClamped(minimumHeight, maximumHeight, noiseValue)
                
            noiseMap.append(noiseValue)

            z+=1

            if(z >= resolution):
                y+=1
                z=0
            
            if(y >= resolution):
                x+=1
                y=0

            if(x >= resolution):
                break




        return noiseMap
