# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:39:50 2017

@author: fabiotovar
"""

class Util:
    @staticmethod
    def binaryStrToBits(s):
        return list(map(lambda x : int(x) - int('0'), s))
        
    @staticmethod
    def bitsToBinaryStr(s):
        result = ''
        for i in range(len(s)):
            result = result + str(s[i])
        return result
        
    @staticmethod
    def stringToBits(s):
        result = []
        for c in s:
            bits = bin(ord(c))[2:]
            bits = '00000000'[len(bits):] + bits
            result.extend([int(b) for b in bits])
        return result
        
    @staticmethod
    def bitsToString(bits):
        chars = []
        for b in range(len(bits) / 8):
            byte = bits[b*8:(b+1)*8]
            chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return ''.join(chars)
        
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))