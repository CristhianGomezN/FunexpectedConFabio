"""
Created on Sat Oct 28 14:08:34 2017
@author: Usuario
"""

import numpy as np
import random as rnd
import copy

class term:
    
    def __init__(self, variables, aggregated = False, coefficient = 1):
        self.var = variables
        self.coefficient = coefficient
        self.aggregated = aggregated
        
    def __getitem__(self, key):
        return self.var[key]
    
    def __len__(self):
        return len(self.var)
    
    def __str__(self):
        s = ""
        if(self.coefficient != 1):
            s = "" + str(self.coefficient)
        for sym in self.var:
            s = s + "x_" + str(sym)
        return str(s)
    
    def __repr__(self):
        s = ""
        if(self.coefficient != 1):
            s = "" + str(self.coefficient)
        for sym in self.var:
            s = s + "x_" + str(sym)
        return str(s)
    
    def __lt__(self,other):
        return self.var < other.var
    
    def __eq__(self, other):
        return self.var == other.var        
    
class block:
    
    def generateMatrix(n):
        """Returns an invertible binary matrix of size n."""
        U = np.identity(n)
        L = np.identity(n)  
        #Perform random start
        for i in range(n):
            for j in range(i+1,n):
                U[i][j]=int(rnd.randint(0,1))
                L[j][i]=int(rnd.randint(0,1))       
        M=U.dot(L)
#        np.linalg.inv(M%2)
        return M%2
    
    def generateEquations(n, mat, arr, s, automaton):
        """
        Transforms an invertible matrix into a list representation with aggregated functions(see guan). 
        
        Transforms an invertible binary matrix into representation of polynomials where each polynomial is a list 
        of terms(summed) with each term as a list of the variables in this term(may be multiple with aggregated functions)
        The aggregated functions are random.
        
        Parameters
        ----------
        n : int
            size of block
        mat: numpy.mat
            invertible binary matrix(invertible system of equations)
        arr: list
            name of the variables of the system of equations
        s : int
            indicates wheter aggregated functons can be added
        automaton: automaton 
            indicates which variable can be used in aggregated functions for these automaton and function aggregates the newly
            used variables to the list of "usable" variables
        Returns
        -------
        
            
        """
        rule = []
        for i in range(n):
            cur = []
            for j in range(n):
                if(mat[i][j] == 1):
                    cur.append(term([arr[j]], False))
            if(s > 0 and rnd.randint(0,1)):
                cur.append(term(sorted(rnd.sample(automaton.lPermitida, k = min(2, rnd.randint(1, s)))), True) ) 
            rule.append(cur)
        automaton.lPermitida = automaton.lPermitida + list(arr) 
        return rule
    
    def __init__(self, n, arr, s, automaton):
        self.n = n
        self.rulemat = block.generateMatrix(n)
        self.invmat = np.linalg.inv(self.rulemat)
        self.rule = block.generateEquations(n, self.rulemat, arr, s, automaton)
        self.arr = arr
    
    def __str__(self):
        return str(self.rule)
    
    def __repr__(self):
        return str(self.rule)
        
class automaton:
    
    def __init__(self, n, rule = None):
        self.lPermitida = []
        
        if(rule == 'copy'):
            return
        self.n = n
        self.blocks = []
        self.seq = np.arange(n)
        np.random.shuffle(self.seq)
        s = 0
        while s < n:
            if(n - s < 3):
                i = n - s
            else:
                i = rnd.randint(3, min(5, n - s))
            self.blocks.append(block(i,self.seq[s:s+i], s, self))
            s += i 
        self.blocklen = [(x.n) for x in self.blocks]
        self.key = automaton.flatten(self.blocks)
       
    
    def flatten(blocks):
        ans = []
        for i in range(len(blocks)):
            ans = ans + blocks[i].rule
        return ans
    
    def merge(A, B):
        n = len(A)
        m = len(B)
        i = 0
        j = 0
        res = []
        while i < n and j < m:
            if(A[i] < B[j]):
                res.append(A[i])
                i = i + 1
            elif(B[j] < A[i]):
                res.append(B[j])
                j = j + 1
            else:
                res.append(A[i])
                i = i + 1
                j = j + 1
        while i < n:
            res.append(A[i])
            i = i + 1
        while j < m:
            res.append(B[j])
            j = j + 1
        return res  
    
    def combineBlockRule(ruleI, ruleJ, k = 1):
        ans = []
        if(len(ruleI) == 0):
            ruleJ = copy.deepcopy(ruleJ)
            for trm in ruleJ:
                trm.coefficient = trm.coefficient*k
            return ruleJ
                               
        for i in range(len(ruleI)):
            for j in range(len(ruleJ)):
                if(ruleI[i].aggregated or ruleJ[j].aggregated):
                    ans.append(term(automaton.merge(ruleI[i], ruleJ[j]),True,k*ruleI[i].coefficient*ruleJ[j].coefficient))
                else:
                    ans.append(term(automaton.merge(ruleI[i], ruleJ[j]),False,k*ruleI[i].coefficient*ruleJ[j].coefficient))
        return ans
    
    
    def combineAutomatons(self, other, normalize):
        """Composes itself with the other automaton, the other automaton being one step back in time."""
        otherRules = other.key
        #Por alguna razón el de abajo no funciona para t > 2
        for i in range(self.n):
            replace = []
            ecn = self.key[i]
            for trm in ecn:
                add = []
                for sym in trm:
                    add = automaton.combineBlockRule(add, otherRules[sym], trm.coefficient)
                replace = replace + add
            replace.sort()
            self.key[i] = replace
        if(normalize):
            self.normalizeKey()
                
        
    def evolve(self, state):
        """Transforms the given state with this automaton rules"""
        assert len(state) == self.n
        data = list(map(lambda x : int(x) - int('0'), state))
        cypher = ""
        for ecn in self.key:
            cell=0
            for i in range(len(ecn)):
                op = 1
                trm = ecn[i]
                for j in trm:
                    op = op*data[j]
                op = op*ecn[i].coefficient
                cell = cell+op
            cypher = cypher + str(cell%2)
        return cypher
    
    def normalizeKey(self):
        for i in range(self.n):    
            self.key[i].sort()
            ecn = self.key[i]
            j = 0
            unique = []            
            while j < len(ecn):
                cur = self.key[i][j]
                coef = 0
                while j < len(ecn) and cur == self.key[i][j]:
                    coef = coef + self.key[i][j].coefficient
                    j = j + 1
                unique.append(term(cur.var, cur.aggregated,  coef))
            self.key[i] = unique    
 
    def __str__(self):
        k = 0
        s = ''
        s = s + "blocks len: " + str(self.blocklen)
        '''
        for block in self.blocks:
            s = s + "\n" + str(block.rulemat) + "\n" + str(block.invmat) + "\n"
        '''
        s = s +"\n" + str(self.seq)                
        for ecn in self.key:
            cell= ''
            for i in range(len(ecn)):
                if(i != len(ecn) - 1):
                    cell = cell + str(ecn[i]) + ' + '
                else:
                    cell = cell + str(ecn[i])
            s = s + '\ny_' + str(k) + ' = ' + cell
            k = k + 1
        
        return s            
                
class encryption:
    
    def __init__(self, n, t, rule = None):
        """Form a encryption pattern."""
        self.n = n
        self.t = t
        self.automatons = []
        for i in range(t):
            self.automatons.append(automaton(n))
        self.composite = copy.deepcopy(self.automatons[t - 1])
        self.composite2 = copy.deepcopy(self.automatons[t - 1])
        for i in range(t-1):
            self.composite.combineAutomatons(self.automatons[t-2-i], True)
        
    
    def encrypt(self, plain):
        """Encrypts give plain text by applying underlaying automaton rules"""
        return self.composite.evolve(plain), self.composite2.evolve(plain)

    def decrypt(self, ciphered):
        """doesnt work"""
        for aut in reversed(self.automatons):   
            val_range=0
            for blo in aut.blocks:
                #Producto M^-1 (y1,y2,...,yn) = (x1,x2,...,xn)
                inv_matrix = np.linalg.inv(blo.rulemat)
                vals=ciphered[val_range : val_range + blo.n]
                prod = np.matmul(inv_matrix, vals)
                val_range = val_range + blo.n
    
    def __str__(self):
        return str(self.composite)
testC = 0
n = 400
t = 3
style = ''
test = min(2**n,1000)
if(testC):
    B = encryption(n,t,style)
    aut0 = B.automatons[0]
    aut1 = B.automatons[1]
#    print("t = 0")
#    print(type(aut0))
#    print( aut0)
#    print("t = 1")
#    print( aut1)
#    print("composite: ")
#    print(B)
    print("done")
else:
    B = encryption(n,t,style)
    print("composite: ")
#    print(B)

    s0 = set([])
    for i in range(test):
        b0 = str(bin(i))[2:]
        if(len(b0) < n):
            b0 = "0"*(n-len(b0)) + b0
        e0  = B.encrypt(b0)
        s0.add(e0)
        print(b0, e0)
    print("compuesta:", str(len(s0)) + "/" + str(test))
