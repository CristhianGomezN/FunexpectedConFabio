""""
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
    
    def merge(self, other, k = 1):
        n = len(self)
        m = len(other)
        i = 0
        j = 0
        res = []
        while i < n and j < m:
            if(self[i] < other[j]):
                res.append(self[i])
                i = i + 1
            elif(other[j] < self[i]):
                res.append(other[j])
                j = j + 1
            else:
                res.append(self[i])
                i = i + 1
                j = j + 1
        while i < n:
            res.append(self[i])
            i = i + 1
        while j < m:
            res.append(other[j])
            j = j + 1
        return term(res, self.aggregated, k)
        
    def __getitem__(self, key):
        return self.var[key]
    
    def eval(self, vals):
        ans = 1
        for sym in self:
            ans = ans*vals[sym]
        return ans%2
    
    def __len__(self):
        return len(self.var)
    
    def __str__(self):
        s = ""
        if(self.coefficient != 1):
            s = "" + str(self.coefficient)
        for sym in self.var:
            s = s + "x_" + str(sym + 1)
        return str(s)
    
    def __repr__(self):
        return str(self)
    
    def __lt__(self,other):
        return self.var < other.var
    
    def __eq__(self, other):
        return self.var == other.var        
    
class polynomial:
    
    def __init__(self, terms = []):
        self.terms = terms
        
    def __len__(self):
        return len(self.terms)
    
    def __add__(self, other):
        return  polynomial(self.terms + other.terms)
    
    
    def multp(self, other, k = 1):
        ans = []
        if(len(self) == 0):
            ans = copy.deepcopy(other.terms)
            for trm in ans:
                trm.coefficient = trm.coefficient*k
        else:
            for i in range(len(self)):
                for j in range(len(other)):
                    coef = k*self[i].coefficient*other[j].coefficient
                    if(coef%2 == 1):
                        ans.append(self[i].merge(other[j], coef))                        
        self.terms = ans
    
    def normalize(self):
        self.terms.sort()
        ecn = self.terms
        i = 0
        unique = []            
        while i < len(ecn):
            cur = self.terms[i]
            coef = 0
            while i < len(ecn) and cur == self.terms[i]:
                coef = coef + self.terms[i].coefficient
                i = i + 1
            if(coef%2 == 1):    
                unique.append(term(cur.var, cur.aggregated,  coef%2))
        self.terms = unique
    
    def sort(self):
        self.terms.sort()
    
    def __str__(self):
        s = ''
        for i in range(len(self)):
            if i != len(self) - 1:
                s = s + str(self.terms[i]) + ' + '
            else:
                s = s + str(self.terms[i])
        return s
    
    def __getitem__(self, key):
        return self.terms[key]
    
    def __repr__(self):
        return str(self)
    
    
class block:
    def __init__(self, n, arr, s, automaton):
        self.n = n
        self.rulemat, self.inv = block.generateMatrix(n)
        self.invmat = np.linalg.inv(self.rulemat)
        self.rule = block.generateEquations(n, self.rulemat, arr, s, automaton)
        self.arr = arr    
    
    @staticmethod
    def generateMatrix(n):
        """Returns an invertible binary matrix of size n."""
        U = np.identity(n)
        L = np.identity(n)  
        #Perform random start
        for i in range(n):
            for j in range(i+1,n):
                U[i][j]=int(rnd.getrandbits(1))
                L[j][i]=int(rnd.getrandbits(1))       
        M=U.dot(L)
        M = M%2
        Mi = np.linalg.inv(M)
        Mi = np.around(Mi/np.linalg.det(Mi)).astype(int)%2
        check = np.matmul(M,Mi)
        check = np.around(check).astype(int)
        check = check%2
        assert np.array_equal(check, np.around(np.identity(n)).astype(int))
        return M, Mi
    
    @staticmethod
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
            if(s > 0 and rnd.getrandbits(1)):
                cur.append(term(sorted(rnd.sample(automaton.lPermitida, k = min(2, rnd.randint(1, s)))), True) ) 
            rule.append(polynomial(cur))
        automaton.lPermitida = automaton.lPermitida + list(arr) 
        return rule
    
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
       
    @staticmethod
    def flatten(blocks):
        ans = []
        for i in range(len(blocks)):
            ans = ans + blocks[i].rule
        return ans
    
    def compose(self, other, normalize):
        """Composes itself with the other automaton, the other automaton being one step back in time."""
        for i in range(self.n):
            replace = polynomial()
            ecn = self.key[i]
            for trm in ecn:
                add = polynomial()
                for sym in trm:
                    add.multp(other.key[sym], trm.coefficient)
                replace = replace + add
            replace.sort()
            self.key[i] = replace
        if(normalize):
            self.normalizeKey()
                
    def evolve(self, state):
        """Transforms the given state with this automaton rules"""
        cypher = []
        for ecn in self.key:
            cell=0
            for i in range(len(ecn)):
                op = 1
                trm = ecn[i]
                for j in trm:
                    op = op*state[j]
                op = op*ecn[i].coefficient
                cell = cell+op
            cypher.append(cell%2)
        return cypher
    
    def normalizeKey(self):
        for poly in self.key:
            poly.normalize()
 
    def __str__(self):
        k = 0
        s = ''
        #s = s + "blocks len: " + str(self.blocklen)
        '''
        for block in self.blocks:
            s = s + "\n" + str(block.ruleevolvemat) + "\n" + str(block.invmat) + "\n"
        '''
        #s = s +"\n" + str(self.seq)                
        for ecn in self.key:
            cell= ''
            for i in range(len(ecn)):
                if(i != len(ecn) - 1):
                    cell = cell + str(ecn[i]) + ' + '
                else:
                    cell = cell + str(ecn[i])
            s += 'y_' + str(k) + ' = ' + cell + '\n'
            k += 1 
        
        return s            
                
class encryption:
    
    def __init__(self, n, t, rule = None):
        """Form a encryption pattern."""
        self.automatons = []
        if(rule == 'guan'):
            self.n = 5
            self.t = 2
            aut0 = automaton(5)
            a = block(3, [0,1,2], 0, aut0)
            a.rule = [polynomial([ term([1],False) ]), polynomial([term([2],False)]),polynomial([term([0],False)])]
            a.rulemat=np.array([[0,1,0],[0,0,1],[1,0,0]])
            a.inv = np.linalg.inv(a.rulemat)
            a.inv = np.around(a.inv/np.linalg.det(a.inv)).astype(int)%2
            b = block(2, [3,4], 3, aut0)
            b.rule = [polynomial([term([4],False), term([0,1],True)]), polynomial([term([3],False), term([1,2],True)])]
            b.rulemat=np.array([[0,1],[1,0]])    
            b.inv = np.linalg.inv(b.rulemat)
            b.inv = np.around(b.inv/np.linalg.det(b.inv)).astype(int)%2
            
            aut0.blocks = [a, b]
            aut0.seq = [0,1,2,3,4]    
            aut0.key = automaton.flatten(aut0.blocks)
            self.automatons.append(aut0)

            
            
            ######################################
            aut1 = automaton(5)
            a = block(2, [3,4], 0, aut1)
            a.rule = [polynomial([term([3],False)]), polynomial([term([4],False)])]
            a.rulemat = np.array([[1,0],[0,1]]) 
            a.inv = np.linalg.inv(a.rulemat)
            a.inv = np.around(a.inv/np.linalg.det(a.inv)).astype(int)%2
            b = block(1, [0], 2, aut1)
            b.rule = [polynomial([term([0],False), term([3,4], True)])] 
            b.rulemat = np.array([[1]])
            b.inv = np.linalg.inv(b.rulemat)
            b.inv = np.around(b.inv/np.linalg.det(b.inv)).astype(int)%2
            c = block(2, [1,2], 3, aut1)
            c.rule = [polynomial([term([1],False), term([0,3], True)]), polynomial([term([2],False), term([0], True)])]
            c.rulemat = np.array([[1,0],[0,1]])
            c.inv = np.linalg.inv(c.rulemat)
            c.inv = np.around(c.inv/np.linalg.det(c.inv)).astype(int)%2
            aut1.blocks = [a, b, c]
            aut1.seq = [3,4,0,1,2]
            aut1.key = automaton.flatten(aut1.blocks)
            self.automatons.append(aut1)
        else:
            self.n = n
            self.t = t    
            for i in range(t):
                self.automatons.append(automaton(n))
        
        self.composite = copy.deepcopy(self.automatons[t - 1])
        for i in range(t-1):
            self.composite.compose(self.automatons[t-2-i], True)
    
    def encrypt(self, plain):
        """Encrypts give plain text by applying underlaying automaton rules"""
        assert self.n == len(plain)
        #plain = list(map(lambda x : int(x) - int('0'), plain))
        return self.composite.evolve(plain)

    def decrypt(self, ciphered):
        """now works"""
        assert self.n == len(ciphered)
        #ciphered = list(map(lambda x : int(x) - int('0'), ciphered))
        i = 0
        for aut in reversed(self.automatons):
            val_range=0
            sol = [0]*self.n            
            for blo in aut.blocks:
                inv_matrix = blo.inv
                inv_matrix = inv_matrix.astype(int)
                for x in range(val_range, val_range + blo.n):
                    for trm in aut.key[x]:
                        if(trm.aggregated):
                            ciphered[x] = (ciphered[x] + trm.eval(sol))%2
                prod = np.matmul(inv_matrix, ciphered[val_range:val_range + blo.n])
                prod = list(prod)
                k = 0
                for x in blo.arr:
                    sol[x] = prod[k]%2
                    k = k + 1
                val_range = val_range + blo.n
            ciphered = copy.copy(sol)
            i = i + 1
        return sol
    
    def __str__(self):
        return str(self.composite)
    

def rndbin(n):
    b0 = bin(rnd.getrandbits(n))[2:]
    if(len(b0) < n):
        b0 = '0'*(n-len(b0)) + b0
    return b0

def binstr(i,n):
    res = [int(d) for d in str(bin(i))[2:]] 
    add = [0] * (n - len(res))
    add.extend(res)
    return add

#testC = 0
#n = 5
#t = 2
#style = 'guan'
#guanTest = True
#test = min(2**n,10000)
#testRan = 100
# 
#C = encryption(5,2, 'guan')
#b0 = [1, 0, 1, 1, 0]
#e0 = C.encrypt(b0)
#d0 = C.decrypt(e0)
#
#print "PRIVATE"
#for i in C.automatons:
#    print i
#    print "Blocks", i.blocks
#    for j in i.blocks:
#        print "rule math", j.rulemat
#        print "rule", j.rule
#    
#print "\n---------------------------\nPUBLIC"
#print(C.composite)