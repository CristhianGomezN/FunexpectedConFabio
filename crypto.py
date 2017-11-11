"""
Created on Sat Oct 28 14:08:34 2017
@author: Usuario
"""

import numpy as np
import random as rnd

class term:
    
    def __init__(self, variables, aggregated):
        self.var = variables
        self.aggregated = aggregated
        
    def __getitem__(self, key):
        return self.var[key]
    
    def __len__(self):
        print(self.var)
        return len(self.var)
    
    def __str__(self):
        return str(self.var)
        
    
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
                cur.append(term(sorted(rnd.sample(automaton.lPermitida, k = min(2, rnd.randint(0, s)))), True) ) 
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
        
        if(rule == 'A'):
            a = block(3, [0,1,2], 0, self)
            a.rule = [[ [1] ], [[2]],[[0]]]
            a.rulemat=np.array([[0,1,0],[0,0,1],[1,0,0]])
            b = block(2, [3,4], 3, self)
            b.rule = [[[4], [0,1]], [[3], [1,2]]]
            b.rulemat=np.array([[0,1],[1,0]])
            self.n = 5
            self.blocks = [a, b]
            self.seq = [0,1,2,3,4]    
            self.key = automaton.flatten(self)
            return 
        elif(rule == 'B'):
            a = block(2, [3,4], 0, self)
            a.rule = [[[3]], [[4]]]
            a.rulemat = np.array([[1,0],[0,1]])
            b = block(3, [0,1,2], 2, self)
            b.rule = [[[0]], [[1], [0]], [[2], [0]]]
            b.rulemat = np.array([[1,0,0],[1,1,0],[1,0,1]])
            self.n = 5
            self.blocks = [a, b]
            self.seq = [3,4,0,1,2]
            self.key = automaton.flatten(self)
            return
        elif(rule == 'copy'):
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
        self.key = automaton.flatten(self)
        
    def copy(self):
        aut = automaton(self.n,'copy')
        aut.blocks = self.blocks
        aut.n = self.n
        aut.seq = self.seq
        aut.key = self.key
        return aut
    
    def flatten(otherAutomaton):
        ans = []
        blocks = otherAutomaton.blocks
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
    
    def combineBlockRule(ruleI, ruleJ):
        if(len(ruleI) == 0):
            return ruleJ
        ans = []
        for i in range(len(ruleI)):
            for j in range(len(ruleJ)):
                print("type", type(ruleI[i]))
                if(ruleI[i].aggregated or ruleJ[j].aggregated):
                    ans.append(term(automaton.merge(ruleI[i], ruleJ[j]), True))
                else:
                    ans.append(term(automaton.merge(ruleI[i], ruleJ[j]),False))
        return ans
    
    
    def combineAutomatons(self, otherAutomaton):
        """Composes itself with the other automaton, the other automaton being one step back in time."""
        otherRules = otherAutomaton.key
        #cada bloque [[{},...,{}],...,[{},...,{}]]
        for i in range(len(self.blocks)):
            rule = self.blocks[i].rule    
            #cada elemento j de un bloque [{},...,{}] 
            for j in range(self.blocks[i].n):
                replace = []
                #cada {}
                for st in rule[j]:     
                    add = []
                    #cada elemento de {x1,..., xn}
                    for l in st:
                        add = automaton.combineBlockRule(add, otherRules[l])    
                    replace = replace + add
                self.blocks[i].rule[j] = replace
    
    def evolve(self, state):
        """Transforms the given state with this automaton rules"""
        assert len(state) == self.n
        data = list(map(lambda x : int(x) - int('0'), state))
        cypher = ""
        for ecn in self.key:
            cell=0
            for i in range(len(ecn)):
                op = 1
                for j in ecn[i]:
                    op = op*data[j]
                cell = cell+op
            cypher = cypher + str(cell%2)
        return cypher
 
    def __str__(self):
        k = 0
        s = ''
        s = s + "blocks len: " + str([(x.n) for x in self.blocks])
        '''
        for block in self.blocks:
            s = s + "\n" + str(block.rulemat) + "\n" + str(block.invmat) + "\n"
        '''
        s = s +"\n" + str(self.seq)                     
        for ecn in self.key:
            cell= ''
            for i in range(len(ecn)):
                cur= ''
                for j in ecn[i]:
                    cur = cur+'x_'+str(j)
                if(i != len(ecn) - 1):
                    cell = cell + cur + ' + '
                else:
                    cell = cell + cur
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
        if(rule == 'guan'):
            self.n = 5
            self.t = 2
            self.automatons = [automaton(5,'A'), automaton(5,'B')]
        self.composite = self.automatons[t - 1].copy()
        for i in range(t-1):
            automaton.combineAutomatons(self.composite, self.automatons[t - 2 - i])
        self.composite.key = automaton.flatten(self.composite)
    
    def encrypt(self, plain):
        """Encrypts give plain text by applying underlaying automaton rules"""
        return self.composite.evolve(plain)

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

'''n = 10
test = min(2**n,10000)
B = encryption(n,2)
aut0 = B.automatons[0]
aut1 = B.automatons[1]
print("t = 0")
print( B.automatons[0])
print("t = 1")
print( B.automatons[1])
print("composite: ")
print(B)'''

n = 5
U = np.identity(n)
L = np.identity(n)  
#Perform random start
for i in range(n):
    for j in range(i+1,n):
        U[i][j]=int(rnd.randint(0,1))
        L[j][i]=int(rnd.randint(0,1))       
M=U.dot(L)
print(M)
M = M%2
print("matrix: ",M)
N = np.linalg.inv(M)
print("inverse:", N)
