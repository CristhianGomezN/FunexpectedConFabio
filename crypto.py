"""
Created on Sat Oct 28 14:08:34 2017

@author: Usuario
"""

import numpy as np
import random as rnd

class block:
    
    '''
    
    Retorna una matrix invertible
    '''
    def generarmatriz(n):
        U = np.identity(n);
        L = np.identity(n);  
        #Perform random start
        for i in range(n):
            for j in range(i+1,n):
                U[i][j]=rnd.randint(0,1)
                L[j][i]=rnd.randint(0,1)        
        M=U.dot(L)
        np.linalg.inv(M%2)
        return M%2
    
    '''
    Retorna la representación de la función basada en una matriz invertible 
    '''
    def generarfuncion(n, mat, arr, s, automaton):
        rule = []
#        print(s, automaton.lPermitida ,len(automaton.lPermitida))
        for i in range(n):
            cur = []
            for j in range(n):
                if(mat[i][j]):
                    add = [arr[j]]
                    if(s > 0 and rnd.randint(0,1)):
                        add = add + rnd.sample(automaton.lPermitida, k = rnd.randint(0, s))
                    cur.append(add)
            rule.append(cur)
        automaton.lPermitida = automaton.lPermitida + list(arr) 
        return rule
    
    '''
    Defines a block 
    '''
    def __init__(self, n, arr, s, automaton):
        self.n = n
        self.rulemat = block.generarmatriz(n)
        self.rule = block.generarfuncion(n, self.rulemat, arr, s, automaton)
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
#            print("g_A",rule)
            return 
        elif(rule == 'B'):
            a = block(2, [3,4], 0, self)
            a.rule = [[[3]], [[4]]]
            a.rulemat = np.array([[1,0],[0,1]])
            b = block(3, [0,1,2], 3, self)
            b.rule = [[[0], [3,4]], [[1], [0,3]], [[2], [0]]]
            b.rulemat = np.array([[1,0,0],[1,1,0],[1,0,1]])
            self.n = 5
            self.blocks = [a, b]
            self.seq = [3,4,0,1,2]
            self.key = automaton.flatten(self)
#            print("g_B",rule)
            return
        elif(rule == 'copy'):
            return
        self.n = n
        self.blocks = []
        self.seq = np.arange(n)
        np.random.shuffle(self.seq)
        s = 0
        while s < n:
#            print("<")
            if(n - s < 3):
                i = n - s
            else:
                i = rnd.randint(3, min(5, n - s))
#            print("i:", i)
            self.blocks.append(block(i,self.seq[s:s+i], s, self))
            s += i 
#            print(">")
        self.key = automaton.flatten(self)
#        print()
        
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
        ruleI.sort()
        ruleJ.sort()
        ans = []
        for i in range(len(ruleI)):
            for j in range(len(ruleJ)):
                ans.append(automaton.merge(ruleI[i], ruleJ[j]))
        return ans
    
    
    '''
    Combina dos automatones
    @param otherAutomaton: automaton en un tiempo anterior a el actual
    '''
    def combineAutomatons(self, otherAutomaton):
        #thisRules = automaton.flatten(self)
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
        s = s +"\n" + str(self.seq) + "\n"                     
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
    
    '''
    @param n tamaño de cada automaton
    @param t Numero de reglas
    '''
    def __init__(self, n, t, rule = None):
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
        return self.composite.evolve(plain)

    def decrypt(self, ciphered):
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
            
B = encryption(5,2)
aut0 = B.automatons[0]
aut1 = B.automatons[1]
print(B.automatons[0])
print(B.automatons[1])
print(B)

s0 = set([])
s1 = set([])
s2 = set([])
for i in range(32):
    b0 = str(bin(i))[2:]
    if(len(b0) < 5):
        b0 = "0"*(5-len(b0)) + b0
    e0 = B.encrypt(b0)
    e1 = aut0.evolve(b0)
    e2 = aut1.evolve(b0)
    s0.add(e0)
    s1.add(e1)
    s2.add(e2)
    print("plaintext: ", b0,"\n cyphertext0: ", e0,"\n cyphertext1: ", e1,"\n cyphertext2: ", e2)    

print("compuesta:", len(s0), "t=0", len(s1),"t=1", len(s2))
