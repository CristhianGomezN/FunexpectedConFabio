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
                L[i][j]=rnd.randint(0,1)
        M=U.dot(L.transpose())
        return M%2
    
    '''
    Retorna la representación de la función basada en una matriz invertible 
    '''
    def generarfuncion(n, mat, arr, s):
        rule = []
        for i in range(n):
            cur = []
            for j in range(n):
                if(mat[i][j]):
                    add = [arr[j]]
                    if(s > 0 and rnd.randint(0,1)):
                        add = add + rnd.sample(range(s),k=rnd.randint(0,s - 1))
                    cur.append(set(add))
            rule.append(cur)            
        return rule
    
    '''
    Defines a block 
    '''
    def __init__(self, n, arr, s):
        self.n = n
        self.rulemat = block.generarmatriz(n)
        self.rule = block.generarfuncion(n, self.rulemat, arr, s)
        self.arr = arr
        
class automaton:
    
    def __init__(self, n, rule = None):
        if(rule == 'A'):
            a = block(3, [0,1,2], 0)
            a.rule = [[set([1])],[set([2])],[set([0])]]
            a.rulemat=np.array([[0,1,0],[0,0,1],[1,0,0]])
            b = block(2, [4,5], 4)
            b.rule = [[set([4]), set([0,1])], [set([3]), set([1,2])]]
            b.rulemat=np.array([[0,1],[1,0]])
            self.n = 5
            self.blocks = [a, b]
            return 
        elif(rule == 'B'):
            a = block(2, [3,4], 0)
            a.rule = [[set([3])], [set([4])]]
            a.rulemat = np.array([[1,0],[0,1]])
            b = block(3, [0,1,2], 3)
            b.rule = [[set([0]), set([3,4])], [set([1]), set([0,3])], [set([2]), set([0])]]
            b.rulemat = np.array([[1,0,0],[1,1,0],[1,0,1]])
            self.n = 5
            self.blocks = [a, b] 
            return
        self.n = n
        self.blocks = []
        seq = np.arange(n)
        np.random.shuffle(seq)
        s = 0
        while s < n:
            if(n - s == 1 ):
                self.blocks.append(block(2,seq[n-2:n], s))
                s += 2
            else:
                i = rnd.randint(1, min(5, n - s - 1))
                self.blocks.append(block(i,seq[s:s+i], s))
                s += i
    
    def flatten(otherAutomaton):
        ans = []
        blocks = otherAutomaton.blocks
        for i in range(len(blocks)):
            ans = ans + blocks[i].rule
        return ans
    
    def combineBlockRule(ruleI, ruleJ):
        if(len(ruleI) == 0):
            return ruleJ
        ans = []
        for i in range(len(ruleI)):
            for j in range(len(ruleJ)):
                ans.append(ruleI[i].union(ruleJ[j]))
        return ans
        
    '''
    Combina dos automatones
    @param otherAutomaton: automaton en un tiempo anterior a el actual
    '''
    def combineAutomatons(self, otherAutomaton):
        #thisRules = automaton.flatten(self)
        otherRules = automaton.flatten(otherAutomaton)      
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
                    print(">", replace, add)
                    replace = replace + add
                self.blocks[i].rule[j] = (np.unique(replace)).tolist()
                        
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
        self.key = self.automatons[t - 1]
        for i in range(t-1):
            automaton.combineAutomatons(self.key, self.automatons[t - 2 - i])
        self.key = automaton.flatten(self.key)
                                       
    def encrypt(self, plain):
        data = list(map(lambda x : int(x) - int('0'), plain))
        ciphered = []
        for ecn in self.key:
            cell=0
            for i in range(len(ecn)):
                op = 1
                for j in ecn[i]:
                    op = op*data[j]
                cell = cell+op
            ciphered.append(cell%2)
        return ciphered

    def decrypt(self, ciphered):
        for aut in reversed(self.automatons):   
            val_range=0
            for blo in aut.blocks:
                #Producto M^-1 (y1,y2,...,yn) = (x1,x2,...,xn)
                inv_matrix = np.linalg.inv(blo.rulemat)
                vals=ciphered[val_range : val_range + blo.n]
                prod = np.matmul(inv_matrix, vals)
                val_range = val_range + blo.n
    
    def publickey(self):
#        publickey = []
        k = 0                        
        for ecn in self.key:
            cell=''
            for i in range(len(ecn)):
                cur=''
                for j in ecn[i]:
                    cur = cur+'x_'+str(j)
                if(i != len(ecn) - 1):
                    cell = cell + cur + ' + '
                else:
                    cell = cell + cur
            print('y_' + str(k) + '=', cell)
            k = k + 1
#            publickey.append(cell)
            
'''            
n = 5
t = 2
B = encryption(n, t)
#autoA = automaton(5)
#autoB = automaton(5)
#autoA.combineAutomatons(autoB)
#B.encrypt("10110")
C = "10110"
#D = list(map(lambda x : int(x) - int('0'), C))
#print(D)

D=B.encrypt(C)
E=B.decrypt(D)
'''

guan = encryption(400, 2)
guan.publickey()   
