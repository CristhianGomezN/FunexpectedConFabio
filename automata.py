import numpy as np
import matplotlib.pyplot as plt
import re

class game:    
    
    dx = [1,1,1,0,0,-1,-1,-1]
    dy = [-1,0,1,-1,1,-1,0,1]
    
    rules = { 'diamonds':'5678/35678',
             'standard':'23/3',
             'highlife':'23/36',
             'diffusion':'2/7',
             'replicant':'1357/1357',
             'death':'245/368',
             'longlife':'15/346',
             'ameba':'1358/357',
             'stain':'235678/3678',
             'spark':'/3',
             'growth':'34/34',
             'carpet':'4/2'}
    
    @staticmethod
    def getrule(rule):
        rule = rule.split('/')
        survive = [False]*9
        birth = [False]*9
        for x in rule[0]:
            survive[ord(x) - ord('0')] = True
        for x in rule[1]:
            birth[ord(x) - ord('0')] = True
        return survive, birth
    
    def __init__(self, N, T, rule = 'standard'):
        if not(re.match("[0-8]/[0-8]",rule)) and(rule in game.rules.keys()) :
            rule = game.rules[rule]
        board = np.matrix([[False]*N]*N)
        self.history = np.array([board]*(T+1));
        self.rule, self.birth = self.getrule(rule)
        self.rule = np.array(self.rule)
        self.birth = np.array(self.birth)
        self.N = N
        self.t = 0
    
    
    def generate(self):
        self.history[0] = np.random.choice(a=[False, True], size=(self.N, self.N))
#        self.history[0][24][10] = True
#        self.history[0][25][10] = True
#        self.history[0][24][11] = True
#        self.history[0][25][11] = True
#        
#        self.history[0][22][44] = True
#        self.history[0][23][44] = True
#        self.history[0][22][45] = True
#        self.history[0][23][45] = True
#        
#        self.history[0][24][20]=True
#        self.history[0][25][20]=True
#        self.history[0][26][20]=True
#        
#        self.history[0][23][21]=True
#        self.history[0][27][21]=True
#        
#        self.history[0][22][22]=True
#        self.history[0][28][22]=True
#        
#        self.history[0][22][23]=True
#        self.history[0][28][23]=True
#        
#        self.history[0][25][24]=True
#        
#        self.history[0][23][25]=True
#        self.history[0][27][25]=True
#        
#        self.history[0][24][26]=True
#        self.history[0][25][26]=True
#        self.history[0][26][26]=True
#        
#        self.history[0][25][27]=True
#        
#        self.history[0][22][30]=True
#        self.history[0][23][30]=True
#        self.history[0][24][30]=True
#        
#        self.history[0][22][31]=True
#        self.history[0][23][31]=True
#        self.history[0][24][31]=True
#        
#        self.history[0][21][32]=True
#        self.history[0][25][32]=True
#        
#        self.history[0][20][34]=True
#        self.history[0][21][34]=True
#        self.history[0][25][34]=True
#        self.history[0][26][34]=True
        
    def timePlot(self,time):
        fig, ax = plt.subplots()
        image = self.history[time]
        ax.imshow(image, cmap=plt.cm.gray)
        plt.title("t = "+str(time))
        print()
        plt.show()
        
    def cellSum(self,i,j):
        aux=self.history[self.t]
        sums=0
        for k in range(8):
            x = i + game.dx[k]
            y = j + game.dy[k]
            if(x >= 0 and x < self.N and y >= 0 and y < self.N and aux[x][y]):
                sums=sums+1
        return sums
    '''
    retorna el numero de celulas que quedaron vivas después de la actualización
    retorna 0 si el estado anterior es igual al nuevo
    '''
    def update(self):
        aux=self.history[self.t]
        cnt = 0
        equal = True
        for i in range (self.N):
            for j in range (self.N):
                sums=self.cellSum(i,j)
                if(aux[i][j]):
                    self.history[self.t+1][i][j] = self.rule[sums]
                elif(self.birth[sums]):
                    self.history[self.t+1][i][j] = True
                if(self.history[self.t+1][i][j]):
                    cnt = cnt + 1
                if(aux[i][j] != self.history[self.t+1][i][j]):
                    equal = False
        self.t=self.t+1
        if(equal):
            cnt = 0
        return cnt

t = 100

game0 = game(50,t, 'death')
game0.generate()

for i in range (t):
    if game0.update() == 0:
        break
for i in range(t):
    game0.timePlot(i)
    if(i >= game0.t):
        break

