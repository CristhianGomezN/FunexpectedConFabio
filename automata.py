import numpy as np
import matplotlib.pyplot as plt
import re
import csv

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
            survive[ord(x) - ord('0')] = 1
        for x in rule[1]:
            birth[ord(x) - ord('0')] = 1
        return survive, birth
    
    def __init__(self, N, T, rule = 'standard'):
        if not(re.match("[0-8]/[0-8]",rule)) and(rule in game.rules.keys()) :
            rule = game.rules[rule]
        else:
            rule = game.rules['standard']
        board = np.matrix([[0]*N]*N)
        self.population=[0]*(T+1)
        self.history = np.array([board]*(T+1));
        self.rule, self.birth = self.getrule(rule)
        self.rule = np.array(self.rule)
        self.birth = np.array(self.birth)
        self.N = N
        self.t = 0

    def rand(self):
        self.history[0]=np.random.choice(a=[0, 1], size=(self.N, self.N))
        pt=np.unique(self.history[0],return_counts=1)
        self.population[0]=pt[1][1]
    
    def logistic(self):
        matrixSize=(self.N*self.N)+1
        u1=np.random.uniform(3.5,4.0)
        u2=np.random.uniform(3.5,4.0)
        X=[0]*matrixSize
        Y=[0]*matrixSize
        X[0]=np.random.uniform(0.0,1.0)
        Y[0]=np.random.uniform(0.0,1.0)
#        print("U1 = ",u1,"U2 = ",u2,"X0 = ",X[0],"Y0 = ",Y[0])
        for i in range (1,matrixSize):
            X[i]=u1*(X[i-1]*(1-X[i-1]))
            Y[i]=u2*(Y[i-1]*(1-Y[i-1]))
#            print("X[",i,"] = ",X[i])
#            print("Y[",i,"] = ",Y[i])
        
        for i in range(1,matrixSize):
            if(X[i]>Y[i]):
                self.history[0][(i-1)%self.N][(i-1)//self.N]=0
            else:
                self.history[0][(i-1)%self.N][(i-1)//self.N]=1
                self.population[0]=self.population[0]+1;
    
    def initialize(self, method):
        func = getattr(self,method)
        func()
        
    def timePlot(self,time):
        fig, ax = plt.subplots()
        image = self.history[time]
        ax.imshow(image, cmap=plt.cm.gray)
        plt.title("t = "+str(time)+", population = "+str(self.population[time]))
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
                    self.history[self.t+1][i][j] = 1
                if(self.history[self.t+1][i][j]):
                    cnt = cnt + 1
                if(aux[i][j] != self.history[self.t+1][i][j]):
                    equal = False
        self.t=self.t+1
        self.population[self.t] = cnt
        if(equal):
            cnt = 0
        return cnt

    def populationPlot(self):
        plt.subplots()
        x = np.linspace(0,self.t,self.t+1)
        plt.plot(x,self.population[0:self.t+1])
        plt.xlabel('time(t)')
        plt.ylabel('population(cells)')
        plt.title('Population over the time')
        plt.grid(True)
        plt.show()
        

    def export(self,filename):       
        f = open(filename,'wt')
        siz=self.N*self.N
        rowNames=["step."]*(siz)
        for i in range (siz):
            rowNames[i]=rowNames[i]+str(i+1)
        try:
            writer = csv.writer(f,dialect='excel')
            writer.writerow((np.append(['id','delta'],rowNames)))
            for i in range(self.t):
                flat=np.ndarray.tolist(self.history[i].flatten())
                writer.writerow((np.append([i+1,i],flat)))
        finally:
            f.close()

#parameters
size=20
steps=1000
        
#game is an object
game0 = game(size,steps)
game0.initialize("rand")

#simulate
for i in range (steps):
    if(game0.update() == 0):
        steps = i
        break

#show
#for i in range (steps):
#    game0.timePlot(i)
game0.timePlot(0)
game0.timePlot(steps)
game0.populationPlot()
game0.export('t1.csv')
