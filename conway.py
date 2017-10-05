import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

class game(object):    
    
    dx = [1,1,1,0,0,-1,-1,-1]
    dy = [-1,0,1,-1,1,-1,0,1]
    
    def __init__(self, rule, birth, N, T):
        board = np.matrix([[0]*N]*N)
        self.history=np.array([board]*T)
        self.population=[0]*T
        self.rule=np.array(rule)
        self.birth=np.array(birth)
        self.N=N
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

    def cellSum(self,i,j):
        aux=self.history[self.t]
        sums=0
        for k in range(8):
            x = i + game.dx[k]
            y = j + game.dy[k]
            if(x >= 0 and x < self.N and y >= 0 and y < self.N and aux[x][y]):
                sums=sums+1
            
        return sums
        
    def update(self):
        aux=self.history[self.t]
        self.population[self.t+1]=self.population[self.t]
        for i in range (self.N):
            for j in range (self.N):
                sums=self.cellSum(i,j)
                if(aux[i][j]):
                    self.history[self.t+1][i][j] = self.rule[sums]
                    if (self.rule[sums]==False):
                        self.population[self.t+1]=self.population[self.t+1]-1
                elif(self.birth[sums]):
                    self.history[self.t+1][i][j] = True
                    self.population[self.t+1]=self.population[self.t+1]+1
        self.t=self.t+1

        
    
    #plots
    def timePlot(self,time):
        fig, ax = plt.subplots()
        image = self.history[time]
        ax.imshow(image, cmap='binary')
        plt.title("t = "+str(time)+", population = "+str(self.population[time]))
        plt.show()
    
    def populationPlot(self):
        plt.subplots()
        x = np.linspace(0,self.t,self.t+1)
        plt.plot(x,self.population)
        
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
game0 = game([0,0,1,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],size,steps)
game0.initialize("rand")

#simulate
for i in range (steps-1):
    game0.update();

#show
#for i in range (steps):
#    game0.timePlot(i)
game0.timePlot(0)
game0.timePlot(steps-1)
game0.populationPlot()
game0.export('t1.csv')
