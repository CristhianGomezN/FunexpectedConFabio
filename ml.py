import numpy as np
#import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import csv


Xdata = []
Ydata = []
with open('data.csv') as csvfile:
    fieldNames = ['ch','steps','plain']
    read = csv.DictReader(csvfile, fieldnames = fieldNames)
    
    for row in read:
        Xaux = row['ch']
        Yaux = row['plain']
        aux = []
        auxa = []
        for i in range(len(Xaux)):
            aux.append(int(Xaux[i]))
            auxa.append(int(Yaux[i]))
        Xdata.append(aux)
        Ydata.append(auxa)
Xdata = np.array(Xdata)
Ydata = np.array(Ydata)

text_file = open("Output.txt", "w")
cn = 1

for i in range (400):
    ######################## prepare the data ########################
    X = Xdata
    y = []
    for j in range (len(Ydata)):
        y.append([Ydata[j][i]])
        
    y = np.array(y)
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size = 0.33)
    
    
    ######################## set learning variables ##################
    learning_rate = 0.000001
    epochs = 50
    batch_size = 3
    
    
    ######################## set some variables #######################
    x = tf.placeholder(tf.float32, [None, 400], name = 'x')   # 3 features
    y = tf.placeholder(tf.float32, [None, 1], name = 'y')   # 2 outputs
    keep_prob = tf.placeholder("float")
    
    
    def multilayer_perceptron(x,weights,biases):
        layer_1 = tf.add(tf.matmul(x,weights['W1']), biases['b1'])
        layer_1= tf.nn.relu(layer_1)
        layer_1_do = tf.nn.dropout(layer_1, keep_prob)
        
        
        layer_2 = tf.add(tf.matmul(layer_1_do,weights['W2']), biases['b2'])
        layer_2= tf.nn.relu(layer_2)
        
        out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
        
        return out_layer
    
    
    weights = {
            'W1': tf.Variable(tf.random_normal([400, 200])),
            'W2': tf.Variable(tf.random_normal([200, 50])),
            'out': tf.Variable(tf.random_normal([50, 1]))
            }
    
    biases = {
            'b1': tf.Variable(tf.random_normal([200])),
            'b2': tf.Variable(tf.random_normal([50])),
            'out': tf.Variable(tf.random_normal([1]))
            }
     
        ####################### Loss Function  #########################
    y_ = multilayer_perceptron(x,weights,biases)
    mse = tf.losses.mean_squared_error(y, y_)
        
        
        ####################### Optimizer      #########################
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(mse)  
        
        
        ###################### Initialize, Accuracy and Run #################
        # initialize variables
    init_op = tf.global_variables_initializer()
        
        # accuracy for the test set
    accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_))) # or could use tf.losses.mean_squared_error
    #run
    with tf.Session() as sess:
         sess.run(init_op)
         total_batch = int(len(y_train) / batch_size)  
         Vals = []
         print(cn)
         
         for epoch in range(epochs):
             avg_cost = 0
             for i in range(total_batch):
                  batch_x, batch_y =  X_train[i*batch_size:min(i*batch_size + batch_size, len(X_train)), :], y_train[i*batch_size:min(i*batch_size + batch_size, len(y_train)), :] 
                  _, c = sess.run([optimizer, mse], feed_dict = {x: batch_x, y: batch_y, keep_prob:.5}) 
                  avg_cost += c / total_batch
             if(avg_cost <= 300):
                 Vals.append(avg_cost)
#             print('Epoch:', (epoch+1), 'cost =', '{:.3f}'.format(avg_cost))
         plt.plot(Vals)
         plt.show()
         print(sess.run(mse, feed_dict = {x: X_test, y:y_test, keep_prob:1.0})) 
         cn = cn+1
text_file.close()
