import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

'''
Object Multilayer_perceptron
'''

def multilayer_perceptron(x,weights,biases):
   layer_1 = tf.add(tf.matmul(x,weights['W1']), biases['b1'])
   layer_1= tf.nn.relu(layer_1)
#   layer_1_do = tf.nn.dropout(layer_1, keep_prob)
        
   out_layer = tf.matmul(layer_1,weights['out']) + biases['out']
   return out_layer
    



'''
Import Data from Data.csv:
    Ch:cypher text
    Plain: plain text
'''


Xdata = []
Ydata = []
for i in range(100):
    Ydata.append([])
with open('data.csv') as csvfile:
    fieldNames = ['ch','steps','plain']
    read = csv.DictReader(csvfile, fieldnames = fieldNames)
    
    for row in read:
        Xaux = row['ch']
        Yaux = row['plain']
        aux = []
        for i in range(len(Xaux)):
            aux.append(int(Xaux[i]))
            if(Yaux[i] == '0'):
                Ydata[i].append([0,1])
            else:
                Ydata[i].append([1,0])
        Xdata.append(aux)
Xdata = np.array(Xdata)
Ydata = np.array(Ydata)

cn = 1




'''
Iteration for each bit
'''

for i in range (100):
    '''
    Prepare the data.
    '''

    X = Xdata
    y = Ydata[i]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size = 0.33)
    

    '''
    Set learning_rate, batch_size and number of iterations (epochs)
    '''
    
    learning_rate = 0.28
    epochs = 100
    batch_size = 10
    
    
    '''
    Define placeholders:
        X: Input
        Y: Output
        Keep_prob: probability using in dropout
    '''
    x = tf.placeholder(tf.float32, [None, 100], name = 'x')
    y = tf.placeholder(tf.float32, [None, 2], name = 'y')
    keep_prob = tf.placeholder("float")
    
    
    
    
    '''
    Define weghts and biases for each layer
    '''
    
    weights = {
            'W1': tf.Variable(tf.random_normal([100, 50])),
            'out': tf.Variable(tf.random_normal([50, 2]))
            }
    
    biases = {
            'b1': tf.Variable(tf.random_normal([50])),
            'out': tf.Variable(tf.random_normal([2]))
            }
     

    '''
    Loss function and prediction
    '''
    y_ = multilayer_perceptron(x,weights,biases)
    sfmx = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y))
        
        
    '''
    Optimizer (Gradient Descent)
    '''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(sfmx)  
        
        
  
    '''
    Initialize variables, define prediction and give accuracy
    '''
    init_op = tf.global_variables_initializer()
    prediction = tf.nn.softmax(y_)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pr = tf.argmax(tf.nn.softmax(y_),1)




    '''
    Train
    '''
    with tf.Session() as sess:
        
        
         sess.run(init_op)
         total_batch = int(len(y_train) / batch_size)  
         Vals = []
         print(cn)
         
         for epoch in range(epochs):
             avg_cost = 0
             for i in range(total_batch):
                  batch_x, batch_y =  X_train[i*batch_size:min(i*batch_size + batch_size, len(X_train)), :], y_train[i*batch_size:min(i*batch_size + batch_size, len(y_train)), :] 
                  _, c = sess.run([optimizer, sfmx], feed_dict = {x: batch_x, y: batch_y, keep_prob:.5}) 
                  avg_cost += c / total_batch
                  
             if(avg_cost <= 300):
                 Vals.append(avg_cost)
             print('Epoch:', (epoch+1), 'cost =', avg_cost)
         plt.title("Learning Curve")
         plt.xlabel("epoch")
         plt.ylabel("mean of cross entropy loss")
         plt.plot(Vals)
         plt.show()
         print('Accuracy:', sess.run(accuracy,feed_dict = {x: X_test, y:y_test, keep_prob:1.0} ))
         cn = cn+1
