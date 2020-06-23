import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
df = pd.read_csv("creditcard.csv")
df.describe()
df.head()
df.isnull().sum() 
columns = "Time V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 
V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 Amount".split()
X = pd.DataFrame.as_matrix(df,columns=columns)
Y = df.Class
Y = Y.reshape(Y.shape[0],1)
X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06)
X_test, X_dev, Y_test, Y_dev = train_test_split(X_test,Y_test, test_size=.5)
np.where(Y_train == 1)
np.where(Y_test == 1)
np.where(Y_dev == 1)
print("No of training Examples : "+str(X_train.shape[0])) # 94% data 
print("No of test Examples : "+str(X_test.shape[0])) # 3% data
print("No of dev Examples : "+str(X_dev.shape[0])) # 3% data
print("Shape of training data : "+str(X_train.shape))
print("Shape of test data : "+str(X_test.shape))
print("Shape of dev data : "+str(X_dev.shape))
print("Shape of Y test data : "+str(Y_test.shape))
print("Shape of Y dev data : "+str(Y_dev.shape))
X_train_flatten = X_train.reshape(X_train.shape[0],-1).T
Y_train_flatten = Y_train.reshape(Y_train.shape[0],-1).T
X_dev_flatten = X_dev.reshape(X_dev.shape[0],-1).T
Y_dev_flatten = Y_dev.reshape(Y_dev.shape[0],-1).T
X_test_flatten = X_test.reshape(X_test.shape[0],-1).T
Y_test_flatten = Y_test.reshape(Y_test.shape[0],-1).T
print("No of training Examples : "+str(X_train_flatten.shape)) 
print("No of test Examples : "+str(Y_train_flatten.shape)) 
print("No of X_dev Examples : "+str(X_dev_flatten.shape)) 
print("No of Y_dev test Examples : "+str(Y_dev_flatten.shape)) 
print("No of X_test Examples : "+str(X_test_flatten.shape)) 
print("No of Y_test Examples : "+str(Y_test_flatten.shape))
print("No of Sanity_test : "+str(X_train_flatten[0:5,0]))
X_train_set = preprocessing.normalize(X_train_flatten)
Y_train_set = Y_train_flatten
print("No of X_train_set shape : "+str(X_train_set.shape)) 
print("No of Y_train_set shape : "+str(Y_train_set.shape)) 
def intialize_parameters(layer_dims):
 parameters = {}
 L = len(layer_dims)
 for l in range(1,L):
 parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
 parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
return parameters
parameters = intialize_parameters([30,20,10,5,2])
print("W1 =" + str(parameters["W1"]))
print("b1 =" + str(parameters["b1"]))
print("W2 =" + str(parameters["W2"]))
print("b2 =" + str(parameters["b2"]))
print("W3 =" + str(parameters["W3"]))
print("b3 =" + str(parameters["b3"]))
print("W4 =" + str(parameters["W4"]))
print("b4 =" + str(parameters["b4"]))
def sigmoid(z):
 
 s = 1/(1+np.exp(-z))
 cache = z
 return s,cache
sigmoid(np.array(([2,7]))) 
def relu(z):
 r = np.maximum(0,z)
 cache = z
 return r,cache
relu([1,-1,21])
def relu_backward(dA, cache):
Z = cache
dZ = np.array(dA, copy=True) 
dZ[Z <= 0] = 0
assert (dZ.shape == Z.shape)
return dZ
def sigmoid_backward(dA, cache):
 Z = cache
 
 s = 1/(1+np.exp(-Z))
 dZ = dA * s * (1-s)
 assert (dZ.shape == Z.shape)
return dZ
def linear_forward(A, W, b):
 Z = np.dot(W,A)+b
 cache = (A, W, b)
return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
if activation == "sigmoid":
 Z, linear_cache = linear_forward(A_prev,W,b)
 A, activation_cache = sigmoid(Z)
elif activation == "relu":
 Z, linear_cache = linear_forward(A_prev,W,b)
 A, activation_cache = relu(Z)
 cache = (linear_cache, activation_cache)
 return A, cache
def forward_propagation(X, parameters):
 caches = []
 A = X
 L = len(parameters) // 2 
 for l in range(1, L):
A, cache = linear_activation_forward(A,parameters["W" + str(l)],parameters["b" 
+ str(l)],activation="relu")
 caches.append(cache)
AL, cache = linear_activation_forward(A,parameters["W" + str(L)],parameters["b" + 
str(L)],activation="sigmoid")
 caches.append(cache)
 return AL, caches
def cost_function(AL, Y):
 m = Y.shape[1]
cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)
cost = np.squeeze(cost) 
 return cost
def linear_backward(dZ, cache):
 A_prev, W, b = cache
 m = A_prev.shape[1]
 dW = (1/m)*np.dot(dZ,A_prev.T)
 db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
 dA_prev = np.dot(W.T,dZ)
 
 return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
linear_cache, activation_cache = cache
 if activation == "relu":
 dZ = relu_backward(dA,activation_cache)
 dA_prev, dW, db = linear_backward(dZ, linear_cache)
 
 elif activation == "sigmoid":
 dZ = sigmoid_backward(dA,activation_cache)
 dA_prev, dW, db = linear_backward(dZ, linear_cache)
 return dA_prev, dW, db
def backward_propagation(AL, Y, caches):
grads = {}
 L = len(caches) 
 Y = Y.reshape(AL.shape) 
 
 dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
 
 
 current_cache = caches[L-1]
 grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = 
linear_activation_backward(dAL,current_cache,activation="sigmoid")
for l in reversed(range(L-1)):
 current_cache = caches[l]
 dA_prev_temp, dW_temp, db_temp = 
linear_activation_backward(grads["dA"+str(l+2)],current_cache,activation="relu")
 grads["dA" + str(l + 1)] = dA_prev_temp
 grads["dW" + str(l + 1)] = dW_temp
 grads["db" + str(l + 1)] = db_temp
 return grads
def update_parameters(parameters, grads, learning_rate):
 L = len(parameters) // 2
 for l in range(1,L+1):
 parameters["W"+str(l)]=parameters["W" + str(l)]-learning_rate*grads["dW" + 
str(l)]
 parameters["b"+str(l)]=parameters["b" + str(l)]-learning_rate*grads["db" + str(l)]
 return parameters
layer_dims = [30,20,10,5,1] 
# Deep Learning network to classify frauds and normal
layer_dims = [30,20,10,5,1] 
# Deep Learning network to classify frauds and normal
def nn_model(X,Y,layer_dims,learning_rate=.0065, 
num_iterations=300,print_cost=False):
 costs = []
parameters = intialize_parameters(layer_dims)
 for i in range(0,num_iterations):
 
 AL, caches = forward_propagation(X, parameter
cost = cost_function(AL, Y)
 
 
 grads = backward_propagation(AL, Y, caches)
parameters = update_parameters(parameters,grads,learning_rate)
 
 if print_cost and i % 100 == 0:
 print ("Cost after iteration %i: %f" %(i, cost))
 if print_cost and i % 100 == 0:
 costs.append(cost)
plt.plot(np.squeeze(costs))
 plt.ylabel('cost')
 plt.xlabel('iterations (per tens)')
 plt.title("Learning rate =" + str(learning_rate))
 plt.show()
 
 return parameters
X_train_set.shape
Y_train_set.shape
parameters = 
nn_model(X_train_set,Y_train_set,layer_dims,learning_rate=.0065,num_iterations = 
300, print_cost = True)
def predict(X, y, parameters):
m = X.shape[1]
 p = np.zeros((1,m))
probas, caches = forward_propagation(X, parameters)
for i in range(0, probas.shape[1]):
 if probas[0,i] > 0.5:
 p[0,i] = 1
 else:
 p[0,i] = 0
print("Accuracy: " + str(np.sum((p == y)/m)))
 return p
pred_train = predict(X_train_set, Y_train_set, parameters)
pred_test = predict(X_test_flatten, Y_test_flatten, parameters)
pred_dev = predict(X_dev_flatten, Y_dev_flatten, parameters
