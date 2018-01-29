import numpy as np
import os
from io import StringIO   # StringIO behaves like a file object
from numpy import genfromtxt

cwd = os.getcwd()
path1 = os.path.join(cwd, "../../Dataset/Test_features.csv")
path2 = os.path.join(cwd, "../../Dataset/Test_labels.csv")
path3 = os.path.join(cwd, "../../Dataset/Train_features.csv")
path4 = os.path.join(cwd, "../../Dataset/Train_labels.csv")

Train_features = genfromtxt(path3, delimiter=',')
Train_labels = genfromtxt(path4, delimiter=',')
Test_features = genfromtxt(path1, delimiter=',')
Test_labels = genfromtxt(path2, delimiter=',')


#print Train_features.shape
N=Train_features.shape[0]
#print N

def sigmoid(x):
  t=np.negative(x)
  y= 1/(1+np.exp(t))
  return y
#print Train_features[0]
y=sigmoid([-1,2,3])
print y
a=np.amax(Train_features)
#print a
Train_features=Train_features*(1/a)
F=np.mean(Train_features)
Train_features=Train_features-F
Test_features=Test_features*(1/a)
Test_features=Test_features-F
print np.mean(Train_features)
#print Test_features

'''length declarations '''
gamma=.01
eta=1e-5
k=4             #number of output neurons
l=10            #number of hidden neurons
x=96            #number of input neurons
w1=k*l          #number of weights for the W1 amtrix
w2=l*x          #number of weights for the W2 matrix
b1=k            #number of biases for the output layer
b2=l            #number of biases for the input layer
t=.1
'''random initialisation of weights '''
W1=np.random.rand(k,l)
W1=W1*t        # W1[0] has 96 elements, ie w1[0] is all the weights going from W0
W2=np.random.rand(l,x)
W2=W2*t
B1=(np.random.rand(1,k))*t
B2=(np.random.rand(1,l))*t

'''forward propagation'''

def Yval(X):
    L=sigmoid(np.dot(W2,X)+B2)
    #print L
    Y=sigmoid(np.dot(L,np.transpose(W1) )+B1)
    #print Y
    #print Y

    return Y

#   print Yval(W1,W2,Train_features[0])
def Lval(X):
    L=sigmoid(np.dot(X,np.transpose(W2))+B2)
    return L

def loss():
    y=0
    lossi=0
    for i in range(N):

        Y=Yval(Train_features[i])
        lossi=lossi+.5*(np.dot(Y-Train_labels[i],np.transpose(Y-Train_labels[i])))
        #print lossi
    return lossi

a=loss()
x= np.reshape(Lval(Train_features[4]),(1,10))
y= np.reshape(Yval(Train_features[4]),(1,4))
#print (Train_features.size)
'''Gradient Descent'''
def GradW1():
    Grad1=np.zeros((k,l))
    for i in range(N):
            #print (Yval(Train_features[i]).size,i)
            TempY=np.reshape(Yval(Train_features[i]),(1,4))
            TempL=np.reshape(Lval(Train_features[i]),(1,10))
            Tempy=np.reshape(Train_labels[i],(1,4))
            Grad1=Grad1-np.dot(np.transpose(np.dot(Tempy-TempY,np.dot(np.transpose(TempY),1-TempY))),TempL)+gamma*W1
            #np.dot((Tempy-TempY),np.dot(np.transpose(TempY),np.dot(1-TempY,np.transpose(TempL))))+gamma*W1
    return Grad1
G1=GradW1()

def GradB1():
    bias1=np.zeros((1,k))
    for i in range(N):
            TempY=np.reshape(Yval(Train_features[i]),(1,4))
            TempL=np.reshape(Lval(Train_features[i]),(1,10) )
            Tempy=np.reshape(Train_labels[i],(1,4))
            bias1=bias1- (np.dot(Tempy-TempY,np.dot(np.transpose(TempY),1-TempY)))
    return  bias1


def GradW2():
    Grad2=np.zeros((10,96))
    for i in range(N):
            TempY=np.reshape(Yval(Train_features[i]),(1,4))
            TempX=np.reshape(Train_features[i],(1,96))
            TempL=np.reshape(Lval(Train_features[i]),(1,10))
            Tempy=np.reshape(Train_labels[i],(1,4))
            Grad2=np.dot(np.transpose(np.dot(Tempy-TempY,np.dot(np.transpose(TempY),np.dot(1-TempY,np.dot(W1,np.dot(np.transpose(TempL),1-TempL)))))),TempX)+gamma*W2
            #np.dot(np.transpose(W1),np.dot(np.transpose(1-TempL),np.dot((TempL),np.dot(np.transpose(TempY),np.dot((1-TempY),np.dot(np.transpose(Tempy-TempY),np.transpose(TempX)))))))+gamma*W2
    return Grad2
G2=GradW2()


def GradB2():
    Grad2=np.zeros((1,l))
    for i in range(N):
            TempY=np.reshape(Yval(Train_features[i]),(1,4))
            TempX=np.reshape(Train_features[i],(1,96))
            TempL=np.reshape(Lval(Train_features[i]),(1,10))
            Tempy=np.reshape(Train_labels[i],(1,4))
            Grad2=(np.dot(Tempy-TempY,np.dot(np.transpose(TempY),np.dot(1-TempY,np.dot(W1,np.dot(np.transpose(TempL),1-TempL))))))

            #print Grad2.shape
    return Grad2





""" Training with gradient descent"""
epochs=100
for e in range(epochs):
    W1=W1-eta*GradW1()
    B1=B1-eta*GradB1()
    W2=W2-eta*GradW2()
    B2=B2-eta*GradB2()
    #print W1
    #print loss()

#def predictf():

def predictf():

    counter=0
    for i in range(80):
        a=Yval(Test_features[i])
        #print a
        b=Test_labels[i]
        at=np.argmax(a)
        bt=np.argmax(b)
        #print (at,bt)
        if (at==bt):
            counter=counter+1
    return counter
a=predictf()
print("Thee number  of correct predictions is ",a)
