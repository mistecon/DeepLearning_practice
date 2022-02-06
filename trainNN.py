import numpy as np
from numpy.random import *
import pandas as pd
import sklearn as sl
from collections import OrderedDict
import matplotlib.pyplot as plt

#read titanic
train = pd.DataFrame(pd.read_csv('C:/Users/Hideto Kamei/Documents/04_Programming/03_Research/NN/titanic/train.csv'))
train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
for it in ['Sex','Embarked']:
    lst = pd.unique(train[it])
    for i, ls in enumerate(lst):
        print(i)
        print(ls)
        train[it] = train[it].replace(ls,i)

tvec = train['Survived']
train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
ageav = np.average(train.loc[np.isnan(train['Age'])==False,'Age'])
train.loc[np.isnan(train['Age']),'Age'] = ageav
#plt.hist(train.ix[tvec==0,4])
#plt.hist(train.ix[tvec==1,4])


t = np.zeros( (len(tvec),len(np.unique(tvec))) ) #one hot shot?表現に変える
for i in range(0,len(t)):
    t[i,tvec[i]] = 1

def ce(xbat,A,B,tbat):
    y = np.exp(-(A.dot(xbat)+np.outer(B,np.ones(xbat.shape[1]))))
    p = y/sum(y)
    ce = sum( sum(-tbat*np.log(p)) )/xbat.shape[1]
    return ce
    
delta = 1/10000

def difA(xbat,A,B,tbat,delta):
    pA = np.zeros(A.shape)
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            #print(i,j)
            dA = np.zeros(A.shape)
            dA[i,j] = delta
            pA[i,j] = (ce(xbat,A+dA,B,tbat)-ce(xbat,A,B,tbat))/delta  
    return pA


def difB(xbat,A,B,tbat,delta):
    pB = np.zeros(B.shape)
    for i in range(0,len(B)):
            #print(i)
            dB = np.zeros(B.shape)
            dB[i] = delta
            pB[i] = (ce(xbat,A,B+dB,tbat)-ce(xbat,A,B,tbat))/delta  
    return pB

def cross_entropy_error(p,t):
    delta = 1e-7
    print('p',p.shape)
    print('t',t.shape)
    err = sum( sum(-t*np.log(p)) )/t.shape[1]
    return err


class affine:   #y=Ax+B, px, pA, pB = partial derivative of y wrt components of x,A,B
    def __init__(self,A,B):
        self.A=A
        self.B=B
        self.x=None #keep it so that it is needed when u compute derivative
        self.y=None
        self.pA=None #keep it so that it is needed when u compute grad
        self.pB=None 
        self.px=None 
        self.py=None 

    def forward(self,x):
        self.x = x
        self.x[np.isnan(self.x)]=0        
        print('A',self.A.shape)
        print('B',self.B.shape)
        print('x',self.x.shape)
        self.y = self.A.dot(x) +np.outer(self.B , np.ones(x.shape[1]) )#needs to be checked
        return self.y
    
    def backward(self,py):
        self.px = np.dot(self.A.T,py)
        self.py = py
        #print(self.x)
        #import pdb; pdb.set_trace()
        self.pA = self.py.dot(self.x.T) #ここがあやしい
        self.pB = np.sum(self.py, axis=1)
        self.pA[np.isnan(self.pA)]=0
        self.pB[np.isnan(self.pB)]=0
        return self.px
    
class smloss:
    def __init__(self):
        self.x=None
        self.y=None
        self.t=None
        self.py=None
        self.px=None
        
    def forward(self,x):
        #self.t = t
        ep = np.exp(-x)
        Z = np.sum(ep,axis=0)
        self.y = ep/Z
        #self.loss = cross_entropy_error(self.y, self.t)
        return self.y

    def backward(self,t):
        self.t = t
        batch_size= self.t.shape[1]
        self.px = -(self.y-t)/batch_size
        return self.px

class simplenet:
    def __init__(self,N1,N2):
       self.params = {}
       self.params['A'] = rand(N2,N1)
       #self.params['B1'] = rand(N2,1)
       self.params['B'] = rand(N2)

       self.affine1 = affine(self.params['A'],self.params['B'])
       self.layer = OrderedDict([('affine1',self.affine1)])
       self.lslayer = smloss()
       self.t = None

    def predict(self,xinit):
        xout = xinit
        for ln in self.layer:
            ly = self.layer[ln]
            xout = ly.forward(xout)
        xout = self.lslayer.forward(xout)
        self.lslayer.y = xout
        print('xout',xout)
        return xout

    def loss(self, x,t): #x:input t:prediction
        y= self.predict(x)
        res = cross_entropy_error(y,t)
        return res

    def backward(self,t):
        print('t',t)
        print('y',self.lslayer.y)
        #py = t-self.lslayer.y
        py = self.lslayer.backward(tbat)
        py[np.isnan(py)]=0 
        py = self.affine1.backward(py)
        self.grads = {}
        print(self.layer['affine1'].pA)
        self.grads['A'] = self.affine1.pA
        self.grads['B'] = self.affine1.pB

N1= 7
N2= 2
snet = simplenet(N1,N2)

batch_size = 20
learning_rate = 0.001

iter = int(train.shape[0]/batch_size)
train_acc_list = []

A = rand(2,7)
B = rand(2)
snet.params['A'] = A
snet.params['B'] = B
snet.affine1.A = A
snet.affine1.B = B
cel = []

for bc in range(0,iter):
    xbat = train.T.iloc[: , (batch_size*bc):min(batch_size*(bc+1),train.shape[0])]
    tbat = t.T[: ,(batch_size*bc):min(batch_size*(bc+1),train.shape[0])]
    pred = snet.predict(xbat)
    lossres = cross_entropy_error(pred, tbat)
    snet.backward(tbat)
    for key in ('A','B'):
       snet.params[key] = snet.params[key] - snet.grads[key] * learning_rate
    #
    snet.affine1.A = snet.params['A']
    snet.affine1.B = snet.params['B']
    pA = difA(xbat,A,B,tbat,delta)
    A = A - pA * 0.001
    pB = difB(xbat,A,B,tbat,delta)
    B = B - pB * 0.001
    #cel.append(ce(xbat,A,B,tbat))
    #
    print(lossres)
    print(ce(xbat,A,B,tbat))
    cel.append(lossres)

plt.plot(cel)
