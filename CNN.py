# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:51:32 2022

@author: Hideto Kamei
"""
#npを使わないつもりでやった方が良い
import numpy as np
from numpy.random import *
import pandas as pd
import sklearn as sl
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch

"""
def softmax(x):
    ep = torch.exp(x)
    z = torch.sum(ep,dim=0)
    return z
"""

#inner product of tensor x and y, but regulated so that \infty*0=0
def mul_regulate(x,y):
    x[y==0]=0
    res = torch.mul(x,y)
    return res

def cross_entropy_error(y,t):
    delta = 1e-5
    print('y',y.shape)
    print('t',t.shape)
    print(torch.log(y+delta).shape)
    print(t.shape)
    err = -torch.sum(mul_regulate(torch.log(y+delta), t)) #tが0のときは0になるようにした。torch.mulの拡張した関数を定義
    batch_size = len(y)/y.shape[0] #軸の番号に注意
    return torch.sum(err) / batch_size

#test
t = torch.tensor([0,0,1,0,0,0,0,0,0,0],dtype=torch.float)
#y = torch.tensor([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0],dtype=torch.float)
y = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],dtype=torch.float)
cross_entropy_error(y,t)
#OK
    
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
        self.x[torch.isnan(self.x)]=0
        print('A',self.A.shape)
        print('B',self.B.shape)
        print('x',self.x.shape)
        #pytorchに直す
        self.y = torch.mm(self.A,x) +torch.einsum( 'i,j->ij',self.B, torch.ones(x.shape[1]) )#einsum is a replacement of np.outer
        return self.y
    
    def backward(self,py):
        self.px = torch.mm(torch.transpose(self.A,0,1),py)
        self.py = py
        #print(self.x)
        #import pdb; pdb.set_trace()
        #pytorchに直す
        self.pA = torch.mm(self.py, torch.transpose(self.x,0,1)) #ここがあやしい
        self.pB = torch.sum(self.py, dim=1) #要チェック
        self.pA[torch.isnan(self.pA)]=0
        self.pB[torch.isnan(self.pB)]=0
        return self.px

##test
aff1 = affine(torch.rand(2,3),torch.rand(2))
aff1.forward(torch.tensor([[1,2],[1,2],[1,2]],dtype=torch.float))
aff1.backward(torch.rand(2,2))

class relu:
    def __init__(self):
        self.x=None
        self.y=None
        self.py=None
        self.px=None

    def forward(self,x):
        self.x=x
        #print((x>0).to(torch.float32))
        self.y=torch.mul(self.x,(self.x>0).to(torch.float32)) #if x>0 return x, x<0 return 0
        return self.y

    def backward(self,py):
        self.py = py
        #print(py)
        #print(self.x>0)
        self.px = torch.mul(self.py, ((self.x>0).to(torch.float32)))
        return self.px

#test
relu1 = relu()
x = torch.rand(2,2,3)-0.5
relu1.forward(x)
relu1.backward(x+0.5)


class convolution:
    def __init__(self,A,B,Filtersize,p1h,p2h,p1w,p2w,S,H,W):
        #Filtersize is a list of length 2, consists of the height and the width of the filter
        #p1w, p2w are the padding size for width, p1h, p2h are the padding size for height
        #S is the Stride for the filter
        #(H,W) is the size of the original data(picture)
        self.A=A
        self.B=B
        self.Hf=Filtersize[0]
        self.Wf=Filtersize[1]
        self.x=None #keep it so that it is needed when u compute derivative
        self.y=None
        self.pA=None #keep it so that it is needed when u compute grad
        self.pB=None
        self.px=None
        self.py=None
        self.Hb = int((H+p1h+p2h-self.Hf)/S+1)
        self.Wb = int((W+p1w+p2w-self.Wf)/S+1)
        self.I = int(self.Hb*self.Wb)
        self.J = int(self.Hf*self.Wf)
        conversion = pd.DataFrame(np.zeros(shape=(self.I*self.J,8)))
        conversion.columns = ["i","j","hb","wb","hf","wf","h","w"]
        for i in range(0,self.I):
            for j in range(0,self.J):
                k = i*self.J+j
                conversion["i"][k] = i
                conversion["j"][k] = j
                (conversion["wb"][k], conversion["hb"][k]) = divmod(i,self.Hb)
                (conversion["wf"][k], conversion["hf"][k]) = divmod(j,self.Hf)
                conversion["h"][k] = conversion["hb"][k]*S+conversion["hf"][k]-p1h
                conversion["w"][k] = conversion["wb"][k]*S+conversion["wf"][k]-p1w
        self.conversion = np.array(conversion)
        
        conversion2 = pd.DataFrame(np.zeros(shape=(self.I,3)))
        conversion2.columns = ["i","hb","wb"]
        for i in range(0,self.I):
            conversion2["i"][i] = i
            (conversion2["wb"][i], conversion2["hb"][i]) = divmod(i,self.Hb)
        self.conversion2 = np.array(conversion2)

    def forward(self, x):
        self.x = x
        self.x[torch.isnan(self.x)]=0
        self.D = torch.zeros(size=(self.x.shape[0],self.x.shape[1],self.I,self.J),dtype=torch.float32)
        print(self.x.shape)
        for i in range(0,self.I):
            for j in range(0,self.J):
                k = i*self.J+j
                #値渡ししなくてOK?
                self.D[:,:,i,j] = torch.tensor(self.x[:,:,int(self.conversion[k,6]),int(self.conversion[k,7])])
        #Bd=np.zeros(shape=(D.shape[0],A.shape[2],self.D.shape[2]))
        print(self.A.shape)
        print(self.D.shape)
        self.E = torch.einsum('cfj,bcij->bfi',self.A,self.D)+torch.einsum('fi,b->bfi',self.B,torch.ones(self.x.shape[0])) #B and outer product of ones
        self.y = torch.zeros(size=(self.E.shape[0],self.E.shape[1],self.Hb,self.Wb))
        for i in range(0,self.I):
            self.y[:,:,int(self.conversion2[i,1]),int(self.conversion2[i,2])] = self.E[:,:,i]
        print("conv-output",y.shape)
        return self.y
    def backward(self,py):
        self.py = py #index:(b,f,hb,wb)
        M = torch.zeros(size=(self.py.shape[0],self.py.shape[1],self.Hb*self.Wb))
        #Mの中身を入れる
        for i in range(0,self.I):
            M[:,:,i] = py[:,:,int(self.conversion2[i,1]),int(self.conversion2[i,2])]
        AM = torch.einsum('cfj,bfi->bcij',self.A,M)
        self.px = torch.zeros(size=(self.x.shape))
        #print(self.px.shape)
        print("AM",AM.shape)
        print("I",self.I)
        print("J",self.J)
        for i in range(0,self.I):
            for j in range(0,self.J):
                k = i*self.J+j
                #print(k, self.px[:,:,int(self.conversion[k,6]),int(self.conversion[k,7])])
                #print(i,j,AM[:,:,i,j])
                #同じ(h,w)の値をとる(i,j)組が複数あるので、全て足し合わせる必要がある
                self.px[:,:,int(self.conversion[k,6]),int(self.conversion[k,7])]+=AM[:,:,i,j]
        self.pA = torch.einsum('bcij,bfi->cfj',self.D,M)
        print(py.shape)
        print(py.shape[0])
        self.pB = torch.einsum('bfi,b->fi',M,torch.ones(self.py.shape[0]))
        
        self.pA[torch.isnan(self.pA)]=0
        self.pB[torch.isnan(self.pB)]=0

        return self.px

##test
#N=2, C=2, H=5, W=5, p=0, S=1, F=3
#A.shape=(C,H*W,F)
A = torch.ones(2,3,4) #(c,f,j)
#B.shape=(N,Hb*Wb)
B = torch.zeros(3,16) #(f,i)
Filtersize=(2,2)
conv1 = convolution(A,B,Filtersize,0,0,0,0,1,5,5)
#x.shape=(N,C,H,W)
#x=torch.rand(2,2,5,5)
x=torch.rand(1,2,5,5) #(b,f,h,w)
conv1.forward(x)
#py.shape=(N,F,Hb,Wb)
py=torch.rand(1,3,4,4) #(b,c,hb,wb)
conv1.backward(py)

def lessint(x):
    if int(x)==x:
        return(int(x))
    else:
        return(int(x+1))

class pooling:
    def __init__(self,Hp,Wp):
        self.Hp=Hp
        self.Wp=Wp
        self.x=None
        self.y=None
        self.py=None
        self.px=None

    def forward(self,x):
        filter = torch.zeros(x.shape)
        self.x=x
        self.H=x.shape[2]
        self.W=x.shape[3]
        self.Hb = lessint(self.H/self.Hp)
        self.Wb = lessint(self.W/self.Wp)
        boxmax = torch.zeros(x.shape[0],x.shape[1],self.Hb,self.Wb)
        self.maxindex = torch.zeros(x.shape[0],x.shape[1],self.Hb,self.Wb,2) 
        for b in range(0,x.shape[0]): #no in batch
            for c in range(0,x.shape[1]): #channel
                for hb in range(0,self.Hb):
                    for wb in range(0,self.Wb):
                        box =x[b,c,int(hb*self.Hp):int((hb+1)*self.Hp),int(wb*self.Wp):int((wb+1)*self.Wp)]
                        boxmax[b,c,hb,wb] = torch.max(box)
                        #print( "res",np.array(np.where(box==boxmax[b,c,hb,wb]))[:,0] )
                        #print("shape",self.maxindex.shape)
                        tmp = np.array(np.where(box==boxmax[b,c,hb,wb]))[:,0] #maxの位置
                        self.maxindex[b,c,hb,wb,0] = int(tmp[0] + self.Hp*hb)
                        self.maxindex[b,c,hb,wb,1] = int(tmp[1] + self.Wp*wb)
        self.pybeforereshape = boxmax.shape
        boxmax = torch.reshape(boxmax, (boxmax.shape[0], -1))
        boxmax = torch.transpose(boxmax, 0, 1)
        print("box shape", boxmax.shape)
        self.y=boxmax
        print("boxmax",boxmax.shape)
        
        #最後にmatrixに変換
        
        return boxmax

    def backward(self,py):
        #print(py)
        py = torch.transpose(py, 0, 1)
        py = torch.reshape(py, (self.pybeforereshape))
        print("py shape",py.shape)
        self.py = py
        self.px = torch.zeros(self.x.shape)
        print("px shape",self.px.shape)
        #まずはmatrixからtensorに変換
        
        for b in range(0,self.py.shape[0]):
            for c in range(0,self.py.shape[1]):
                for hb in range(0,self.Hb):
                    for wb in range(0,self.Wb):
                        #print(int(self.maxindex[b,c,hb,wb,0]))
                        #print(int(self.maxindex[b,c,hb,wb,1]))
                        #print(self.py[b,c,hb,wb].shape)
                        #print(self.px[b,c,int(self.maxindex[b,c,hb,wb,0]),int(self.maxindex[b,c,hb,wb,1])].shape)
                        self.px[b,c,int(self.maxindex[b,c,hb,wb,0]),int(self.maxindex[b,c,hb,wb,1])] = self.py[b,c,hb,wb]
        return self.px

#test
pool1 = pooling(2,2)
x = torch.rand(2,2,6,6)
y = pool1.forward(x)
test = pool1.backward(y) #OK

class smloss:
    def __init__(self):
        self.x=None
        self.y=None
        self.t=None
        self.py=None
        self.px=None
        
    def forward(self,x):
        self.x = x
        #self.t = t
        #dimが0で合っているかは不明…。桁落ちを防ぐために平均を引く（Zで割っているので問題ない）
        self.x = self.x-torch.mean(self.x,dim=0)
        ep = torch.exp(-self.x)
        Z = torch.sum(ep,dim=0)
        self.y = ep/Z
        #self.loss = cross_entropy_error(self.y, self.t)
        return self.y

    def backward(self,t):
        self.t = t
        batch_size= self.t.shape[0]
        self.px = -(self.y-t)/batch_size
        return self.px

class simpleconvnet:
    def __init__(self, channel_no=1, height=28, width=28, \
                 conv_param={'filter_no':30,'filter_size':[5,5],'filter_pad':[0,0,0,0],'filter_stride':1}, \
                 pooling_size=(2,2), hidden_size=100, output_size=10, weight_init_std=0.001):
        self.channel_no=channel_no
        self.height=height
        self.width=width
        self.filter_no=conv_param['filter_no']
        self.filter_size=conv_param['filter_size']
        self.filter_pad=conv_param['filter_pad']
        self.filter_stride=conv_param['filter_stride']
        self.pooling_size=pooling_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.weight_init_std=weight_init_std
        
        self.conv_output_h=(self.height-self.filter_size[0]+self.filter_pad[0]+self.filter_pad[1])/self.filter_stride+1
        self.conv_output_w=(self.width-self.filter_size[1]+self.filter_pad[2]+self.filter_pad[3])/self.filter_stride+1
        self.pool_output_size=int(self.filter_no*lessint(self.conv_output_h/self.pooling_size[0])*lessint(self.conv_output_w/self.pooling_size[1]))
        self.output_size = output_size
        
        self.params ={}
        self.params['AC']=weight_init_std*torch.rand(int(self.channel_no),int(self.filter_no),int(self.filter_size[0]*self.filter_size[1]))
        self.params['BC']=torch.zeros(int(self.filter_no),int(self.conv_output_h*self.conv_output_w))
        self.params['A1']=weight_init_std*torch.rand(self.hidden_size, self.pool_output_size)
        self.params['B1']=torch.zeros(self.hidden_size)
        self.params['A2']=weight_init_std*torch.rand(self.output_size,self.hidden_size)
        self.params['B2']=torch.zeros(self.output_size)
        
        self.layers = OrderedDict()
        self.layers['conv1'] = convolution(self.params['AC'],self.params['BC'],self.filter_size,\
                                 self.filter_pad[0],self.filter_pad[1],self.filter_pad[2],self.filter_pad[3],\
                                 conv_param['filter_stride'],self.height, self.width)
        self.layers['relu1'] = relu()       
        #self.layers['pool1'] = pooling(self.conv_output_h,self.conv_output_w) #check!
        self.layers['pool1'] = pooling(pooling_size[0],pooling_size[1])
        self.layers['affine1'] = affine(self.params['A1'],self.params['B1'])
        self.layers['relu2'] = relu()
        self.layers['affine2'] = affine(self.params['A2'],self.params['B2'])
        self.lslayer = smloss()
        self.t = None

    def predict(self,xinit):
        xout = xinit
        for ln in self.layers:
            print(ln)
            ly = self.layers[ln]
            xout = ly.forward(xout)
            print("xout",xout.shape)
        xout = self.lslayer.forward(xout)
        #self.lslayer.y = xout
        #print('xout',xout)
        return xout

    def loss(self, x,t): #x:input t:prediction
        y= self.predict(x)
        res = cross_entropy_error(y,t)
        return res

    def backward(self,t):
        py = self.lslayer.backward(tbat)
        py[torch.isnan(py)]=0
        for ln in reversed(self.layers):
            ly = self.layers[ln]
            print(ln)
            py = ly.backward(py)
        self.grads = {}
        self.grads['AC'] = self.layers['conv1'].pA #self.conv1.pA
        self.grads['BC'] = self.layers['conv1'].pB #self.conv1.pB
        self.grads['A1'] = self.layers['affine1'].pA #self.affine1.pA
        self.grads['B1'] = self.layers['affine1'].pB #self.affine1.pB
        self.grads['A2'] = self.layers['affine2'].pA #self.affine2.pA
        self.grads['B2'] = self.layers['affine2'].pB #self.affine2.pB


def difA(snet,xbat,tbat,delta,layer):
    Ashape = snet.layers[layer].A.shape
    pA = np.zeros((3,3))
    snet.layers[layer] =snet.layers[layer]
    #for i in range(0,Ashape[0]):
    for i in range(0,3):
        #for j in range(0,Ashape[1]):
        for j in range(0,3):
            print(i,j)
            dA = torch.zeros(Ashape)
            dA[i,j] = delta
            ce_base = snet.loss(xbat,tbat)
            snet.layers[layer].A = snet.layers[layer].A + dA
            ce_delta = snet.loss(xbat,tbat)
            snet.layers[layer].A = snet.layers[layer].A - dA
            pA[i,j] = (ce_delta-ce_base)/delta  
    return pA

def difB(snet,xbat,tbat,delta,layer):
    Bshape = snet.layers[layer].B.shape
    pB = np.zeros(3)
    #for i in range(0,Bshape[0]):
    for i in range(0,3):
            print(i)
            dB = torch.zeros(Bshape)
            dB[i] = delta
            ce_base = snet.loss(xbat,tbat)
            snet.layers[layer].B = snet.layers[layer].B + dB
            ce_delta = snet.loss(xbat,tbat)
            snet.layers[layer].B = snet.layers[layer].B - dB
            pB[i] = (ce_delta-ce_base)/delta  
    return pB



#construct convolutional neural network
snet = simpleconvnet(channel_no=1, height=28, width=28, \
                 conv_param={'filter_no':30,'filter_size':(5,5),'filter_pad':(0,0,0,0),'filter_stride':1},\
                 pooling_size=(2,2), hidden_size=100, output_size=10, weight_init_std=0.01)

#read mnist
train = pd.DataFrame(pd.read_csv('C:/Users/Hideto Kamei/Documents/04_Programming/03_Research/NN/mnist/train.csv'))

tvec = train['label']
train_orig = train.iloc[:,1:]
train_orig = np.array(train.values)
#convert train data (matrix) to tensor
train = torch.zeros(train_orig.shape[0],1,snet.height,snet.width)
for i in range(0,snet.height):
    for j in range(0,snet.width):
        k=i+j*snet.height
        train[:,0,i,j] = torch.tensor(train_orig[:,k])

#convert label to one hot shot
t = torch.zeros( (len(tvec),len(np.unique(tvec))) )
for i in range(0,len(t)):
    t[i,tvec[i]] = 1


iter = int(train.shape[0]/batch_size)
train_acc_list = []

cel = []

for bc in range(0,iter):

    #learning
    batch_size = 10
    #learning_rate = 1/(bc+1)**0.5
    learning_rate = 1/(bc+1)

    print("bc",bc)
    xbat = train[(batch_size*bc):min(batch_size*(bc+1),train.shape[0]),:,:,:]
    tbat = t[(batch_size*bc):min(batch_size*(bc+1),train.shape[0])]
    tbat = torch.transpose(tbat, 0, 1)
    #forward probagation
    pred = snet.predict(xbat)
    lossres = cross_entropy_error(pred, tbat)
    
    #backward propagation
    snet.backward(tbat)
    
    print('A2', snet.grads['A2'])
    print('B2', snet.grads['B2'])
    #print('difA1',difA(snet,xbat,tbat,delta=0.0001,layer='affine1'))
    print('A1', snet.grads['A1'])
    #print('difB1',difB(snet,xbat,tbat,delta=0.0001,layer='affine1'))
    print('B1', snet.grads['B1'])
    print('AC', snet.grads['AC'])
    print('BC', snet.grads['BC'])

    print('affine2 x', snet.layers['affine2'].x)
    print('affine1 x', snet.layers['affine1'].x)
    print('conv1 x', snet.layers['conv1'].x)


    
    #for key in ('AC','BC','A2','B2','A1','B1'):
    for key in ('A1','B1'):
        #print(key, snet.grads[key])
        snet.params[key] = snet.params[key] - snet.grads[key] * learning_rate
    #
    snet.layers['conv1'].A = snet.params['AC']
    snet.layers['conv1'].B = snet.params['BC']
    snet.layers['affine2'].A = snet.params['A2']
    snet.layers['affine2'].B = snet.params['B2']
    snet.layers['affine1'].A = snet.params['A1']
    snet.layers['affine1'].B = snet.params['B1']
    
    print("loss",lossres)
    cel.append(lossres)

#plt.plot(cel)
