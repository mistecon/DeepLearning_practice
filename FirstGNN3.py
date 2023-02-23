# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:38:48 2023

@author: kamei
"""

import pandas as pd
import torch
import math
import random
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

#1.1 adjacency matrix generation
C = 5
N = 50
p_intra = 0.6
p_inter = 0.2

#number of communities
#N/Cは割り切れる必要がある
n_com = math.floor(N/C)
adj = torch.zeros(N,N)

for i in range(0,C):
    for j in range(0,C):
        if i == j:
            p_edge = p_intra
        else:
            p_edge = p_inter
        edge_part = (torch.rand(n_com, n_com)<=p_edge)*1
        adj[(i*n_com):((i+1)*n_com),(j*n_com):((j+1)*n_com)] = edge_part
        adj = torch.triu(adj, diagonal=1)+torch.transpose(torch.triu(adj, diagonal=1),0,1)
        
#1.2 
#number of sources
M = 10
#range of signal
a = 0
b = 10
#total period
T = 4
#deriving laplacian matrix
L = adj.to(torch.float64)
adjsum = torch.sum(adj,dim=0)
for i in range(0,L.shape[0]):
    L[i,i] = -adjsum[i]
L = -L

sqrt_Dinv = torch.tensor(np.diag(L.diagonal()**(-0.5)))

Lbar = np.matmul( np.matmul(sqrt_Dinv, L), sqrt_Dinv)

S = Lbar

Rec = 2100

#maximum power of S when we construct the model is K-1
K=8

#list of source node(we will not change this)
source = random.sample(range(0,N),M)

#generate Rec numbers of history
X = torch.zeros(Rec,N).to(torch.float64)
Y = torch.zeros(Rec,N).to(torch.float64)

for rec in range(0,Rec):
    signal = torch.tensor(np.random.uniform(low=a, high=b, size=M))
    z = torch.zeros(N).to(torch.float64)
    z[source] = signal
    #output for training data
    Y[rec,:] = z
    for t in range(0,T):
        w = torch.tensor(np.random.normal(0,1,N))
        z = torch.matmul(S,z)+w
    #input for training data
    X[rec,:] = z

#Graph filter function
def GFilter_func(S, x, H):
    filter_res = 0
    #h is a vector of length K
    for k in range(0,len(H)):
        #filter_res += torch.matmul(x, torch.matrix_power(S, k)) * h[k]
        print(x.shape, x.dtype)
        print(S.shape, S.dtype)
        print(H[k,:,:].shape, H[k,:,:].dtype)
        filter_res += torch.einsum('ij,bjp,pq->biq', torch.matrix_power(S, k), x, H[k,:,:])
    return filter_res


class GFilter(nn.Module):
    """ Custom layer for Graph filter """
    def __init__(self, K, S, Fin, Fout):
        super().__init__()
        self.K = K
        self.S = S
        #h = torch.zeros(K).to(torch.float64)
        H = torch.randn(K, Fin, Fout).to(torch.float64)
        self.H = nn.Parameter(H)
        
    def forward(self, x):
        GFilter_res = GFilter_func(self.S, x, self.H)
        return GFilter_res

class GNN(nn.Module):
    
    def __init__(self, Klist, S, F):
        super(GNN, self).__init__()
        self.gf1 = GFilter(Klist[0], S, F[0], F[1])
        self.gf2 = GFilter(Klist[1], S, F[1], F[2])
        self.gf3 = GFilter(Klist[2], S, F[2], F[3])

    def forward(self, x):
        x = x.reshape(x.shape[0],x.shape[1],1)
        x = self.gf1(x)
        x = F.relu(x)
        x = self.gf2(x)
        x = F.relu(x)
        x = self.gf3(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0],x.shape[1])
        return x

#parameters for train
E=30
Batchsize = 200
learning_rate = 0.05
BatchNo = math.ceil(X.shape[0]/Batchsize)

#gfilter = GNN([8,1], S, [1,32,1])
gfilter = GNN([5,5,1], S, [1,16,4,1])

losslist = []
for epoch in range(0,E):
    perm = list(range(0,len(X)))
    random.shuffle(perm)
    X_perm = X[perm]
    Y_perm = Y[perm]
    for batch in range(0, BatchNo):
        X_batch = X_perm[(batch*Batchsize):((batch+1)*Batchsize),:]
        Y_batch = Y_perm[(batch*Batchsize):((batch+1)*Batchsize),:]
        
        output = gfilter(X_batch)
        optimizer = optim.Adam(gfilter.parameters(), lr=learning_rate/np.sqrt(epoch+10))
        criterion = nn.MSELoss()
        loss = criterion(output, Y_batch)
        losslist.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
plt.yscale('log')
plt.plot(pd.Series(losslist).rolling(10).mean())
