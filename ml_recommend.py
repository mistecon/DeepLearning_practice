# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 23:46:45 2023

@author: kamei
"""
import pandas as pd
import sklearn.model_selection
import numpy as np
from numpy import linalg as LA
import math
import random
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#1.1 Loading the data
ml_list = pd.read_csv("C:/Users/kiuch/Documents/99_other/GNN/lab3/u.data", sep='\t', header=None )
ml_list.columns = ["user_id", "movie_id", "rating", "timestamp"]
user_ids = sorted(ml_list["user_id"].unique())
movie_ids = sorted(ml_list["movie_id"].unique())

#convert to matrix form
ml_matrix = pd.DataFrame(0,index=user_ids,columns=movie_ids)
print(ml_matrix)
print(ml_matrix.shape)

for i in range(0,ml_list.shape[0]):
    ml_matrix.loc[ml_list.loc[i,"user_id"],ml_list.loc[i,"movie_id"]] = ml_list.loc[i,"rating"]

print(ml_matrix)

#1.2 Data Clean-up
cond = (ml_matrix.sum(axis=0)>=150)
ml_matrix = ml_matrix.loc[:,cond]
ml_matrix = ml_matrix.rename(columns={257: "Contact"})
contact_loc = list(ml_matrix.columns).index('Contact')
ml_matrix.shape

def inner_prod_func(x1,x2):
    cond = ((x1>0)&(x2>0))*1
    mu1 = (1/len(cond))*sum(cond*x1)
    mu2 = (1/len(cond))*sum(cond*x2)
    inner_prod = (1/len(cond))*sum(cond*(x1-mu1)*(x2-mu2))
    return inner_prod

def corr_movie_func(x1, x2):
    corr_movie = inner_prod_func(x1,x2)/(np.sqrt(inner_prod_func(x1,x1)*inner_prod_func(x2,x2)))
    return corr_movie

def adj_matrix_func(ml_train):
    """
    adj_matrix = ml_train.corr()
    np.fill_diagonal(adj_matrix.values, 0)
    """
    adj_matrix = pd.DataFrame(0, index=ml_train.columns,columns=ml_train.columns)
    for i in range(0,len(ml_train.columns)):
        print(i)
        for j in range(i+1,len(ml_train.columns)):
            adj_matrix.iloc[i,j] = corr_movie_func(ml_train.iloc[:,i],ml_train.iloc[:,j])
    adj_matrix = adj_matrix + adj_matrix.T
        
    #1.4 for each row, keep 40 largest entries
    """
    for i in range(0,adj_matrix.shape[0]):
        mask = (abs(adj_matrix.iloc[i,:])>sorted(abs(adj_matrix.iloc[0,:]))[-40])*1
        adj_matrix.iloc[i,:] = adj_matrix.iloc[i,:]*mask
    
    """
    eigenval, eigenvec  = LA.eig(adj_matrix)
    adj_matrix = adj_matrix/max(abs(eigenval))
    adj_matrix = torch.tensor(np.array(adj_matrix)).to(torch.float64)
    return adj_matrix

def normalized_Lap_func(adj_matrix):
    L = adj_matrix.to(torch.float64)
    adjsum = torch.sum(adj_matrix,dim=0)
    for i in range(0,L.shape[0]):
        L[i,i] = -adjsum[i]
    L = -L
    #sqrt_Dinv = torch.tensor(np.diag(L.diagonal()**(-0.5)))
    #Lbar = np.matmul( np.matmul(sqrt_Dinv, L), sqrt_Dinv)
    return L

def create_graph(ml_train):
    
    # Everything below 1e-9 is considered zero
    zeroTolerance = 1e-9
    
    # Number of nodes is equal to the number of columns (movies)
    N = ml_train.shape[1]
    
    # Isolating users used for training
    XTrain = np.transpose(ml_train)
    
    # Calculating correlation matrix
    binaryTemplate = (XTrain > 0).astype(XTrain.dtype)
    sumMatrix = XTrain.dot(binaryTemplate.T)
    countMatrix = binaryTemplate.dot(binaryTemplate.T)
    countMatrix[countMatrix == 0] = 1
    avgMatrix = sumMatrix / countMatrix
    sqSumMatrix = XTrain.dot(XTrain.T)
    correlationMatrix = sqSumMatrix / countMatrix - avgMatrix * avgMatrix.T 
    
    # Normalizing by diagonal weights
    sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
    nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance)\
                                                 .astype(sqrtDiagonal.dtype)
    sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1.
    invSqrtDiagonal = 1/sqrtDiagonal
    invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
    normalizationMatrix = np.diag(invSqrtDiagonal)
    
    # Zero-ing the diagonal
    normalizedMatrix = normalizationMatrix.dot(
                            correlationMatrix.dot(normalizationMatrix)) \
                            - np.eye(correlationMatrix.shape[0])

    # Keeping only edges with weights above the zero tolerance
    normalizedMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0.
    W = normalizedMatrix
    """
    # Sparsifying the graph
    WSorted = np.sort(W,axis=1)
    threshold = WSorted[:,-knn].squeeze()
    thresholdMatrix = (np.tile(threshold,(N,1))).transpose()
    W[W<thresholdMatrix] = 0
    """
    # Normalizing by eigenvalue with largest magnitude
    E, V = np.linalg.eig(W)
    W = W/np.max(np.abs(E))
    
    return W


#1.3 Movie Similarity Graph
#1.5 Training data
#1.6 Test data
def preprocess(ml_train, ml_test):
    ml_train = ml_train[ml_train["Contact"]>0]
    ml_test = ml_test[ml_test["Contact"]>0]
    ml_train_Y = ml_train["Contact"]
    ml_test_Y = ml_test["Contact"]
    ml_train_X = ml_train
    ml_test_X = ml_test
    ml_train_X["Contact"] = 0
    ml_test_X["Contact"] = 0
    ml_train_X = torch.tensor(np.array(ml_train_X)).to(torch.float64)
    ml_train_Y = torch.tensor(np.array(ml_train_Y)).to(torch.float64)
    ml_test_X = torch.tensor(np.array(ml_test_X)).to(torch.float64)
    ml_test_Y = torch.tensor(np.array(ml_test_Y)).to(torch.float64)
    return (ml_train_X, ml_train_Y, ml_test_X, ml_test_Y)

#Graph filter function
def GFilter_func(x, S, H):
    filter_res = 0
    #h is a vector of length K
    for k in range(0,len(H)):
        #print(x.shape, x.dtype)
        #print(S.shape, S.dtype)
        #print(H[k,:,:].shape, H[k,:,:].dtype)
        filter_res += torch.einsum('ij,bjp,pq->biq', torch.matrix_power(S, k), x, H[k,:,:])
    return filter_res

class GFilter(nn.Module):
    """ Custom layer for Graph filter """
    def __init__(self, K, Fin, Fout):
        super().__init__()
        self.K = K
        #h = torch.zeros(K).to(torch.float64)
        H = torch.randn(K, Fin, Fout).to(torch.float64)*0.1
        self.H = nn.Parameter(H)
        
    def forward(self, x, S):
        GFilter_res = GFilter_func(x, S, self.H)
        return GFilter_res

class GNN(nn.Module):
    
    def __init__(self, Klist, F):
        super(GNN, self).__init__()
        self.gf1 = GFilter(Klist[0], F[0], F[1])
        self.gf2 = GFilter(Klist[1], F[1], F[2])
        self.gf3 = GFilter(Klist[2], F[2], F[3])

    def forward(self, x, S):
        x = x.reshape(x.shape[0],x.shape[1],1)
        x = self.gf1(x, S)
        x = F.relu(x)
        x = self.gf2(x, S)
        x = F.relu(x)
        x = self.gf3(x, S)
        #x = 6*torch.sigmoid(x)
        return x

#parameters for train
E=500
Batchsize = 5
learning_rate = 0.5

gfilter = GNN([5,5,1], [1,64,32,1])

losslist = []
outputlist = [] 
for epoch in range(0,E):
    ml_train, ml_test =  sklearn.model_selection.train_test_split(ml_matrix, train_size=0.9, random_state=epoch, shuffle=True)
    #adj_matrix = adj_matrix_func(ml_train)
    adj_matrix = torch.tensor(create_graph(np.array(ml_train)))
    S = normalized_Lap_func(adj_matrix)
    ml_train_X, ml_train_Y, ml_test_X, ml_test_Y = preprocess(ml_train, ml_test)
    BatchNo = math.ceil(ml_test_X.shape[0]/Batchsize)

    for batch in range(0, BatchNo):
        X_batch = ml_test_X[(batch*Batchsize):((batch+1)*Batchsize),:]
        Y_batch = ml_test_Y[(batch*Batchsize):((batch+1)*Batchsize)]
        
        output = gfilter(X_batch, S)[:,contact_loc].squeeze()
        optimizer = optim.Adam(gfilter.parameters(), lr=learning_rate/(1+epoch))
        criterion = nn.MSELoss()
        loss = criterion(output, Y_batch)
        print(output, Y_batch)
        print(loss)
        losslist.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
plt.yscale('log')
plt.plot(pd.Series(losslist).rolling(10).mean())
