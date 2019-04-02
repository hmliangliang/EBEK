# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\4\1 0001 20:46:46
# File:         LKPE.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math

def LKPE(data, m):
    '''
	Elbagoury A, Ibrahim R, Kamel M S, et al. EBEK: Exemplar-Based Kernel Preserving Embedding[C]//IJCAI. 2016: 1441-1447.主要完成的是降维操作
    data:输入的数据，每一行代表一个样本,每一列代表一个特征 n*d
    m: 降维后数据的维数
    '''
    epsilon = 0.64 #原文算法2中epsilon参数
    n = data.shape[0] #数据的样本数
    d = data.shape[1] #数据的维数
    A = data.transpose() #将输入的数据进行转置,与文中的参数的形式保持一致
    U, Sigma, V = np.linalg.svd(A) #对A进行奇异值分解
    if len(Sigma)>=m:
        Sigma = np.diag(Sigma[0:m])
    else:
        Sigma = np.diag(Sigma,m)
    #选择出m个独立的样本
    E = [0] #用于记录选择出来的列
    size = 0
    for i in range(1,min(m,n)):
        ai = A[:,[i]]
        for j in range(0,size):
            aj = A[:,[E[j]]]
            ai = ai - sum(ai*aj)/sum((aj*aj)*aj)
        if np.linalg.norm(ai,1)!=0:
            for ii in range(len(E)):
                if np.dot(A[:,ii],A[:,[i]]) <= epsilon:
                    size = size + 1
                    E.append(i)
                    break
    if len(E)<m:#防止epsilon取值不合理,出现选择的样本数少于m个
        condicate = []
        for i in range(n):
            condicate.append(i)
        for i in condicate:
            if len(E)<m:
               if i not in E:
                   E.append(i)
    #按照算法3计算对角矩阵
    SE = np.dot(A[:,E].transpose(), A[:,E])
    I = np.eye(m)
    D = np.zeros((m,m))
    for i in range(1,m):
        D[i,:] = SE[i,:]/math.sqrt(SE[i,j])
        D[:,[i]] = SE[:,[i]]/math.sqrt(SE[i,j])
        I[i,:] = I[i,:]/math.sqrt(SE[i,j])
        for j in range(i+1,m):
            mult = -D[j,i]/D[i,j]
            D[j,:] = mult*D[i,:] + D[j,:]
            D[:,[j]] = mult*D[:,[i]] + D[:,[j]]
            I[j,:] = mult*I[i,:] + I[j,:]
        P = I.transpose()
    #算法的5-7步
    T = np.dot(np.dot(P,Sigma),V[0:m,:])
    Q,R = np.linalg.qr(A[:,E])#进行Q,R分解
    W = np.dot(Q,T)#最终的降维结果,m*n的矩阵
    W = W.transpose()#转化为与输入矩阵的形式
    return W
