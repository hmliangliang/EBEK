# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\4\2 0002 08:51:51
# File:         demo.py
# Software:     PyCharm
#------------------------------------

import numpy  as np
from scipy.io import loadmat
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
import LKPE
if __name__ == '__main__':
    start = time.time()
    d = 8#d为数据降维后的维数
    #导入矩阵数据
    data = loadmat('Breastw.mat')
    data = data['Breastw']
    data = np.array(data)#将数据转换成narray格式
    col = data.shape[1]#数据的列数
    label = data[:,[col-1]]#获取数据的类标签
    data = data[:,0:col-1]#获取数据的数据部分
    data = minmax_scale(data, axis=0)
    data = LKPE.LKPE(data, d)#执行LKPE算法
    #进行分类预测
    num = int(2*data.shape[0]/3)#获取数据的训练集样本个数
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data[0:num,:],label[0:num,0])
    y = model.predict(data[num:data.shape[0]-1,:])
    print('算法测试结果的准确率为:', sum(y == label[num:data.shape[0]-1,0])/(data.shape[0]-num))
    end = time.time()
    print('算法的运行时间为:',end-start)