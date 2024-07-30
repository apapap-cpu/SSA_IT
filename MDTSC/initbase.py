# init a base
# 创建并预处理一个三维数组

import numpy as np
from parm import Para as P
import math

def init3D(P):
    patsize = P.patsize
    r = P.r
    Dmat = np.random.rand(patsize**3,r)*2-1
    Dm = np.sqrt(np.sum(Dmat*Dmat,axis = 0))  # a 1*30 array
    szi =np.shape(Dm)
    for i in range(szi[0]):
        Dmat[:,i] =Dmat[:,i]/Dm[i]
    # 将每列标准化
    D = np.transpose(np.reshape(Dmat,[patsize**2,patsize,r],order = 'F'),[0,2,1])
    # 将矩阵重塑为三维化，并进行转置
    return D

if __name__ == '__main__':
    print(init3D(P).shape)