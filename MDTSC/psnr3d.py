# compute the psnr 
# 计算两个图像的峰值信噪比（衡量图像压缩的效果）
import numpy as np
import math

def psnr(image1,image2):
    # 通道：颜色分量
    [m,n,k] = image1.shape # 高度，宽度，通道数
    gg =0
    for kk in range(k):
        mse =np.square(image1[:,:,kk]-image2[:,:,kk]).sum()/(m*n) # 对每个颜色通道计算两个图像的均方误差
        gg = gg + 10*math.log10(255**2/mse)
    
    gg = gg/k # 所有通道的平均psnr
    return gg

if __name__ == '__main__': # 如果直接运行模块，随机生成两个小图像并计算psnr
   a = np.random.randint(0,5,size = [2,3,2])
   b = np.random.randint(0,5,size = [2,3,2])
   print(a[:,:,0])
   print(a[:,:,1])
   print(b[:,:,0])
   print(b[:,:,1])
   ab= psnr(a,b)
  # a = np.array([[1,2,3],[1,2,3]])
  # print(a)
  # b = np.array([[2,2,2],[3,3,3]])
  # print(b)
  # ab = psnr(a,b)
   print(ab)


