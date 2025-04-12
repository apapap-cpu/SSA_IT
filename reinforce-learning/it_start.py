# where the code start
from skimage.metrics import structural_similarity as ssim
from parm import Para as P
from PIL import Image
import scipy.io as sio
from scipy.ndimage import zoom
import numpy as np
from tenor2block import * 
from initbase import *
from tsa import *
from tprod import *
from tdl import *
from psnr3d import *
from to_mat import *
from additional import *
from oabac import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
# 原始完整图像已经存放在： ./source/(left_ori; middle_ori).png
# 原始完整图像mat文件已经存放在： ./caseandresult/(left_ori; middle_ori).mat
# 原始分开图像已经存放在：./split/(lc; lu; mlc; mlu).png
# 原始分开图像mat文件已经存放在：./caseandresult/(lc; lu; mlc; mlu).mat
def denoise(c1, c2, u1, u2):
    plt.figure(figsize=(10, 10))
    iter = 1
    start1 = datetime.datetime.now()
    it0, D0, Xc0, Bcl, Bcm = getbest_cross(
        './split/lc.png', 
        './split/mlc.png', 
        './caseandresult/lc.mat', 
        './caseandresult/mlc.mat',
        './result_cross/left_c.png',
        './result_cross/middle_c.png',
        iter
        )
    end1 = datetime.datetime.now()
    time1 = (end1 - start1).seconds # 计算cross部分用时

    start2 = datetime.datetime.now()
    it3, D1, Xc1, Bul = getbest_uncross('./split/lu.png', './caseandresult/lu.mat', './result_cross/left_un.png', iter)
    end2 = datetime.datetime.now()
    time2 = (end2 - start2).seconds # Left的uncross部分用时

    start3 = datetime.datetime.now()
    it4, D2, Xc2, Bum = getbest_uncross('./split/mlu.png', './caseandresult/mlu.mat', './result_cross/middle_un.png', iter)
    end3 = datetime.datetime.now()
    time3 = (end3 - start3).seconds # Middle的uncross部分用时

    img1 = './result_cross/left_c.png'
    img2 = './result_cross/left_un.png'
    img3 = './result_cross/middle_c.png'
    img4 = './result_cross/middle_un.png'
    out1 = './result_cross/left.png'
    out2 = './result_cross/middle.png'

    start4 = datetime.datetime.now()
    concatenate_2images(img1, img2, out1) # Left重建顺序为 cross uncross
    end4 = datetime.datetime.now()
    time4 = (end4 - start4).seconds # Left的图像拼接用时

    start5 = datetime.datetime.now()
    concatenate_2images(img4, img3, out2) # Middle重建顺序为 uncross cross
    end5 = datetime.datetime.now()
    time5 = (end5 - start5).seconds # Middle的图像拼接用时
    
    mat1 = './caseandresult/left.mat'
    mat2 = './caseandresult/middle.mat'
    ori1 = './source/left_ori.png'
    ori2 = './source/middle_ori.png'
    mat3 = './caseandresult/left_ori.mat'
    mat4 = './caseandresult/middle_ori.mat'
    ps1 = get_psnr(out1, mat1, ori1, mat3)
    ps2 = get_psnr(out2, mat2, ori2, mat4)

    ssim1 = get_ssim(ori1, out1)
    ssim2 = get_ssim(ori2, out2)

    time_left = time1 + time2 + time4
    time_middle = time1 + time3 + time5
    
    print('PSNR 1 = {}, PSNR 2 = {}, SSIM 1 = {}, SSIM 2 = {}'.format(ps1, ps2, ssim1, ssim2)) 
    print('Best iteration for cross = {}, for uncross 1 = {}, for uncross 2 = {}'.format(it0, it3, it4))
    print('Time for left = {}, for middle = {}, for cross = {}'.format(time_left, time_middle, time1))

    OX1 = sio.loadmat(mat1)
    OX1 = OX1['imageTensor']
    OX2 = sio.loadmat(mat2)
    OX2 = OX2['imageTensor']
    # D3, Xc1, Xhat1, size_X1 = process_image(OX1)
    # D4, Xc2, Xhat2, size_X2 = process_image(OX2)
    Xc1 = t2b(OX1 / 255, P)
    Xc2 = t2b(OX2 / 255, P)
    D1 = np.concatenate((D0, D1), axis=0)
    D2 = np.concatenate((D2, D0), axis=0)
    B1 = np.concatenate((Bcl, Bul), axis=1)
    B2 = np.concatenate((Bum, Bcm), axis=1)
    print('Left size: D = {}, Xc = {}, B = {}'.format(D1.nbytes, Xc1.nbytes, B1.nbytes))
    print('Middle size: D = {}, Xc = {}, B = {}'.format(D2.nbytes, Xc2.nbytes, B2.nbytes))
    get_oabac(B1, 'Left')
    get_oabac(B2, 'Middle')

    # with open('./mytxt/B.txt', 'w') as f:
    #     f.write('B1:\n')
    #     for i in range(B1.shape[0]): 
    #         f.write(f'Slice {i}:\n')  
    #         np.savetxt(f, B1[i], fmt='%.6f', delimiter=',')  
    #         f.write('\n') 

    #     f.write('B2:\n')
    #     for i in range(B2.shape[0]):
    #         f.write(f'Slice {i}:\n')
    #         np.savetxt(f, B2[i], fmt='%.6f', delimiter=',')
    #         f.write('\n')
    # Xc1 = np.concatenate((Xc0, Xc1), axis=1)
    # Xc2 = np.concatenate((Xc2, Xc0), axis=1)    

    # r1 = calculate_compression(D1, Xc1, mat3, 'left')
    # r2 = calculate_compression(D2, Xc2, mat4, 'middle')

if __name__ =='__main__':
    c1 = sio.loadmat('./caseandresult/lc.mat')
    c2 = sio.loadmat('./caseandresult/mlc.mat')
    u1 = sio.loadmat('./caseandresult/lu.mat')
    u2 = sio.loadmat('./caseandresult/mlu.mat')

    c1 = c1['imageTensor']
    c2 = c2['imageTensor']
    u1 = u1['imageTensor']
    u2 = u2['imageTensor']
    denoise(c1, c2, u1, u2)