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
import matplotlib.pyplot as plt
import datetime
# 原始完整图像已经存放在： ./source/(left_ori; middle_ori).png
# 原始完整图像mat文件已经存放在： ./caseandresult/(left_ori; middle_ori).mat
# 原始分开图像已经存放在：./split/(lc; lu; mlc; mlu).png
# 原始分开图像mat文件已经存放在：./caseandresult/(lc; lu; mlc; mlu).mat
def non():
    iter = 1
    start2 = datetime.datetime.now()
    it3, D1, Xc1, Bul = getbest_uncross('./temp/lu_right_split.png', './caseandresult2/lu_right.mat', './result_cross2/lu_right_compressed.png', iter)
    end2 = datetime.datetime.now()
    time2 = (end2 - start2).seconds # Left的uncross部分用时

    start3 = datetime.datetime.now()
    it4, D2, Xc2, Bum = getbest_uncross('./temp/mlu_right_split.png', './caseandresult2/mlu_right.mat', './result_cross2/mlu_right_compressed.png', iter)
    end3 = datetime.datetime.now()
    time3 = (end3 - start3).seconds # Middle的uncross部分用时
    print('D1 = {}, Xc1 = {}'.format(D1.nbytes, Xc1.nbytes))
    print('D2 = {}, Xc2 = {}'.format(D2.nbytes, Xc2.nbytes))
    print('Time for non-left = {}, non-middle = {}'.format(time2, time3))

if __name__ == '__main__':
    non()