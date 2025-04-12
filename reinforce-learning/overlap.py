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
def overlap():
    iter = 1
    start = datetime.datetime.now()
    it0, D0, Xc0, Bcl, Bcm = getbest_cross(
        './split/lc.png', 
        './split/mlc.png', 
        './caseandresult/lc.mat', 
        './caseandresult/mlc.mat',
        './result_cross/left_c.png',
        './result_cross/middle_c.png',
        iter
        )
    end = datetime.datetime.now()
    time = (end - start).seconds # 计算cross部分用时

    print('Time for cross = {}'.format(time))
    

    

if __name__ == '__main__':
    overlap()
