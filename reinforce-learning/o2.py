from tenor2block import * 
from initbase import *
from tsa import *
from tprod import *
from tdl import *
from psnr3d import *
from to_mat import *
from additional import *
from oabac import *
import datetime
import matplotlib
matplotlib.use('Agg')  # 使用无图形界面后端

def process_image_pair(image1_path, image2_path, mat1_path, mat2_path, result1_path, result2_path, iter):
    """
    处理一对裁剪后的图像（如 lc_left_split.png 和 mlc_left_split.png），调用 getbest_cross 进行压缩
    """
    start = datetime.datetime.now()
    # 调用 getbest_cross，解包为 5 个变量
    it, D, Xc, B1, B2 = getbest_cross(
        image1_path,
        image2_path,
        mat1_path,
        mat2_path,
        result1_path,
        result2_path,
        iter
    )
    end = datetime.datetime.now()
    time = (end - start).seconds
    print(f"Time for processing pair: {time}")
    print('D = {}, Xc = {}'.format(D.nbytes, Xc.nbytes))
    return it, D, Xc, B1, B2, time

def process_images():
    """
    按图像对（裁剪后的左或右部分）处理图像
    """
    # 定义左部分图像对和右部分图像对
    image_pairs = [
        # 左部分图像对
        ('./overlap_folder2/lc_4_split.png', './overlap_folder2/mlc_4_split.png', './caseandresult4/lc_4.mat', './caseandresult4/mlc_4.mat', './result_cross4/lc_4_compressed.png', './result_cross4/mlc_4_compressed.png')
        
    ]

    # 迭代次数
    iter = 1

    # 分别处理每对图像
    for image1_path, image2_path, mat1_path, mat2_path, result1_path, result2_path in image_pairs:
        prefix1 = os.path.basename(image1_path).split('_')[0]  # 提取第一个图像的前缀 lc 或 mlc
        prefix2 = os.path.basename(image2_path).split('_')[0]  # 提取第二个图像的前缀 lc 或 mlc
        part = os.path.basename(image1_path).split('_')[1]     # 提取部分信息 left 或 right
        # 调用 process_image_pair
        it, D, Xc, B1, B2, time = process_image_pair(
            image1_path, image2_path, mat1_path, mat2_path, result1_path, result2_path, iter
        )
       
    img1 = './result_cross4/lc_1_compressed.png'
    img2 = './result_cross4/lc_2_compressed.png'
    img3 = './result_cross4/lc_3_compressed.png'
    img4 = './result_cross4/lc_4_compressed.png'
    img5 = './result_cross4/lu_1_compressed.png'
    img6 = './result_cross4/lu_2_compressed.png'
    img7 = './result_cross4/lu_3_compressed.png'
    img8 = './result_cross4/lu_4_compressed.png'
    img9 = './result_cross4/mlc_1_compressed.png'
    img10 = './result_cross4/mlc_2_compressed.png'
    img11 = './result_cross4/mlc_3_compressed.png'
    img12 = './result_cross4/mlc_4_compressed.png'
    img13 = './result_cross4/mlu_1_compressed.png'
    img14 = './result_cross4/mlu_2_compressed.png'
    img15 = './result_cross4/mlu_3_compressed.png'
    img16 = './result_cross4/mlu_4_compressed.png'
    out1 = './result_cross4/left.png'
    out2 = './result_cross4/middle.png'
    
    concatenate_8images(img1, img2, img3, img4, img5, img6, img7, img8, out1)
    concatenate_8images(img13, img14, img15, img16, img9, img10, img11, img12, out2)

if __name__ == '__main__':
    process_images()
