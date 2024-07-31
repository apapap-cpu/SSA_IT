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
import matplotlib.pyplot as plt
import cv2
import sys
import shutil

def process_image(OX):
    OX = OX/255  
    size_OX = OX.shape
    X = OX+0.2*np.random.rand(size_OX[0],size_OX[1],size_OX[2])

    size_X = X.shape
    Xc = t2b(X,P)
    Xhat = np.fft.fft(Xc,axis=-1)
    D = init3D(P)
    
    return D, Xc, Xhat, size_X

def calculate_compression(D, Xc, ori_mat_path, tag): 
    OX = sio.loadmat(ori_mat_path)
    OX = OX['imageTensor']
    ori_size = OX.nbytes
    new_size = D.nbytes + Xc.nbytes
    ratio = (1 - (new_size / ori_size)) * 100

    print('{} size of original = {}, dic = {}, Xc = {}, total = {}'.format(tag, ori_size, D.nbytes, Xc.nbytes, new_size))
    print('{} compression ratio: {}'.format(tag, ratio))
    return ratio

def calculate_dictionary(Xc, Xhat):
    B = tsta(Xc, P, init3D(P))
    D = tendl(Xhat, B, P)
    return D

def reconstruct_image(D, Xc, size_X):
    B = tsta(Xc, P, D)
    lu = tensor_prod(D, 'a', B, 'a')
    reconstructed = b2t(lu, P, size_X)
    return reconstructed, B

def merge_image(Xc1, Xc2, size_X1, size_X2):
    w1 = size_X1[1]
    w2 = size_X2[2]
    size_X = np.array([size_X1[0], w1 + w2, size_X1[2]])
    Xc = np.concatenate((Xc1, Xc2), axis=1)
    Xhat = np.fft.fft(Xc,axis=-1)
    D = calculate_dictionary(Xc, Xhat) 
    reconstructed = reconstruct_image(D, Xc, size_X)
    return reconstructed

def concatenate_2images(img_path1, img_path2, output_path):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    total_width = img1.width + img2.width
    new_img = Image.new('RGB', (total_width, img1.height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.save(output_path)

def concatenate_3images(img_path1, img_path2, img_path3, output_path):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    img3 = Image.open(img_path3)

    heights = [img1.height, img2.height, img3.height]
    max_height = max(heights)
    total_width = img1.width + img2.width + img3.width

    new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))
    new_img.save(output_path)

def save_image(image_data, filename):
    fig,ax = plt.subplots()
    ax.imshow(image_data, cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def resize_image(image, reference_shape):
    zoom_factors = [n / i for n, i in zip(reference_shape, image.shape)]
    return zoom(image, zoom_factors, order=3) 

def get_psnr(img1_path, mat1_path, img2_path, mat2_path):
    img_to_mat(img1_path, mat1_path)
    OX1 = sio.loadmat(mat1_path)
    OX1 = OX1['imageTensor']

    img_to_mat(img2_path, mat2_path)
    OX2 = sio.loadmat(mat2_path)
    OX2 = OX2['imageTensor']

    if OX1.shape != OX2.shape:
        OX1 = resize_image(OX1, OX2.shape)

    ps = psnr(OX1, OX2)
    return ps

def get_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path) 
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    average_ssim = ssim(img1_gray, img2_gray)

    return average_ssim

def copy_img(src, dest):
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    shutil.copy2(src, dest)

tmp_img = './tmp_img/tmpimg.png'
tmp_img1 = './tmp_img/tmpimg1.png'
tmp_img2 = './tmp_img/tmpimg2.png'
tmp_mat = './tmp_mat/tmpmat.mat'
tmp_mat1 = './tmp_mat/tmpmat1.mat'
tmp_mat2 = './tmp_mat/tmpmat2.mat'

def deal_with_dic(D1, D2):
    D0 = np.empty_like(D1) 
    small_val = 1e-10

    for i in range(D1.shape[0]): 
        for j in range(D1.shape[1]): 
            for k in range(D1.shape[2]):
                d1_value = D1[i, j, k]
                d2_value = D2[i, j, k]
                
                if np.isclose(d1_value, 0, atol=small_val) and np.isclose(d2_value, 0, atol=small_val):
                    D0[i, j, k] = 0  
                elif np.isclose(d1_value, 0, atol=small_val):
                    D0[i, j, k] = d2_value  
                elif np.isclose(d2_value, 0, atol=small_val):
                    D0[i, j, k] = d1_value 
                else:
                    D0[i, j, k] = (d1_value + d2_value) / 2
    return D0

def getbest_cross(img1, img2, mat1, mat2, dest1, dest2, iter):
    OX1 = sio.loadmat(mat1)
    OX1 = OX1['imageTensor']
    OX2 = sio.loadmat(mat2)
    OX2 = OX2['imageTensor']
    D1, Xc1, Xhat1, size_X1 = process_image(OX1)
    D2, Xc2, Xhat2, size_X2 = process_image(OX2)
    max_ps = 0
    ct = -1
    for i in range(iter):
        print('Now for cross iteration {}'.format(i))
        D1 = calculate_dictionary(Xc1, Xhat1)
        D2 = calculate_dictionary(Xc2, Xhat2)
        D0 = deal_with_dic(D1, D2)
        Xc0 = 0.5 * (Xc1 + Xc2)
        re1, B1 = reconstruct_image(D0, Xc1, size_X1)
        re2, B2 = reconstruct_image(D0, Xc2, size_X2)
        save_image(re1[:,:,1], tmp_img1)
        save_image(re2[:,:,1], tmp_img2)
        tmp_ps = get_psnr(tmp_img1, tmp_mat1, img1, mat1)
        if tmp_ps > max_ps:
            max_ps = tmp_ps
            copy_img(tmp_img1, dest1)
            copy_img(tmp_img2, dest2)
            ct = i
            D = D0
            Br1 = B1
            Br2 = B2
    return ct, D, Xc0, Br1, Br2

def getbest_uncross(img, mat, dest, iter):
    OX = sio.loadmat(mat)
    OX = OX['imageTensor']
    D, Xc, Xhat, size_X = process_image(OX)
    max_ps = 0
    ct = -1
    for i in range(iter):
        print('Now for uncross iteration {}'.format(i))
        D0 = calculate_dictionary(Xc, Xhat)
        re0, B0 = reconstruct_image(D0, Xc, size_X)
        save_image(re0[:,:,1], tmp_img)
        tmp_ps = get_psnr(tmp_img, tmp_mat, img, mat)
        if tmp_ps > max_ps:
            max_ps = tmp_ps
            D = D0
            copy_img(tmp_img, dest)
            ct = i
            B = B0
    return ct, D, Xc, B
