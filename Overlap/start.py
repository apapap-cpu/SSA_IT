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
import matplotlib.pyplot as plt
import datetime
import argparse

def denoise(ori1, ori2, cross1, cross2, uncross1, uncross2, out1, out2, iter):
    ori1_mat = './caseandresult/ori1.mat'
    ori2_mat = './caseandresult/ori2.mat' 
    img_to_mat(ori1, ori1_mat)
    img_to_mat(ori2, ori2_mat)   

    cross1_mat = './caseandresult/cross1.mat'
    cross2_mat = './caseandresult/cross2.mat'
    img_to_mat(cross1, cross1_mat)
    img_to_mat(cross2, cross2_mat)

    uncross1_mat = './caseandresult/uncross1.mat'
    uncross2_mat = './caseandresult/uncross2.mat'
    img_to_mat(uncross1, uncross1_mat)
    img_to_mat(uncross2, uncross2_mat)

    img1 = './tmp_res/cross1.png'
    img2 = './tmp_res/uncross1.png'
    img3 = './tmp_res/cross2.png'
    img4 = './tmp_res/uncross2.png'

    out_mat1 = './caseandresult/out1.mat'
    out_mat2 = './caseandresult/out2.mat'

    plt.figure(figsize=(10, 10))
    start1 = datetime.datetime.now()
    it0, D0, Xc0, Bcl, Bcm = getbest_cross(cross1, cross2, cross1_mat, cross2_mat, img1, img3, iter)
    end1 = datetime.datetime.now()
    time1 = (end1 - start1).seconds 

    start2 = datetime.datetime.now()
    it3, D1, Xc1, Bul = getbest_uncross(uncross1, uncross1_mat, img2, iter)
    end2 = datetime.datetime.now()
    time2 = (end2 - start2).seconds 

    start3 = datetime.datetime.now()
    it4, D2, Xc2, Bum = getbest_uncross(uncross2, uncross2_mat, img4, iter)
    end3 = datetime.datetime.now()
    time3 = (end3 - start3).seconds 

    start4 = datetime.datetime.now()
    concatenate_2images(img1, img2, out1) # Please note the sequence of two images here
    end4 = datetime.datetime.now()
    time4 = (end4 - start4).seconds

    start5 = datetime.datetime.now()
    concatenate_2images(img4, img3, out2) # Please note the sequence of two images here
    end5 = datetime.datetime.now()
    time5 = (end5 - start5).seconds
    
    ps1 = get_psnr(out1, out_mat1, ori1, ori1_mat)
    ps2 = get_psnr(out2, out_mat2, ori2, ori2_mat)

    ssim1 = get_ssim(ori1, out1)
    ssim2 = get_ssim(ori2, out2)

    time_left = time1 + time2 + time4
    time_middle = time1 + time3 + time5
    
    print('PSNR 1 = {}, PSNR 2 = {}, SSIM 1 = {}, SSIM 2 = {}'.format(ps1, ps2, ssim1, ssim2)) 
    print('Time for img1 = {}, for img2 = {}, for cross = {}'.format(time_left, time_middle, time1))

    OX1 = sio.loadmat(out_mat1)
    OX1 = OX1['imageTensor']
    OX2 = sio.loadmat(out_mat2)
    OX2 = OX2['imageTensor']

    Xc1 = t2b(OX1 / 255, P)
    Xc2 = t2b(OX2 / 255, P)
    D1 = np.concatenate((D0, D1), axis=0)
    D2 = np.concatenate((D2, D0), axis=0)
    B1 = np.concatenate((Bcl, Bul), axis=1)
    B2 = np.concatenate((Bum, Bcm), axis=1)
    print('Img1 size: D = {}, Xc = {}, B = {}'.format(D1.nbytes, Xc1.nbytes, B1.nbytes))
    print('Img2 size: D = {}, Xc = {}, B = {}'.format(D2.nbytes, Xc2.nbytes, B2.nbytes))
    get_oabac(B1, 'Img1')
    get_oabac(B2, 'Img2')

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Denoise images.')
    
    parser.add_argument('--ori', required=True, nargs=2, help='Paths to the two original images (ori_img1, ori_img2).')
    parser.add_argument('--cross', required=True, nargs=2, help='Paths to the two cross images (cross1, cross2).')
    parser.add_argument('--uncross', required=True, nargs=2, help='Paths to the two uncross images (uncross1, uncross2).')
    parser.add_argument('--output', required=True, nargs=2, help='Paths to save output images (out1, out2).')
    parser.add_argument('--iter', required=True, type=int, help='Number of iterations for processing.')

    args = parser.parse_args()

    if len(args.ori) != 2 or len(args.cross) != 2 or len(args.uncross) != 2 or len(args.output) != 2:
        print("Error: Please provide exactly two images for each category (ori, cross, uncross, output).")
        print("Correct format: python start.py --ori <ori_img1> <ori_img2> --cross <cross1> <cross2> --uncross <uncross1> <uncross2> --output <out1> <out2> --iter <iteration>")
    else:
        denoise(args.ori[0], args.ori[1], args.cross[0], args.cross[1], args.uncross[0], args.uncross[1], args.output[0], args.output[1], args.iter)
