from skimage.metrics import structural_similarity as ssim
from parm import Para as P
from PIL import Image
import scipy.io as sio
import numpy as np
from tenor2block import * 
from initbase import *
from tsa import *
from tprod import *
from tdl import *
from psnr3d import *
import matplotlib.pyplot as plt
import os
import cv2
import datetime
import argparse

def denoise(OX, original_img_path, output_dir):
    OX = OX / 255
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 4, 1), plt.title('origin')
    plt.imshow(OX[:, :, 1], cmap='gray'), plt.axis('off')

    size_OX = OX.shape
    X = OX + 0.2 * np.random.rand(size_OX[0], size_OX[1], size_OX[2])

    plt.subplot(3, 4, 2), plt.title('dirty')
    plt.imshow(X[:, :, 1], cmap='gray'), plt.axis('off')

    size_X = X.shape
    Xc = t2b(X, P)
    size_Xc = Xc.shape
    Xhat = np.fft.fft(Xc, axis=-1)
    D0 = init3D(P)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(P.maxiter):
        plt.figure(figsize=(10, 10))
        starttime = datetime.datetime.now()

        if i == 0:
            B = tsta(Xc, P, D0)
        else:
            B = tsta(Xc, P, D0, B0)

        D0 = tendl(Xhat, B, P)
        B0 = tsta(Xc, P, D0)
        lu = tensor_prod(D0, 'a', B0, 'a')
        emsi = b2t(lu, P, size_X)

        plt.imshow(emsi[:, :, 1], cmap='gray'), plt.axis('off')

        ps = psnr(OX * 255, emsi * 255)

        endtime = datetime.datetime.now()
        plt.savefig(os.path.join(output_dir, f'sparsecoding{i}.png'))

        img1 = cv2.imread(original_img_path) 
        img2 = cv2.imread(os.path.join(output_dir, f'sparsecoding{i}.png'))
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        average_ssim = ssim(img1_gray, img2_gray)

        print('iter={}, current PSNR = {}, SSIM = {}, running time = {}, size_X = {}'.format(i, ps, average_ssim, (endtime - starttime).seconds, size_X))
    
    plt.savefig(os.path.join(output_dir, 'sparsecoding.png'))


def main(input_dir, original_img_path, output_dir):
    if not os.path.exists(input_dir):
        print(f"Error: Path of input image '{input_dir}' does not exist.")
        return
    if not os.path.exists(original_img_path):
        print(f"Error: Path of original image '{original_img_path}' does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    OX = sio.loadmat(input_dir)
    OX = OX['imageTensor']
    denoise(OX, original_img_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image denoising program.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input MAT file (e.g. ./caseandresult/input.mat)')
    parser.add_argument('--ori', type=str, required=True, help='Path to the original image (e.g. ./source/original.png)')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the output images (e.g. ./result)')

    args = parser.parse_args()

    main(args.input, args.ori, args.output)
