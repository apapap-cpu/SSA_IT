import numpy as np
from PIL import Image
from scipy.io import savemat
import sys
import os

def img_to_mat(img_file_path, mat_file_path):
    img = Image.open(img_file_path).convert("RGB")
    img_data = np.array(img, dtype=np.float32)
    mat_file_folder = os.path.dirname(mat_file_path)
    if not os.path.exists(mat_file_folder):
        os.makedirs(mat_file_folder)
    data = {'imageTensor': img_data}
    savemat(mat_file_path, data)