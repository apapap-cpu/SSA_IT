import cv2
import numpy as np
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

def extract_features(image, sift):
    return sift.detectAndCompute(image, None)

def main(input1_img, input2_img, output1_img, output2_img):
    start_time = time.time()

    img1 = cv2.imread(input1_img, 0)
    img2 = cv2.imread(input2_img, 0)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one of the input images. Please check the file paths.")

    sift = cv2.SIFT_create()
    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(extract_features, img1, sift)
        future2 = executor.submit(extract_features, img2, sift)
        keypoints1, descriptors1 = future1.result()
        keypoints2, descriptors2 = future2.result()

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    height, width = img2.shape
    img1_transformed = cv2.warpPerspective(img1, M, (width, height))

    overlap = cv2.bitwise_and(img1_transformed, img2)

    contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    min_x = width
    max_x = 0

    for contour in contours:
        if cv2.contourArea(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            max_x = max(max_x, x + w)

    overlap_area = sum(cv2.contourArea(contour) for contour in contours)
    overlap_width = overlap_area / 1920.0
    overlap_x_img1 = min_x + overlap_width / 2

    if 0 < min_x < width:
        x_draw = min_x
    elif 0 < max_x < width:
        x_draw = max_x
    
    print(x_draw)

    left_overlap = x_draw < width / 2

    if left_overlap:
        overlap_img2 = width - x_draw
        overlap_img1 = overlap_x_img1 

        max_overlap_width = max(overlap_img1, overlap_img2)
        overlap_x_img1 = max_overlap_width
        x_draw = width - max_overlap_width

        cv2.rectangle(img1_color, (5, 0), (int(overlap_x_img1), height-5), (0, 255, 0), 15)
        cv2.rectangle(img2_color, (int(x_draw), 0), (width - 7, height), (0, 255, 0), 15)
    else:
        overlap_img2 = x_draw
        overlap_img1 = width - overlap_x_img1
        max_overlap_width = max(overlap_img1, overlap_img2)
        x_draw = max_overlap_width
        overlap_x_img1 = width - max_overlap_width
        cv2.rectangle(img1_color, (int(overlap_x_img1), 0), (width - 7, height - 5), (0, 255, 0), 15)
        cv2.rectangle(img2_color, (5, 0), (int(x_draw), height-5), (0, 255, 0), 15)
        
    print(overlap_x_img1)
    print(x_draw)
    print("Vertical line in img1 at x-coordinate: {}".format(overlap_x_img1))
    print("Vertical line in img2 at x-coordinate: {}".format(x_draw))

    cv2.imwrite(output1_img, img1_color)
    cv2.imwrite(output2_img, img2_color)

    end_time = time.time()
    print("Time Delay: {} seconds".format(end_time - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process two images and find their overlap.')
    parser.add_argument('--input1', required=True, help='Path to the first input image.')
    parser.add_argument('--input2', required=True, help='Path to the second input image.')
    parser.add_argument('--output1', required=True, help='Path to save the first output image.')
    parser.add_argument('--output2', required=True, help='Path to save the second output image.')

    args = parser.parse_args()

    if not os.path.isfile(args.input1) or not os.path.isfile(args.input2):
        print("Error: One or both input files do not exist. Please check the file paths.")
    else:
        main(args.input1, args.input2, args.output1, args.output2)
