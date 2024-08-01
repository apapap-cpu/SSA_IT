# SSA_IT
# MDTSC
The code in the folder 'MDTSC' is based on related work to compress images.<br>
Working:<br>
    python start.py --input <input_dir> --ori <original_image_dir> --output <output_dir><br>
    Notice that input_dir is the path of file in '.mat' format of the image; original_image_dir is the path of file in '.png' or '.jpg' format of the image and output_dir is the directory you would like to save the result.<br>
    A possible input is like:<br>
        python start.py --input './caseandresult/input.mat' --ori './source/original.png' --output './result'

# Split_img
The code in the folder 'Split_img' is designed to add dividing lines to two images, resulting in new images. The dividing lines partition each image into overlapping and non-overlapping sections, where the overlapping section comprises the common parts of both images, and the non-overlapping section contains the remaining parts.<br>
Working:<br>
    python split.py --input1 <input1_dir> --input2 <input2_dir> --output1 <output1_dir> --output2 <output2_dir><br>
    Notice that all the four paths of images are files in '.png' or '.jpg' formats of the images. Input are the images you'd like to process and output are the paths you'd like to save the result.<br>
    A possible input is like:(if you put the input images in folder 'Split_img')<br>
        python split.py --input1 'img1.png' --input2 'img2.png' --output1 'res1.png' --output2 'res2.png'

# Overlap
The code in the folder 'Overlap' is designed to deal with the overlapping and non-overlapping sections of an image respectively and concatenate them together to reconstruct the image. <br>
Working:<br>
    python start.py --ori <ori_img1> <ori_img2> --cross <cross1> <cross2> --uncross <uncross1> <uncross2> --output <out1> <out2> --iter <iteration><br>
    Notice that ori_img1, cross1, uncross1 are respectively the original file(the original image that hasn't been split), overlapping file, non-overlapping file in '.png' or '.jpg' formats of image 1. The same rule is for image 2 with ori_img2, cross2 and uncross2. Out1 and out2 are where you'd like to save the two processed images. Iteration shows how many times of iterations the code will execute, and the best iteration will be choosed to print the final result.<br>
    A possible input is like:<br>
        python start.py --ori './source/img1.png' './source/img2.png' --cross './source/cross1.png' './source/cross2.png' --uncross './source/uncross1.png' './source/uncross2.png' --output './result/out1.png' './result/out2.png' --iter 10

# Additional tips of Overlap
1. ./caseandresult, ./tmp_img, ./tmp_res, ./tmp_mat are four folders of vital importance to the successful execution of the code. Although the code will examine whether they exist and create these folders, they may cause unpredicted errors so that please examine whether these four folders have been created after the first time start.py has been executed.
2. When the two sections of an image, namely overlapping and non-overlapping sections are concatenated to create the final result, their sequence that which section is on the left and which section is on the right is critical to the quality of reconstructed images. Please pay attention to the two comments in start.py. You may change the order if your sequence is different from the current situation in the code.
