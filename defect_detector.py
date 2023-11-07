'''
matchShapes is a powerful OpenCV function that takes in two binary images ( or contours ) and finds the
distance between them using Hu Moments (M. K. Hu, "Visual Pattern Recognition by Moment Invariants", IRE
Trans. Info. Theory, vol. IT-8, pp.179–187, 1962). See some information about Hu Moments and how to use
matchShapes on the web at: https://learnopencv.com/shape-matching-using-hu-moments-c-python/.
Using matchShapes, write a program that takes as input the image shown in Figure 5 (a), “spade-terminal.png,”
detects defective spade terminals and marks them by painting them in red as shown in Figure 5 (b). The program
should save the output image as spade-terminal-output.png


Note to self: be sure to run "conda activate cve" before running this file

'''
import cv2
import numpy as np
import argparse
import random as rand



if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description='defective part detector')
    parser.add_argument('-r', '--ref', default = 'spade-reference.png')
    parser.add_argument('-i', '--input', default = 'spade-terminal.png')

    args = parser.parse_args()
    # Read image
    img = cv2.imread(args.input)

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binary
    thr,dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    
    uneroded = dst
    # clean up
    for i in range(1):
        dst = cv2.erode(dst, None)
    for i in range(1):
        dst = cv2.dilate(dst, None)

    # find contours with hierachy
    cont, hier = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)