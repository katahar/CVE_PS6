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
    ref_img = cv2.imread(args.ref)
    img = cv2.imread(args.input)


    # reference image processing
    # Convert to gray-scale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    # Binary
    ref_thr,ref_dst = cv2.threshold(ref_gray, 60, 255, cv2.THRESH_BINARY)
    
    # clean up
    for i in range(1):
        ref_dst = cv2.erode(ref_dst, None)
    for i in range(1):
        ref_dst = cv2.dilate(ref_dst, None)

    # find contours with hierachy
    ref_conts, ref_hiers = cv2.findContours(ref_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # ref_contour = ref_conts[1]
    
    ref_contour_img = 255*np.ones((ref_img.shape[0], ref_img.shape[1],3), dtype=np.uint8) #cv2.cvtColor(blobs,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(ref_contour_img, ref_conts, 1,(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)), thickness=3)
    cv2.drawContours(ref_contour_img, ref_conts, 1,(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)), thickness=3)

    # input image processing
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

    all_contours = 255*np.ones((img.shape[0], img.shape[1],3), dtype=np.uint8) #cv2.cvtColor(blobs,cv2.COLOR_GRAY2BGR)
    for c in range(len(cont)):
        if(c!=0):
            cv2.drawContours(all_contours, cont, c,(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)), thickness=10)






    cv2.namedWindow("Reference contour",cv2.WINDOW_NORMAL)
    cv2.imshow("Reference contour", ref_contour_img)

    cv2.namedWindow("Reference binary",cv2.WINDOW_NORMAL)
    cv2.imshow("Reference binary", ref_dst)

    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)

    cv2.namedWindow("All contours",cv2.WINDOW_NORMAL)
    cv2.imshow("All contours", all_contours)


    cv2.waitKey()

