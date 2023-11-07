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
    ref_contour = ref_conts[1]
    
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

    minRect = [None]*len(cont)

    # gets minimum area rectangle for each contour
    # for i, c in enumerate(cont):
    #     minRect[i] = cv2.minAreaRect(c)
    
    # image for debugging
    # all_contours = 255*np.ones((img.shape[0], img.shape[1],3), dtype=np.uint8) #cv2.cvtColor(blobs,cv2.COLOR_GRAY2BGR
    for c in range(len(cont)):
        
        # creates minumum size and ignores the first background contour
        if(c!=0 and cv2.contourArea(cont[c])>10):
            # draws all  contour
            # cv2.drawContours(all_contours, cont, c,(0,0,0), thickness=10)

            good = (255,0,0)
            bad = (0,0,255)

            match_score = cv2.matchShapes(ref_contour, cont[c], cv2.CONTOURS_MATCH_I3,0)
            # print(match_score)

            # draws the bounding box
            # box = cv2.boxPoints(minRect[c])
            # box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
            if(match_score >0.2 ):
                print(match_score)
                # cv2.drawContours(all_contours, [box], 0, bad, thickness=5)
                cv2.drawContours(img, cont, c, bad, thickness=cv2.FILLED)
            # else:
                # cv2.drawContours(all_contours, [box], 0, good, thickness=5)

    # cv2.namedWindow("Reference contour",cv2.WINDOW_NORMAL)
    # cv2.imshow("Reference contour", ref_contour_img)

    # cv2.namedWindow("Reference binary",cv2.WINDOW_NORMAL)
    # cv2.imshow("Reference binary", ref_dst)

    cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    cv2.imshow("Original Image", img)

    # cv2.namedWindow("All contours",cv2.WINDOW_NORMAL)
    # cv2.imshow("All contours", all_contours)

    cv2.imwrite("spade-terminal-output.png", img)


    cv2.waitKey()

