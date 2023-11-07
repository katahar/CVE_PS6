'''
In this problem set, you will study the sample program in Appendix A, understand how it works, and then
modify the program so that all five types of mechanical parts are correctly identified. Paint the internal lock
washers in purple and external lock washers in yellow.
Your program should take as input an image, “all-parts.png,” shown in Figure 3 (a) and generate a new file
“all-parts-output.png.”

Note to self: be sure to run "conda activate cve" before running this file

'''
import cv2
import numpy as np
import argparse
import random as rand


# check size (bounding box) is square
def isSquare(siz):
    ratio = abs(siz[0] - siz[1]) / siz[0]
    #print (siz, ratio)
    if ratio < 0.1:
        return True
    else:
        return False

# chekc circle from the arc length ratio
def isCircle(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    len = cv2.arcLength(cnt, True)
    ratio = abs(len - np.pi * 2.0 * radius) / (np.pi * 2.0 * radius)
    #print(ratio)
    if ratio < 0.1:
        return True
    else:
        return False

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description='Hough Circles')
    parser.add_argument('-i', '--input', default = 'all-parts.png')

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

    all_contours = 255*np.ones((img.shape[0], img.shape[1],3), dtype=np.uint8) #cv2.cvtColor(blobs,cv2.COLOR_GRAY2BGR)
    for c in range(len(cont)):
        cv2.drawContours(all_contours, cont, c,(rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255)), thickness=10)


    # each contoure
    for i in range(len(cont)):
        c = cont[i]
        h = hier[0,i]
        if h[2] == -1 and h[3] == 0:
            # no child and parent is image outer - spade terminal
            img = cv2.drawContours(img, cont, i, (0,0,255),-1)
        elif h[3] == 0 and hier[0,h[2]][2] == -1: 
            # with child
            if isCircle(c):
                if isCircle(cont[h[2]]): #washer
                    # double circle
                    img = cv2.drawContours(img, cont, i, (0,255,0),-1)
                else: #internal lockwasher, because the inside is not a circle
                    img = cv2.drawContours(img, cont, i, (128,0,128),-1)
            else:
                # 1 child and shape bounding box is not squre 
                if not isSquare(cv2.minAreaRect(c)[1]) and hier[0,h[2]][0] == -1 and hier[0,h[2]][1] == -1: #ring terminal
                    img = cv2.drawContours(img, cont, i, (255,0, 0),-1)
                #elif(isSquare(cv2.minAreaRect(c)[1])): #square bounding box - external lock washer
                else:
                    img = cv2.drawContours(img, cont, i, (0,255,255),-1)

                    


    # cv2.namedWindow("pre erode",cv2.WINDOW_NORMAL)
    # cv2.imshow("pre erode", uneroded)

    
    # cv2.namedWindow("post erode",cv2.WINDOW_NORMAL)
    # cv2.imshow("post erode", dst)

    # cv2.namedWindow("contours",cv2.WINDOW_NORMAL)
    # cv2.imshow("contours", all_contours)

    cv2.namedWindow("Processed_image",cv2.WINDOW_NORMAL)
    cv2.imshow("Processed_image", img)

    cv2.imwrite("all-parts-output.png", img)
    cv2.waitKey()
