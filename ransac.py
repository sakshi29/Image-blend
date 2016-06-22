import numpy as np
import cv2
from matplotlib import pyplot as plt



def drawMatches(gray1, kp1, gray2, kp2, matches):


    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = gray1.shape[0]
    cols1 = gray1.shape[1]
    rows2 = gray2.shape[0]
    cols2 = gray2.shape[1]
    ##print rows1,rows2,cols1,cols2
    
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([gray1, gray1, gray1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([gray2, gray2, gray2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 255, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    ##cv2.imshow('Matched Features', out)
    ##cv2.waitKey(0)
    ##cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out



MIN_MATCH_COUNT = 10
img1 = cv2.imread('kleft.jpeg')
img2 = cv2.imread('kright.jpeg')
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
##img4 = drawMatches(img1,kp1,img2,kp2,matches[:10])

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print M
    h,w,c= img2.shape
    
    x,y,c=img1.shape
    x2,y2,c2=img2.shape
    print h,w
    print x,y
    print x2,y2
    
    
    img2=  cv2.warpPerspective(img2,M,(y+y2,h))

    
    
   
    out = np.zeros((max([x,x2]),y+y2,3), dtype='uint8')
    ##out[:x2,y:] = np.dstack([gr, gray2, gray2])

    """
    for i in range(0,y-1):
        for j in range(0,x2-1):
             out[j,i]=img1[j,i]
     
    k=200           
    for i in range (0,y2-1):
        for j in range (0,x2-1):
          if(k+i<y):
             out[j,k+i][0]=img2[j,i][0]*0.67+img1[j,k+i][0]*0.33
             out[j,k+i][1]=img2[j,i][1]*0.67+img1[j,k+i][1]*0.33
             out[j,k+i][2]=img2[j,i][2]*0.67+img1[j,k+i][2]*0.33
          else:
             out[j,k+i][0]=img2[j,i][0]*0.98
             out[j,k+i][1]=img2[j,i][1]*0.98
             out[j,k+i][2]=img2[j,i][2]*0.98
    """
    for i in range (0,y2-1):
        for j in range (0,x2-1):
          
             img2[j,i][0]=img2[j,i][0]*0.10+img1[j,i][0]*0.90
             img2[j,i][1]=img2[j,i][1]*0.10+img1[j,i][1]*0.90
             img2[j,i][2]=img2[j,i][2]*0.10+img1[j,i][2]*0.90
          
             
##cv2.imshow('img',out)
cv2.imshow('img1',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
##cv2.imshow('result',img2)
##cv2.imshow('re',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()    
