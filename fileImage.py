############################################
## Import OpenCV
import numpy as np
import cv2
import matplotlib.pyplot as plt
#############################################
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
##############################################
##Read the image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img1 = cv2.imread('testsa.jpg')
img2 = cv2.imread('testsa3.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2=  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

##face detection
faces = face_cascade.detectMultiScale(gray1, 1.3, 5)
for (x,y,w,h) in faces:
         cv2.rectangle(gray1,(x,y),(x+w,y+h),(255,0,0),2)
         
faces = face_cascade.detectMultiScale(gray2, 1.3, 5)
for (x2,y2,w2,h2) in faces:
         cv2.rectangle(gray2,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)           
       
height, width, channels = img1.shape
##print height,width,channels

height2, width2, channels2 = img2.shape
##print height2,width2,channels2

##1st image
xl=x-w
xr=4*w
yt=y-h
yb=height
print x,y,w,h
print xl,xr,yt,yb
##2nd image
"""
xl2=x2-w2
xr2=4*w2
yt2=y2-h2
yb2=height2
print xl2,xr2,yt2,yb2
cv2.rectangle(gray2,(xl2,yt2),(xl2+xr2,yt2+yb2),(255,0,0),2)
"""
cv2.rectangle(gray1,(xl,yt),(xl+xr,yt+yb),(255,0,0),2)


############################################
"""
# Initiate ORB detector

orb = cv2.ORB()

mat=[]
mat2=[]
des11=[]
des22=[]
a=[]
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray1,None)
kp2, des2 = orb.detectAndCompute(gray2,None)



for i in range (0,len(des1)):
    des11.append(des1[i].tolist())
des11 = np.array(des11)
for i in range (0,len(des2)):
    des22.append(des2[i].tolist())
des22 = np.array(des22)
print des22  
############################################
for m in kp1:
    
   if(  m.pt[0]>(xl+xr) or m.pt[1]<yt ):
        mat.append(m)
               
############################################
  
for n in kp2:
    if (n.pt[0]<(xl2) or n.pt[1]<yt2):
        mat2.append(n)
       
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in  order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

###############ra####################################################

# Draw first 10 matches.
img4 = drawMatches(gray1,kp1,gray2,kp2,matches[:10])
##dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
############################################
## Show the image
"""
cv2.imshow('img',gray1)
##cv2.imshow('image',img2)
############################################

############################################
## Close and exit
cv2.waitKey(0)
cv2.destroyAllWindows()
############################################
