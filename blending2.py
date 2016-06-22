import numpy as np
import cv2


img1 = cv2.imread('bus1.jpg')
img2 = cv2.imread('bus2.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
x,y,z=img1.shape
x2,y2,z2=img2.shape
out = np.zeros((max([x,x2]),y+y2,3), dtype='uint8')
for i in range(0,y-1):
    for j in range(0,x-1):
             out[j,i]=img1[j,i]
k=500              
for i in range (0,y2-1):
    for j in range (0,x2-1):
          if(k+i<y):
             out[j,k+i][0]=img2[j,i][0]*0.33+img1[j,k+i][0]*0.67
             out[j,k+i][1]=img2[j,i][1]*0.33+img1[j,k+i][1]*0.67
             out[j,k+i][2]=img2[j,i][2]*0.33+img1[j,k+i][2]*0.67
          else:
             out[j,k+i][0]=img2[j,i][0]*0.98
             out[j,k+i][1]=img2[j,i][1]*0.98
             out[j,k+i][2]=img2[j,i][2]*0.98
cv2.imshow('img',out)    
cv2.waitKey(0)
cv2.destroyAllWindows()
