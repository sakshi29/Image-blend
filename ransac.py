
import numpy as np
import cv2
class perspective:
    
    def __init__(self,image1,image2):
        self.img1=image1
        self.img2=image2
        
    def pers(self,x12):    
        MIN_MATCH_COUNT = 10
        ##img1 = cv2.imread('A0001 (4).jpg')
        ##img2 = cv2.imread('testsa.jpg')
        sift = cv2.SIFT()
        kp1, des1 = sift.detectAndCompute(self.img1,None)
        kp2, des2 = sift.detectAndCompute(self.img2,None)
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
            h,w,c= self.img2.shape
            
            x,y,c=self.img1.shape
            x2,y2,c2=self.img2.shape
            print h,w
            print x,y
            print x2,y2
            
            
            self.img2=  cv2.warpPerspective(self.img2,M,(y+y2+10,h))
            l,b,k=self.img2.shape
            print l,b
            
           
            out = np.zeros((max([x,x2]),y+y2,3), dtype='uint8')
            ##out[:x2,y:] = np.dstack([gr, gray2, gray2])

            
            for i in range(0,y-1):
                for j in range(0,x-1):
                     out[j,i]=self.img1[j,i]
              
                    
            for i in range (0,x12):
                for j in range (0,x2-1):
                   
                  
                     out[j,i][0]=out[j,i][0]*0.90+self.img2[j,i][0]*0.10
                     out[j,i][1]=out[j,i][1]*0.90+self.img2[j,i][1]*0.10
                     out[j,i][2]=out[j,i][2]*0.90+self.img2[j,i][2]*0.10

            for i in range (x12+1,y2-1):
                for j in range (0,x2-1):
               
                  
                      out[j,i][0]=out[j,i][0]*0.30+self.img2[j,i][0]*0.70
                      out[j,i][1]=out[j,i][0]*0.30+self.img2[j,i][1]*0.70
                      out[j,i][2]=out[j,i][0]*0.30+self.img2[j,i][2]*0.70
                

                
            
            """
            for i in range (0,y-1):
                for j in range (0,x2-1):
                  
                     img2[j,i][0]=img2[j,i][0]*0.10+img1[j,i][0]*0.90
                     img2[j,i][1]=img2[j,i][1]*0.10+img1[j,i][1]*0.90
                     img2[j,i][2]=img2[j,i][2]*0.10+img1[j,i][2]*0.90
                  
            """   
        crop_img = out[0:550, 0:850]    
        cv2.imshow('img1',crop_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        ##cv2.imshow('result',img2)
        ##cv2.imshow('re',img1)

