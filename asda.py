import cv2
import numpy as np
import time

img = cv2.imread("cola_web.jpg", cv2.IMREAD_GRAYSCALE)
 
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
#  brief = cv2.DescriptorExtractor_create("BRIEF")

orb = cv2.ORB_create(nfeatures=1500)
start = time.time()
keypoints, descriptors = orb.detectAndCompute(img, None)
stop = time.time()
 
print("time", stop - start)
print("Count:", len(keypoints))

# img = cv2.drawKeypoints(img, keypoints, None)
 
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()