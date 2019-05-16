import cv2
import numpy as np
 
img1 = cv2.imread("cola_real2.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("cola_web.jpg", cv2.IMREAD_GRAYSCALE)
 
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect the CenSurE key points
star = cv2.xfeatures2d.StarDetector_create()
keyPoints1 = star.detect(img1, None)
keyPoints2 = star.detect(img2, None)
# Create the BRIEF extractor and compute the descriptors
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp1, des1 =  brief.compute(img1, keyPoints1)
kp2, des2 =  brief.compute(img2, keyPoints2)

 #for surf and sift
# norm = cv2.NORM_L1
# for orb
# norm = cv2.NORM_HAMMING
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2,k=2)
# matches = sorted(matches, key = lambda x:x.distance)

# distances = list(map((lambda x: x.distance), matches))

# print(distances)
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append([m])
print(len(good))
matching_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
