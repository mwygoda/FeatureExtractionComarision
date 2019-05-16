import cv2
import numpy as np
 
img1 = cv2.imread("cola_real2.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("cola_web.jpg", cv2.IMREAD_GRAYSCALE)
 
orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

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
