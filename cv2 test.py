import cv2
import numpy as np

def filter(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good

slike = []
imena = []

aubrey = cv2.imread('imagesQuery/aubrey.png', 0)
taubrey = cv2.imread('imagesTrain/tAubrey.png', 0)

orb = cv2.ORB_create(nfeatures = 1000)

kpAubrey, desAubrey = orb.detectAndCompute(aubrey, None)
kpTaubrey, desTaubrey = orb.detectAndCompute(taubrey, None)
#imgKpAubrey = cv2.drawKeypoints(aubrey, kpAubrey, None)
#imgKpTaubrey = cv2.drawKeypoints(taubrey, kpTaubrey, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desAubrey, desTaubrey, k = 2)



good = filter(matches)

img3 = cv2.drawMatchesKnn(aubrey, kpAubrey, taubrey, kpTaubrey, good, None, flags=2)

cv2.imshow('rez', img3)
cv2.waitKey(0)