import cv2
import numpy as np

def bfmatch(des1, des2):
    matches = []
    for i, desc in enumerate(des1):
        matches2=[]
        distances = []
        for j, desc2 in enumerate(des2):
            distance = np.count_nonzero(desc != desc2)
            distances.append(distance)
        sortiraniidx = np.argsort(distances)
        
        for j in range(2):
            matches2.append(cv2.DMatch(i, sortiraniidx[j], distances[sortiraniidx[j]]))
        matches.append(matches2)  
    return matches

def filter(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good

#ucitaj i nadji
Qimg = cv2.imread('dateSlike/pano2.jpg')
Timg = cv2.imread('dateSlike/pano3.jpg')

orb = cv2.ORB_create(nfeatures=1000)

kpQimg, desQimg = orb.detectAndCompute(Qimg, None)
kpTimg, desTimg = orb.detectAndCompute(Timg, None)

#match
matches = bfmatch(desQimg, desTimg)

#filter
good = filter(matches)

#crtaj
rez = cv2.drawMatchesKnn(Qimg, kpQimg, Timg, kpTimg, good, None, flags=2)
rezres = cv2.resize(rez, (960, 540))
cv2.imshow('Rezultat', rezres)
cv2.waitKey(0)