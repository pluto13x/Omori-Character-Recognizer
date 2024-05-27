import cv2
import numpy as np
import os

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

def provera(ime):  
    ima = False
    for i in imena:
        if ime == i:
            ima = True
            break
    if ima == False:
        print('Taj lik ne postoji, pokušaj ponovo ')
        return False
    else:
        return True

#ucitaj imena
imena = []
lista = os.listdir('ImagesQuery')

for cl in lista:
    imena.append(os.path.splitext(cl)[0])

#pitaj

while True:
    prva = input('Sliku kog lika želiš da uporediš? ')
    if (provera(prva)):
        break

while True:
    druga = input('Drugu sliku kog lika želiš da uporediš? ')
    if (provera(druga)):
        break

#ucitaj i nadji
Qimg = cv2.imread(f'imagesQuery/{prva}.png')
Timg = cv2.imread(f'imagesTrain/t{druga}.png')

orb = cv2.ORB_create(nfeatures=1000)

kpQimg, desQimg = orb.detectAndCompute(Qimg, None)
kpTimg, desTimg = orb.detectAndCompute(Timg, None)

#match
matches = bfmatch(desQimg, desTimg)

#filter
good = filter(matches)

#crtaj
rez = cv2.drawMatchesKnn(Qimg, kpQimg, Timg, kpTimg, good, None, flags=2)
cv2.imshow('Rezultat', rez)
cv2.waitKey(0)