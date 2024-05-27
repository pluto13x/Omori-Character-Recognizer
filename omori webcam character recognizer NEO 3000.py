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
            good.append(
                [m])
    return good

def findDes(slike):
    desList = []
    for slika in slike:
        kp, des = orb.detectAndCompute(slika, None)
        desList.append(des)
    return desList

def findID(slika, desList):
    kp2, des2 = orb.detectAndCompute(slika, None)
    matchevi = []
    rezidx = -1
    try:
        for des in desList:
            matches = bfmatch(des, des2)
            good = filter(matches)
            matchevi.append(len(good))
    except:
        pass 
    if len(matchevi) != 0:
        if max(matchevi) > 4:
            rezidx = matchevi.index(max(matchevi))
    return rezidx

orb = cv2.ORB_create(nfeatures=500)

slike = []
imena = []
lista = os.listdir('ImagesQuery')

for cl in lista:
    imgCur = cv2.imread(f'ImagesQuery/{cl}', 0)
    slike.append(imgCur)
    imena.append(os.path.splitext(cl)[0])

desList = findDes(slike)

cap = cv2.VideoCapture(0)

while True:
    uspeh, snimak = cap.read()
    snimakboja = snimak.copy()
    slika = cv2.cvtColor(snimak, cv2.COLOR_BGR2GRAY)
    rezidx = findID(snimak, desList)
    if rezidx != -1:
        cv2.putText(snimakboja, imena[rezidx], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (153, 51, 255), 2)
    cv2.imshow('Rezultat', snimakboja)
    
    key = cv2.waitKey(1)
    
    if key == 27 or cv2.getWindowProperty('Rezultat', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
cap.release()
    