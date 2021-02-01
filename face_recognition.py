import numpy as np
import cv2 as cv

personas = ['george_michael', 'larry_david', 'yann_lecun']
haarCascade = cv.CascadeClassifier("haar_face.xml")

recog_cara = cv.face.LBPHFaceRecognizer_create()
recog_cara.read('caras_entrenadas.yml')

img = cv.imread('Photos/FaceRecognition/val/gm2.jpg')

gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cara_rec = haarCascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=4)

for(x,y,w,h) in cara_rec:
    cara_region = gris[y:y+h, x:x+w]
    label, confidence = recog_cara.predict(cara_region)
    print(f'Label = {label} con confianza de {confidence}')
    cv.putText(img, str(personas[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, fontScale=1.2, color= (0,255,0), thickness= 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Cara detectada', img)

cv.waitKey(0)
