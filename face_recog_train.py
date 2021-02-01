import os
import cv2 as cv
import numpy as np

personas = ['george_michael', 'larry_david', 'yann_lecun']
directorio = r'.Photos/FaceRecognition/Train'

haarCascade = cv.CascadeClassifier("haar_face.xml")

features = []
labels = []

def crear_train():
    for persona in personas:
        path = os.path.join(directorio, persona)
        label = personas.index(persona)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_arr = cv.imread(img_path)
            gris = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            detect_cara = haarCascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=4)
            for (x,y,w,h) in detect_cara:
                caras_region = gris[y:y+h, x:x+w]
                features.append(caras_region)
                labels.append(label)


crear_train()

features = np.array(features, dtype='object')
labels = np.array(labels)
reconocimiento_caras = cv.face.LBPHFaceRecognizer_create()
reconocimiento_caras.train(features, labels)

reconocimiento_caras.save('caras_entrenadas.yml')
np.save('features.npy', features)
np.save('labels.npy',labels)