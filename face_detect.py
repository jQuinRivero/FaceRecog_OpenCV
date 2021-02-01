import cv2 as cv

losCaraChicas = cv.imread("Photos/losCaraChicas.jpg")
#cv.imshow("Pibe", pibe)

pibe_gris = cv.cvtColor(losCaraChicas, cv.COLOR_BGR2GRAY)
#cv.imshow("Pibe Gris", pibe_gris)

haarCascade = cv.CascadeClassifier("haar_face.xml")

recog_cara = haarCascade.detectMultiScale(pibe_gris, scaleFactor=1.1, minNeighbors=6)

print(len(recog_cara))

for (x,y,w,h) in recog_cara:
    cv.rectangle(losCaraChicas, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Cara detectada", losCaraChicas)
cv.waitKey(0)