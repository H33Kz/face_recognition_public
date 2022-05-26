import easygui
import cv2 as cv
import pickle

#Load cascade classifiers for faces from opencv data
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'./haarcascade_frontalface_alt2.xml')

#Load trained recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

#Import label names from pickle file
labels = {}
with open("lablel.pkl",'rb') as file:
    labels = pickle.load(file)
    labels = {val:key for key,val in labels.items()}

pth = easygui.fileopenbox()

image = cv.imread(pth)
grayscale = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grayscale,1.2,4)

for (x,y,w,h) in faces:
    #Take cord of the center of the face
    centerCordX = int(x+(w/2))
    centerCordY = int(y+(h/2))

    #Recognizer prediction
    id,conf = recognizer.predict(grayscale[y:y+h,x:x+w])
    print(str(labels[id]) + '  ' + str(conf))

    #Draw rectangle and middlepoint on a face
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.circle(image,(centerCordX ,centerCordY), 2, (255, 0, 0),-1)

    #Create and display string made of center cords and recognized person
    cords = '('+ str(centerCordX) +','+ str(centerCordY)+')'
    cv.putText(image,cords,(x,y+h+30),cv.QT_FONT_NORMAL,0.7,(0, 255, 0),1)
    if conf<90:
        cv.putText(image,str(labels[id]),(x,y+h+60),cv.QT_FONT_NORMAL,0.7,(0, 255, 0),1)
    else:
        cv.putText(image,'Unknown',(x,y+h+60),cv.QT_FONT_NORMAL,0.7,(0, 0, 255),1)

cv.imshow('img',image)


cv.waitKey(0)
cv.destroyAllWindows()