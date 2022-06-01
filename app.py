import cv2 as cv
import pickle



def main():
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

    #Setup for capturing webcams video
    cap = cv.VideoCapture(0)

    while True:
        #Capture a frame into img variable
        succ, img = cap.read()

        #Grayscale a frame(Needed for face detection)
        grayscale = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        #Detect faces present in the frame and load their parameters into variable
        faces = face_cascade.detectMultiScale(grayscale,1.2,4)

        #Go trough all parameters and add marks into original(colored) frame
        for (x,y,w,h) in faces:
            #Print Cords of upper left corner(Starting cords for face detection) + width and height of detected face(All in pixels)
            #print(x,y,w,h)

            #Take cord of the center of the face
            centerCordX = int(x+(w/2))
            centerCordY = int(y+(h/2))

            #Recognizer prediction
            id,conf = recognizer.predict(grayscale[y:y+h,x:x+w])
            print(str(labels[id]) + '  ' + str(conf))

            #Draw rectangle and middlepoint on a face
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.circle(img,(centerCordX ,centerCordY), 2, (255, 0, 0),-1)

            #Create and display string made of center cords and recognized person
            cords = '('+ str(centerCordX) +','+ str(centerCordY)+')'
            cv.putText(img,cords,(x,y+h+30),cv.QT_FONT_NORMAL,0.7,(0, 255, 0),1)
            if conf<90:
                cv.putText(img,str(labels[id]),(x,y+h+60),cv.QT_FONT_NORMAL,0.7,(0, 255, 0),1)
            else:
                cv.putText(img,'Unknown',(x,y+h+60),cv.QT_FONT_NORMAL,0.7,(0, 0, 255),1)
            
            
        #Display image in a window
        cv.imshow('Camera',img)

        #ESC key detection
        k = cv.waitKey(30) & 0xff
        if k==27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()