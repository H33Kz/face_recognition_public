import os
import pickle
import cv2 as cv
from PIL import Image
import numpy as np

#Create recognizer object to train
recognizer = cv.face.LBPHFaceRecognizer_create()

#Load cascade classifier for faces from opencv data
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'./haarcascade_frontalface_alt2.xml')

#Crate a constant out of this python file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Create directory out of BASE_DIR by adding name of the folder that contains images
image_dir = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids = {}
y_labels = []
x_train =[]

#Go trough given directory and its parameters
for root, dirs, files in os.walk(image_dir):
    #Go trough all files in directory
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            #Save path to variable and create label based on dir name
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]


            #Open image with PILLOW library
            pil_image = Image.open(path).convert("L") #Load image and grayscale it
            image_array = np.array(pil_image, "uint8") #Turn image into numpy array


            faces = face_cascade.detectMultiScale(image_array,1.2,4) #Detect faces in a given image
            #Parse region of interest and put their array values to x_train list and id to label list
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)


#Export labels and their ids as pickle file for identification in face recognition app
with open("lablel.pkl",'wb') as file:
    pickle.dump(label_ids,file)

#Train recognizer with data from x_train list - data is signed with label's id
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')