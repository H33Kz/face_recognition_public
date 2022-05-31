# Face recognition app
Face recognition app created in python with opencv library.

### app.py
  This is the main file with which you start the recognition. It opens a window in which you can see image from your deafult camera. In the first step app detects all faces present in the frame using standard model present in opencv library. Next step is to recognize it with LBPH(Local Binary Pattern Histogram) method with usage of model created with **faces_train.py**. When face is recognized app draws rectangle over detected face with information underneath: X and Y location, name of the person recognized(If person is not in a created model app displays "Unknown" string).

### faces_train.py
  File made to create models for face recogniition. In order to create model you have to create "images" folder in which you place folder with name thats name is the name of person you  want to have in your model. In this folder you place photos of a person(Photos have to contain only the person to model). This script creates two files: "labels.pkl" and "trainer.yml". First one is just a file to contain labels for each face present in a model(Needed beacause model contains only numerical values for each face). Second one is a model you created.

### app_as_func.py
  Main file modified to work as a separate function that returns the name of person recognized. Uses the same models and labels.

### still_image_recognition.py
  Main file modified to recognize people in still image. Uses the same models and labels.
