# Face recognition app
Face recognition app created in python with opencv library.

### app.py
This is the main file with which you start the recognition. It opens a window in which you can see image from your deafult camera. In the first step app detects all faces present in the frame using standard model present in opencv library. Next step is to recognize it with LBPH(Local Binary Pattern Histogram) method with usage of model created with **faces_train.py**. When face is recognized app draws rectangle over detected face with information underneath: X and Y location, name of the person recognized(If person is not in a created model app displays "Unknown" string).

### faces_train.py
