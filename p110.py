import cv2 
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(0)
myModel = tf.keras.models.load_model("keras_model.h5")

while True:
    status, frame = camera.read()
    if status:
        frame = cv2.flip(frame, 1)
        resizeFrame = cv2.resize(frame, (224,224))
        resizeFrame = np.expand_dims(resizeFrame, axis = 0)
        resizeFrame = resizeFrame/255
        prediction = myModel.predict(resizeFrame)
        rock = int(prediction[0][0]*100)
        paper = int(prediction[0][1]*100)
        scissor = int(prediction[0][2]*100)
        print(f"rock: {rock} %, paper: {paper} %, scissors: {scissor}% ")
        cv2.imshow("feed", frame)
        code = cv2.waitKey(1)
        if code == 32:
            break

camera.release()
cv2.destroyAllWindows()