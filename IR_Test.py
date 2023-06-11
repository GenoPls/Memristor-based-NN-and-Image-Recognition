import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    if prediction < 0.5:
        print("This animal is a cat.")
    else:
        print("This animal is a dog.")

model_path = 'D:/ImageRecognition/ImageRecognition_model.h5'
image_path = 'D:/ImageRecognition/PetImages/TestImage/test1/51.jpg'
model = load_model(model_path)

predict_image(image_path, model)
