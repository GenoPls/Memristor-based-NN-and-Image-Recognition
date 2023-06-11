import os
import cv2
import numpy as np
from NNv2 import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(cat_folder, dog_folder, num_samples, img_size):
    imgs = []
    labels = []

    for file in os.listdir(cat_folder)[:num_samples]:
        img_path = os.path.join(cat_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img = cv2.resize(img, (img_size, img_size))
        imgs.append(img)

        labels.append(0)

    for file in os.listdir(dog_folder)[:num_samples]:
        img_path = os.path.join(dog_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img = cv2.resize(img, (img_size, img_size))
        imgs.append(img)

        labels.append(1)

    # Normalize pixel values to between 0 and 1
    imgs = np.array(imgs) / 255.0
    labels = np.array(labels)

    return imgs, labels

cat_folder = 'D:/ImageRecognition/PetImages/Cat'
dog_folder = 'D:/ImageRecognition/PetImages/Dog'
num_samples = 12500
img_size = 64

imgs, labels = load_data(cat_folder, dog_folder, num_samples, img_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)

model = build_model()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

model.save('D:/ImageRecognition/ImageRecognition_model.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

