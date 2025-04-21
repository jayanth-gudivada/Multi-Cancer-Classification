import cv2
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def classify(image, model, class_names):
    # Ensure the image is a PIL Image object, then convert to numpy array
    if isinstance(image, Image.Image):
        image = img_to_array(image)

    # Convert the numpy array to uint8 type if necessary
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Resize the image
    img = cv2.resize(image, (150, 150))

    # Ensure the image is in the correct shape for the model
    img = img.reshape(1, 150, 150, 3)

    # Predict the class
    prediction = model.predict(img)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score