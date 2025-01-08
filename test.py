from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
# from ultralytics import YOLO
model = load_model('model-4.h5')

model.summary()
img = image.load_img('brain-tumor-mri-dataset/Training/pituitary/Tr-pi_0234.jpg', target_size=(256, 256))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

predictions = model.predict(img)
predicted_class = np.argmax(predictions)


print(f"Predicted Class: {predicted_class}")
