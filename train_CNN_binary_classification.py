# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt

gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, zoom_range=(0.99, 0.99), dtype=np.float32)

train = gen.flow_from_directory("./Brain Tumor Data Set/",
                                target_size=(150, 150),
                                batch_size=256,
                                class_mode="binary",
                                color_mode="rgb",
                                shuffle=True,
                                seed=123,
                                subset="training")

val = gen.flow_from_directory("./Brain Tumor Data Set/",
                              target_size=(150, 150),
                              batch_size=8,
                              class_mode="binary",
                              color_mode="rgb",
                              shuffle=True,
                              seed=123,
                              subset="validation")
classes = val.class_indices

# t = 0
# h = 0
# for i in range(15):
#     a, b = next(train)
#     for j in b:
#         if j == 1:
#             h += 1
#         else:
#             t += 1

# sns.barplot(x=['tumor', 'healty'], y=[t, h])

batch = next(train)

# plt.imshow(batch[0][0])

from keras.layers import Conv2D, MaxPool2D, LeakyReLU, BatchNormalization, Dropout, Dense, InputLayer, Flatten
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

model = keras.Sequential()
model.add(InputLayer(input_shape=(150, 150, 3)))
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPool2D())
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPool2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.3))
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(rate=0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True,
    show_layer_names=True,
)

from keras import utils, callbacks

earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                        patience=5, restore_best_weights=True)

history = model.fit(train, verbose=1, callbacks=[earlystopping], epochs=10, validation_data=(val))
# 保存模型
model.save('model.h5')  # 保存为 HDF5 格式
from keras.models import load_model

model = load_model('model.h5')



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

