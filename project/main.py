import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image as keras_img
from keras.backend import tensorflow_backend as tf_backend
from PIL import Image, ImageOps
import helpers.config as config
from keras.applications import VGG16


def load_and_crop_image(img_path, width, height):
    image = Image.open(img_path)
    pl_image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    # print(keras_img.img_to_array(pl_image).shape)
    # pl_image.show()
    # print("")
    img_arr = keras_img.img_to_array(pl_image)
    return (img_arr - img_arr.mean())/255


_IMG_SIZE = 100
_BATCH_SIZE = 16
_EPOCHS = 10

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)
tf_backend.set_session(sess)

_training_image_loc = config.get_train_image_loc()
_training_label_loc = config.get_train_labels_loc()

labels = pd.read_csv(_training_label_loc, index_col=0).values
dog_breeds, labels = np.unique(labels, return_inverse=True)
no_of_breeds = len(dog_breeds)
labels = labels.reshape(-1, 1)
encoder = OneHotEncoder()
labels = encoder.fit_transform(labels)

# pl_image = keras_img.load_img(os.path.join(_training_image_loc, "00a338a92e4e7bf543340dc849230e75.jpg"),
#                               target_size=(254, 254), interpolation="box")

list_of_imgs_paths = [os.path.join(_training_image_loc, x) for x in os.listdir(_training_image_loc)
                      if x.endswith(".jpg")]
print(len(list_of_imgs_paths))
list_of_imgs = np.array([load_and_crop_image(x, _IMG_SIZE, _IMG_SIZE) for x in list_of_imgs_paths])
# list_of_imgs = (list_of_imgs - list_of_imgs.mean())/255
X_train, X_test, y_train, y_test = train_test_split(list_of_imgs, labels, test_size=0.1)
# print(y_train[0])
print(len(X_train))
print(y_train.shape)

model = Sequential()
model.add(Conv2D(input_shape=(_IMG_SIZE, _IMG_SIZE, 3), filters=64, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=no_of_breeds, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=_BATCH_SIZE, epochs=_EPOCHS, validation_data=(X_test, y_test))
