import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

import helpers.config as config


_training_image_loc = config.get_train_image_loc()
_training_label_loc = config.get_train_labels_loc()
_testing_image_loc = config.get_test_image_loc()
_sample_sub_loc = config.get_sample_submission_loc()

df_train = pd.read_csv(_training_label_loc)
df_test = pd.read_csv(_sample_sub_loc)

targets_series = pd.Series(df_train["breed"])
one_hot = pd.get_dummies(targets_series, sparse=True)

one_hot_labels = np.asarray(one_hot)

im_size = 150
batch_size = 32

x_train = []
y_train = []
# x_test = []


i = 0
for f, breed in tqdm(df_train.values):
    img = cv2.imread(os.path.join(_training_image_loc, "{}.jpg".format(f)))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

x_train_raw = np.array(x_train, np.float32)/255.
x_train = None
y_train_raw = np.array(y_train, np.uint8)

# for f in tqdm(df_test["id"].values):
#     img = cv2.imread(os.path.join(_testing_image_loc, "{}.jpg".format(f)))
#     x_test.append(cv2.resize(img, (im_size, im_size)))
# x_test = np.array(x_test, np.float32)/255.

# print(x_train_raw.shape)
# print(y_train_raw.shape)
# print(x_test.shape)

num_class = y_train_raw.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(2048, activation="relu")(x)
predictions = Dense(num_class, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
callbacks_list = [keras.callbacks.EarlyStopping(monitor="val_acc", patience=3, verbose=1)]
model.summary()

model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid), verbose=1, batch_size=batch_size)
