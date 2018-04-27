import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
import os
import helpers.config as config
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
# model = xception.Xception(weights="imagenet")


INPUT_SIZE = 224
SEED = 1987
NUM_CLASSES = 16

_training_image_loc = config.get_train_image_loc()
_training_label_loc = config.get_train_labels_loc()
_testing_image_loc = config.get_test_image_loc()
_sample_sub_loc = config.get_sample_submission_loc()

labels = pd.read_csv(_training_label_loc)
sample_submission = pd.read_csv(_sample_sub_loc)
print(len(os.listdir(_training_image_loc)), len(labels))
print(len(os.listdir(_testing_image_loc)), len(sample_submission))

selected_breed_list = list(labels.groupby("breed").count().sort_values(by="id", ascending=False)
                           .head(NUM_CLASSES).index)
labels = labels[labels["breed"].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
ytr = y_train[train_idx]
yv = y_train[valid_idx]


def read_img(img_id, training, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    if training == 'train':
        file_path = _training_image_loc
    else:
        file_path = _testing_image_loc
    img = image.load_img(os.path.join(file_path, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img


model = ResNet50(weights="imagenet")
j = int(np.sqrt(NUM_CLASSES))
i = int(np.ceil(1. * NUM_CLASSES/j))
fig = plt.figure(1, figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)

# for i, (img_id, breed) in enumerate(labels.loc[labels["rank"] == 1, ["id", "breed"]].values):
#     ax = grid[i]
#     img = read_img(img_id, "train", (224, 224))
#     ax.imshow(img/255.)
#     x = preprocess_input(np.expand_dims(img.copy(), axis=0))
#     preds = model.predict(x)
#     _, imagenet_class_name, prob = decode_predictions(preds, top=1)[0][0]
#     ax.text(10, 180, "ResNet50: %s (%.2f)" % (imagenet_class_name, prob), color="w", backgroundcolor="k", alpha=0.8)
#     ax.text(10, 200, "LABEL: %s" % breed, color="k", backgroundcolor="w", alpha=0.8)
#     ax.axis("off")
# plt.show()

INPUT_SIZE = 224
POOLING = "avg"

x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype="float32")
for i, img_id in tqdm(enumerate(labels["id"])):
    img = read_img(img_id, "train", (INPUT_SIZE, INPUT_SIZE))
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x

print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print((Xtr.shape, Xv.shape, ytr.shape, yv.shape))
vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size=4, verbose=1)
valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=4, verbose=1)
print('VGG train bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))
print('VGG valid bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)
logreg.fit(train_vgg_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_vgg_bf)
valid_preds = logreg.predict(valid_vgg_bf)
print('Validation VGG LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation VGG Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))


# Xception
INPUT_SIZE = 299
POOLING = "avg"

x_train = np.zeros((len(labels), INPUT_SIZE, INPUT_SIZE, 3), dtype="float32")
for i, img_id in tqdm(enumerate(labels["id"])):
    img = read_img(img_id, "train", (INPUT_SIZE, INPUT_SIZE))
    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_train[i] = x
print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

Xtr = x_train[train_idx]
Xv = x_train[valid_idx]
print(Xtr.shape, Xv.shape, ytr.shape, yv.shape)

xception_bottleneck = xception.Xception(weights="imagenet", include_top=False, pooling=POOLING)
train_x_bf = xception_bottleneck.predict(Xtr, batch_size=8, verbose=1)
valid_x_bf = xception_bottleneck.predict(Xv, batch_size=8, verbose=1)
print("Xception train bottleneck feature shape: {} size: {:,}"
      .format(train_x_bf.shape, train_x_bf.size))

print("Xception validation bottleneck feature shape: {} size: {:,}"
      .format(valid_x_bf.shape, valid_x_bf.size))

logreg = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=SEED)
logreg.fit(train_x_bf, (ytr * range(NUM_CLASSES)).sum(axis=1))
valid_probs = logreg.predict_proba(valid_x_bf)
valid_preds = logreg.predict(valid_x_bf)
print('Validation Xception LogLoss {}'.format(log_loss(yv, valid_probs)))
print('Validation Xception Accuracy {}'.format(accuracy_score((yv * range(NUM_CLASSES)).sum(axis=1), valid_preds)))


