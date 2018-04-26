import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import cv2
import random
import os

# returns the confusion matrix for output prediction output from network
# each row is assumed to be a datapoint
def gen_confusion_mat(y_real, y_pred, pprint=True):
    y_pred_indices = []
    y_real_indices = []
    for arr in range(len(y_pred)):
        y_pred_indices.append(y_pred.argmax())
        y_real_indices.append(y_real.argmax())
    confusion_mat = confusion_matrix(y_real_indices, y_pred_indices)
    if pprint: print(confusion_mat)
    return confusion_mat

# returns a flat vector instead of an image matrix. Also converts image to gray
def flatten_img(img):
    gray = cv2.rgb2gray(img)
    return gray.flatten()

# TODO: add flags
flags = tf.app.flags

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('Input shape: ', x_train.shape)
num_classes = 10
y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)
# basic preprocessing
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train/255, x_test/255

# the parameters
BATCH = 800
EPOCHS = 10
LEARN = 0.001 # learning rate

# the network (feedforward without convolutional layers)
model = Sequential()
model.add(Dense(100, input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# compile network
model.compile(optimizer=RMSprop(lr=LEARN),
                loss=categorical_crossentropy,
                metrics=['accuracy'])

print(model.summary())

# split data (this is where most work is) and run model
# A. FULLY RANDOM
#zipped = list(zip(x_train, y_train))
#random.shuffle(zipped)
#x_train, y_train = zip(*zipped)

# run
model.fit(x_train, y_train,
            batch_size=BATCH,
            epochs=EPOCHS,
            validation_split=0.2,
            shuffle=True)

predictions =  model.predict(x_test)
gen_confusion_mat(y_test, predictions)
