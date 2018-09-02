import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.initializers import he_normal
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model
from keras import backend as K
from keras import regularizers
import tensorflow as tf

# set seed
np.random.seed(42)
tf.set_random_seed(42)

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("x_train:", X_train.shape, "y_train:", Y_train.shape,
      "x_test:", X_test.shape, "y_test:", Y_test.shape)

# re-format data, train consists of 0-4 and test of 5-9
x_train = np.array([X_train[key]
                    for (key, label) in enumerate(Y_train) if int(label) < 5])
y_train = np.array(Y_train[Y_train < 5])
x_test = np.array([X_train[key]
                   for (key, label) in enumerate(Y_train) if int(label) > 4])
y_test = np.array(Y_train[Y_train > 4])
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_test = pd.DataFrame(y_test)
y_test = pd.get_dummies(y_test.loc[:, 0]).values
y_train = pd.DataFrame(y_train)
y_train = pd.get_dummies(y_train.loc[:, 0]).values
print("x_train:", x_train.shape, "y_train:", y_train.shape,
      "x_test:", x_test.shape, "y_test:", y_test.shape)

# take a sample of 100 for each of the test set values, i.e. 5, 6, 7, 8, and 9
x_test_5 = np.array([X_train[key] for (key, label)
                     in enumerate(Y_train) if int(label) == 5][:100])
x_test_6 = np.array([X_train[key] for (key, label)
                     in enumerate(Y_train) if int(label) == 6][:100])
x_test_7 = np.array([X_train[key] for (key, label)
                     in enumerate(Y_train) if int(label) == 7][:100])
x_test_8 = np.array([X_train[key] for (key, label)
                     in enumerate(Y_train) if int(label) == 8][:100])
x_test_9 = np.array([X_train[key] for (key, label)
                     in enumerate(Y_train) if int(label) == 9][:100])
x_test_100 = np.concatenate(
    (x_test_5, x_test_6, x_test_7, x_test_8, x_test_9), axis=0)
y_test_5 = np.array(Y_train[Y_train == 5][:100])
y_test_6 = np.array(Y_train[Y_train == 6][:100])
y_test_7 = np.array(Y_train[Y_train == 7][:100])
y_test_8 = np.array(Y_train[Y_train == 8][:100])
y_test_9 = np.array(Y_train[Y_train == 9][:100])
y_test_100 = np.concatenate(
    (y_test_5, y_test_6, y_test_7, y_test_8, y_test_9), axis=0)
y_test_100 = pd.DataFrame(y_test_100)
y_test_100 = pd.get_dummies(y_test_100.loc[:, 0]).values
print("sampling 100 of each from x_test:", x_test_100.shape,
      "/n" "sampling 100 of each from y_test:", y_test_100.shape)

# flatten train and test sets to 2 dimensions
x_train = x_train.reshape((30596, 784))
x_test = x_test.reshape((29404, 784))
x_test_100 = x_test_100.reshape((500, 784))

# clear session and set CPU equal to 1 (comment out if running on GPU)
K.clear_session()
config = tf.ConfigProto(intra_op_parallelism_threads=1
	, inter_op_parallelism_threads=1
	, allow_soft_placement=True
	, device_count={'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)

# run train model
m = Sequential([
    Dense(100, input_shape=(784,), kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer='he_normal'),
    Dropout(0.2),
    BatchNormalization(),
    Activation('elu'),
    Dense(100, kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer='he_normal'),
    Dropout(0.5),
    BatchNormalization(),
    Activation('elu'),
    Dense(100, kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer='he_normal'),
    Dropout(0.5),
    BatchNormalization(),
    Activation('elu'),
    Dense(100, kernel_regularizer=regularizers.l2(
        0.01), kernel_initializer='he_normal'),
    Dropout(0.5),
    BatchNormalization(),
    Activation('elu'),
    Dense(5),
    BatchNormalization(),
    Activation('softmax')
])


m.compile(loss='categorical_crossentropy',
          optimizer='adam', metrics=['accuracy'])

# use early stopping (stops when validation loss > training loss)
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5'
             	, monitor='val_loss'
             	, save_best_only=True)]

history = m.fit(x_train, y_train, epochs=100, callbacks=callbacks,
                batch_size=128, validation_split=0.33)

# summarize history for loss
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])
ax[0].set_title('model loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'validation'], loc='upper right')
ax[1].plot(history.history['acc'])
ax[1].plot(history.history['val_acc'])
ax[1].set_title('model accuracy')
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'validation'], loc='upper left')
plt.show()

# Freeze all hidden layers and replace the softmax output layer with a new one.
for layer in m.layers[:12]:
    layer.trainable = False

# Add top layers on top of frozen (not re-trained) layers of previous model
x = m.output
x = Dense(100, activation="elu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
new_layer = Dense(5, activation="softmax")(x)
m2 = Model(input=m.input, output=new_layer)

# compile and run new (2nd) model
m2.compile(loss="categorical_crossentropy",
           optimizer='adam', metrics=["accuracy"])
history2 = m2.fit(x_test_100, y_test_100, epochs=100,
                  callbacks=callbacks, batch_size=128, validation_split=0.33)

# see score on test set
score_m2 = m2.evaluate(x_test_100, y_test_100, batch_size=128)
print("Score on sampled test set: ", score_m2[1])

# Try freezing fewer hidden layers and replace them with new ones.
for layer in m.layers[9:12]:
    layer.trainable = True

# Add more layers to the frozen (not re-trained) layers of previous model
x = m.output
x = Dense(100, activation="elu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(100, activation="elu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
new_layer = Dense(5, activation="softmax")(x)
m3 = Model(input=m.input, output=new_layer)

# compile and run new (3rd) model
m3.compile(loss="categorical_crossentropy",
           optimizer='adam', metrics=["accuracy"])
history3 = m3.fit(x_test_100, y_test_100, epochs=100,
                  callbacks=callbacks, batch_size=128, validation_split=0.33)

# see score on test set
score_m3 = m3.evaluate(x_test_100, y_test_100, batch_size=128)
# stuck in a local minima (TODO: remove earlystopping)
print("Score on sampled test set: ", score_m3[1])
