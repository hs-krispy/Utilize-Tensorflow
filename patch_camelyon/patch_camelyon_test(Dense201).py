import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import Sequential

# print(device_lib.list_local_devices())
import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization, ReLU, Concatenate, AvgPool2D, \
    Input, GlobalAvgPool2D, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet201

df, info = tfds.load('patch_camelyon', shuffle_files=True, with_info=True, as_supervised=True)

np.random.seed(100)
tf.random.set_seed(100)
AUTOTUNE = tf.data.experimental.AUTOTUNE
buffer_size = 1000
batch_size = 64
epochs = 100
rows = 96
cols = 96
channel = 3

def preprocess(image, labels):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, labels


train_data = df['train'].map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(batch_size)
valid_data = df['validation'].map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(batch_size)
test_data = df['test'].map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(batch_size)

class DenseNET():
    def __init__(self, train_data, valid_data, test_data, row, col, channel):
        self.rows = row
        self.cols = col
        self.channel = channel
        self.input_shape = (self.rows, self.cols, self.channel)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        # 5 dense net
        self.repetitions = 5

    def train(self):
        model = Sequential()
        model.add(DenseNet201(weights="imagenet", include_top=False, input_shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        plot_model(model)
        model.summary()
        model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=['accuracy'])
        history = model.fit(self.train_data, epochs=100, verbose=1, validation_data=self.valid_data)

        return model, history

    def plot(self, history):
        plt.title('model accuracy')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        plt.show()

    def predict(self, model):
        print('\nTest accuracy: {:.4f}'.format(model.evaluate_generator(self.test_data)[1]))


D = DenseNET(train_data, valid_data, test_data, rows, cols, channel)
time = []
time.append(datetime.datetime.now())
model, history = D.train()
time.append(datetime.datetime.now())
print(time)
D.predict(model)
D.plot(history)

# 17시간 30분 소요
# Test accuracy: 0.7353
