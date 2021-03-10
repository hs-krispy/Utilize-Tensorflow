import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


df, info = tfds.load('patch_camelyon', with_info=True, as_supervised=True)

train_data = df['train']
valid_data = df['validation']
test_data = df['test']

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


# Applying the preprocess function we the use of map() method
train_data = train_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size).prefetch(1)
valid_data = valid_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size).prefetch(1)
test_data = test_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(buffer_size).batch(batch_size).prefetch(1)


class CNN():
    def __init__(self, train_data, valid_data, test_data, row, col, channel):
        self.rows = row
        self.cols = col
        self.channel = channel
        self.input_shape = (self.rows, self.cols, self.channel)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def train(self):
        global history
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(3, 3),
                              input_shape=self.input_shape,
                              padding="same", activation="relu", kernel_initializer="he_uniform"))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
        self.model.add(Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                              kernel_initializer="he_uniform"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(
            Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
        self.model.add(Conv2D(64, strides=2, kernel_size=(3, 3), padding='same', activation='relu',
                              kernel_initializer="he_uniform"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu',
                              kernel_initializer="he_uniform"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', kernel_initializer="he_uniform"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, activation='relu', kernel_initializer="he_uniform"))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.summary()
        # callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
        self.model.fit(self.train_data, steps_per_epoch=len(self.train_data), epochs=100, verbose=1,
                       validation_data=self.valid_data)

    def plot(self):
        plt.title('model accuracy')
        plt.plot(self.model.history['accuracy'])
        plt.plot(self.model.history['val_accuracy'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'])
        plt.show()

    def predict(self):
        print('\nTest accuracy: {:.4f}'.format(self.model.evaluate(self.test_data)[1]))


model = CNN(train_data, valid_data, test_data, rows, cols, channel)
time = []
time.append(datetime.datetime.now())
model.train()
time.append(datetime.datetime.now())
print(time)
model.predict()

# 15시간 40분 소요 (GPU 가용 - RTX 2070)
# test accuracy : 0.8335

# model.plot()
