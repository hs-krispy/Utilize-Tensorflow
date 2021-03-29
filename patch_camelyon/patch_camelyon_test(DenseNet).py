import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow_datasets as tfds
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization, ReLU, Concatenate, AvgPool2D, \
    Input, GlobalAvgPool2D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

df, info = tfds.load('patch_camelyon', shuffle_files=True, with_info=True, as_supervised=True)

train_data = df['train']
valid_data = df['validation']
test_data = df['test']
#
# datagen = ImageDataGenerator(
#     preprocessing_function=lambda x: x / 255.,
#     width_shift_range=4,  # randomly shift images horizontally
#     height_shift_range=4,  # randomly shift images vertically
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=True  # randomly flip images
# )

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


# Applying the preprocess function we the use of map() method
train_data = train_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(
    batch_size)
valid_data = valid_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(
    batch_size)
test_data = test_data.map(preprocess, num_parallel_calls=AUTOTUNE).shuffle(batch_size).batch(batch_size).prefetch(
    batch_size)


class pre_activation(Layer):
    def __init__(self, filters=None, kernel_size=None):
        super(pre_activation, self).__init__()

        self.t_BN = BatchNormalization()
        self.t_act = ReLU()
        self.t_conv = Conv2D(filters, kernel_size, padding='same')

    def call(self, x):
        x = self.t_BN(x)
        x = self.t_act(x)
        x = self.t_conv(x)

        return x


class Transition_layer(Model):
    def __init__(self, do, theta):
        super(Transition_layer, self).__init__()

        f = int(np.shape(do)[-1] * theta)
        self.pre_act = pre_activation(filters=f, kernel_size=1)
        self.avg_pool = AvgPool2D(pool_size=2, strides=2, padding='same')

    def call(self, x):
        x = self.pre_act(x)
        x = self.avg_pool(x)

        return x


class Dense_block(Model):
    def __init__(self, k, num):
        super(Dense_block, self).__init__()

        self.num = num
        self.pre_act1 = pre_activation(filters=4 * k, kernel_size=1)
        self.pre_act2 = pre_activation(filters=k, kernel_size=3)
        self.concat = Concatenate()

    def call(self, x):
        for _ in range(self.num):
            input = x
            x = self.pre_act1(x)
            x = self.pre_act2(x)
            x = self.concat([input, x])

        return x


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
        global d
        k = 16
        theta = 0.5
        input = Input(shape=self.input_shape)
        x = Conv2D(filters=2 * k, kernel_size=3, padding="valid")(input)
        for _ in range(self.repetitions):
            d = Dense_block(k, 1)(x)
            x = Transition_layer(d, theta)(d)
        # 마지막 dense block 뒤에는 transition layer가 없음
        x = GlobalAvgPool2D()(d)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(input, output)
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

# 5시간 50분 소요
# Test accuracy: 0.8459