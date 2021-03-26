import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, Flatten, Dense, ZeroPadding2D


class FeatureExtractor(Layer):
    def __init__(self, filter1, filter2):
        super(FeatureExtractor, self).__init__()

        self.conv1 = Conv2D(filter1, 5, activation="tanh")
        self.conv1_pool = AveragePooling2D(2, 2)
        self.conv2 = Conv2D(filter2, 5, activation="tanh")
        self.conv2_pool = AveragePooling2D(2, 2)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv1_pool(x)
        x = self.conv2(x)
        x = self.conv2_pool(x)

        return x


class LeNet1(Model):
    def __init__(self):
        super(LeNet1, self).__init__()

        self.fe = FeatureExtractor(4, 12)

        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(10, activation="softmax"))

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)

        return x


class LeNet4(Model):
    def __init__(self):
        super(LeNet4, self).__init__()

        self.zero_padding = ZeroPadding2D(2)
        self.fe = FeatureExtractor(4, 16)

        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(120, activation="tanh"))
        self.classifier.add(Dense(10, activation="softmax"))

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)

        return x


class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.zero_padding = ZeroPadding2D(2)
        self.fe = FeatureExtractor(6, 16)

        self.classifier = Sequential()
        self.classifier.add(Flatten())
        self.classifier.add(Dense(140, activation="tanh"))
        self.classifier.add(Dense(84, activation="tanh"))
        self.classifier.add(Dense(10, activation="softmax"))

    def call(self, x):
        x = self.fe(x)
        x = self.classifier(x)

        return x
