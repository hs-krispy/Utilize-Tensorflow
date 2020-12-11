from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
# input_shape=(128, 128)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

nb_epoch = 30
nb_classes = 10
seed = 100
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train_mean = np.mean(X_train, axis=(0, 1, 2))
X_train_std = np.std(X_train, axis=(0, 1, 2))
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

last = base_model.get_layer('block5_pool').output
x = Flatten()(last)
x = Dense(1024, activation='relu', kernel_initializer="he_uniform")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_initializer="he_uniform")(x)
x = Dense(256, activation='relu', kernel_initializer="he_uniform")(x)
output = Dense(10, activation='softmax')(x)

model = Model(base_model.input, output)

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epoch, batch_size=128, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0]) # 0.71
print("acc: %.2f" % scores[1]) # 0.86

plt.title('model accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

model.save('VGG16.h5')