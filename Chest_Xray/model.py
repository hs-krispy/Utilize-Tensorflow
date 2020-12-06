import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import image, ImageDataGenerator

X_train = np.load('D:/dataset/train/X_train.npy')
y_train = np.load('D:/dataset/train/y_train.npy')
X_val = np.load('D:/dataset/val/X_val.npy')
y_val = np.load('D:/dataset/val/y_val.npy')
X_test = np.load('D:/dataset/test/X_test.npy')
y_test = np.load('D:/dataset/test/y_test.npy')

y_train = np_utils.to_categorical(y_train, 2)
y_val = np_utils.to_categorical(y_val, 2)
y_test = np_utils.to_categorical(y_test, 2)

seed = 100
np.random.seed(seed)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(150, 150, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu', kernel_initializer="he_uniform"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_initializer="he_uniform"))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

print('\nTrain accuracy: {:.4f}\nTest accuracy: {:.4f}'.format(model.evaluate(X_train, y_train)[1], model.evaluate(X_test, y_test)[1]))
# Train accuracy - 0.9918
# Test accuracy - 0.8093

plt.title('model accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


