import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

seed = 100
np.random.seed(seed)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(32, 32, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
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
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=250, verbose=1)

plt.title('model accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

print('\nTrain accuracy: {:.4f}\nTest accuracy: {:.4f}'.format(model.evaluate(X_train, y_train)[1], model.evaluate(X_test, y_test)[1]))

# Train acc : 0.9693
# Test acc : 0.8745