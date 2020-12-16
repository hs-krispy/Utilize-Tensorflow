import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, SeparableConv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

X_train = np.load('D:/dataset/train/X_train(150_gray).npy')
y_train = np.load('D:/dataset/train/y_train(150_gray).npy')
X_val = np.load('D:/dataset/val/X_val(150_gray).npy')
y_val = np.load('D:/dataset/val/y_val(150_gray).npy')
X_test = np.load('D:/dataset/test/X_test(150_gray).npy')
y_test = np.load('D:/dataset/test/y_test(150_gray).npy')

y_train = np_utils.to_categorical(y_train, 2)
y_val = np_utils.to_categorical(y_val, 2)
y_test = np_utils.to_categorical(y_test, 2)

epoch = 50
batch_size = 16
seed = 100
np.random.seed(seed)

generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                               zoom_range=0.3, vertical_flip=True, horizontal_flip=True)
train_data = generator.flow(X_train, y_train, batch_size=batch_size, seed=seed)

model = Sequential()
model.add(SeparableConv2D(16, kernel_size=(3, 3), input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), padding="same", activation="relu", kernel_initializer="he_uniform"))
model.add(SeparableConv2D(32, kernel_size=(3, 3), padding="same", activation="relu", kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())
model.add(SeparableConv2D(64, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(64, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(SeparableConv2D(128, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(128, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(SeparableConv2D(256, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(SeparableConv2D(256, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer="he_uniform"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_initializer="he_uniform"))
model.add(Dense(512, activation='relu', kernel_initializer="he_uniform"))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer="he_uniform"))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer=Adam(lr=0.0001, decay=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

callbacks = [ModelCheckpoint(filepath='best_model.h5', verbose=1, monitor='acc', save_best_only=True)]
history = model.fit(train_data, steps_per_epoch=len(X_train) // batch_size, epochs=epoch, verbose=1,
                    class_weight={0:3.0, 1:1.0}, validation_data=(X_val, y_val), callbacks=callbacks)
print('\nTest loss: {:.4f}\nTest accuracy: {:.4f}'.format(model.evaluate(X_test, y_test)[0], model.evaluate(X_test, y_test)[1]))

plt.title('model accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

model.save('model.h5')


