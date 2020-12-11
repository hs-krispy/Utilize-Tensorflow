from keras.datasets import cifar10
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
import efficientnet.keras as efn
import matplotlib.pyplot as plt
import numpy as np
img_width, img_height = 32, 32
base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32, 32, 3), drop_connect_rate=0.5)
nb_epoch = 50
nb_classes = 10
seed = 100
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

last = base_model.get_layer('top_activation').output
x = Flatten()(last)
x = Dense(1024, activation='relu', kernel_initializer="he_uniform")(x)
x = Dense(512, activation='relu', kernel_initializer="he_uniform")(x)
x = Dense(256, activation='relu', kernel_initializer="he_uniform")(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

model = Model(base_model.input, output)

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epoch, batch_size=128, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print("loss: %.2f" % scores[0])
print("acc: %.2f" % scores[1])
# test acc : 0.88

plt.title('model accuracy')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()