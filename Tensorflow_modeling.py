import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

n_train, n_val, n_test = 1000, 300, 300

X_train = np.random.normal(0, 1, size=(n_train, 1)).astype(np.float32)
X_train_noise = X_train + 0.2 * np.random.normal(0, 1, (n_train, 1))
y_train = (X_train_noise > 0).astype(np.int32)

X_val = np.random.normal(0, 1, size=(n_val, 1)).astype(np.float32)
X_val_noise = X_val + 0.2 * np.random.normal(0, 1, (n_val, 1))
y_val = (X_val_noise > 0).astype(np.int32)

X_test = np.random.normal(0, 1, size=(n_test, 1)).astype(np.float32)
X_test_noise = X_test + 0.2 * np.random.normal(0, 1, (n_test, 1))
y_test = (X_test_noise > 0).astype(np.int32)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean, CategoricalAccuracy

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(n_train).batch(8)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(n_val)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(n_test)

model = Sequential()
model.add(Dense(2, activation="softmax"))


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation="softmax")

    def call(self, x):
        x = self.d1(x)
        return x


# model = MyModel()

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

val_loss = Mean()
val_acc = SparseCategoricalAccuracy()

test_loss = Mean()
test_acc = SparseCategoricalAccuracy()

epochs = 30

train_losses, val_losses = [], []
train_accs, val_accs = [], []

@tf.function  # 최적화 코드 (학습을 더 빠르게)
def train_step(x, y):
    global model, loss_object
    global train_loss, train_acc

    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(y, predictions)


@tf.function
def validation():
    global val_ds, model, loss_object
    global val_loss, val_acc

    for x, y in val_ds:
        predictions = model(x)
        loss = loss_object(y, predictions)

        val_loss(loss)
        val_acc(y, predictions)


def train_reporter():
    global epoch
    global train_loss, train_acc
    global val_loss, val_acc

    print(colored("Epoch: ", 'red', 'on_white'), epoch + 1)
    template = "Train loss: {:.4f}\t Train accuracy: {:.2f}% Validation loss: {:.4f}\t Validation accuracy : {:.2f}%\n"

    print(template.format(train_loss.result(), train_acc.result() * 100, val_loss.result(), val_acc.result() * 100))


def metric_resetter():
    global train_loss, train_acc
    global val_loss, val_acc

    train_losses.append(train_loss.result())
    train_accs.append(train_acc.result() * 100)
    val_losses.append(val_loss.result())
    val_accs.append(val_acc.result() * 100)

    # 한 epoch당 loss, acc 평균, result_states 안하면 전체 epoch 누적평균
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()


def final_result_visualization():
    global train_losses, validation_losses
    global traina_accs, val_accs

    fig, axes = plt.subplots(2, 1, figsize=(20, 15))

    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(val_losses, label="Validation loss")
    axes[1].plot(train_accs, label="Train accuracy")
    axes[1].plot(val_accs, label="Validation accuracy")

    axes[0].set_ylabel("Binary Cross Entropy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[0].legend()
    axes[1].legend()


for epoch in range(epochs):
    for x, y in train_ds:
        train_step(x, y)

    validation()
    train_reporter()
    metric_resetter()

for x, y in test_ds:
    predictions = model(x)
    loss = loss_object(y, predictions)

    test_loss(loss)
    test_acc(y, predictions)

final_result_visualization()

print(colored("Final result: ", 'cyan', 'on_white'))
template = "Test loss: {:.4f}\t Test accuracy: {:.2f}%\n"

print(template.format(test_loss.result(), test_acc.result() * 100))