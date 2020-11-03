import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


timesteps = seq_length = 7
data_dim = 5
output_dim = 1
hidden_dim = 5
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1] # 시간순으로 만들기 위해 reverse
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]

x_data = []
y_data = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]
    print(_x, "->", _y)
    x_data.append(_x)
    y_data.append(_y)

train_size = int(len(y_data) * 0.7)
test_size = int(len(y_data) * 0.3) - train_size
train_x, test_x = np.array(x_data[0: train_size]), np.array(x_data[train_size: len(x_data)])
train_y, test_y = np.array(y_data[0: train_size]), np.array(y_data[train_size: len(y_data)])
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
# batch size, seq_length, data_dimension
Y = tf.placeholder(tf.float32, [None, 1])
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
print(cell)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
# outputs 중 마지막 하나만 사용, 최종 출력 크기는 1

loss = tf.reduce_mean(tf.square(Y_pred - Y))
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train, loss], feed_dict={X: train_x, Y: train_y})
    print(i, l)
testPredict = sess.run(Y_pred, feed_dict={X: test_x})

import matplotlib.pyplot as plt
plt.plot(test_y)
plt.plot(testPredict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()