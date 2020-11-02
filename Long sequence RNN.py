import tensorflow as tf
import numpy as np

sentence = "  if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
char_set = list(set(sentence))
char_dict = {c: i for i, c in enumerate(char_set)}  # key가 c이고 value가 i인 dictionary 생성

num_classes = len(char_set)
input_dim = len(char_set)
hidden_size = len(char_set)  # output from the LSTM. 5 to directly predict one-hot
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dict[c] for c in x_str]  # x str to index
    y = [char_dict[c] for c in y_str]  # y str to index

    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)


def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell


cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
# softmax_b = tf.get_variable("softmax_b", [num_classes])
# outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# activation function을 거치지 않은 outputs
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train, loss, outputs], feed_dict={X: x_data, Y: y_data})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: x_data})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
