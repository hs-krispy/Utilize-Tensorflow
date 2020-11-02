import tensorflow as tf
import numpy as np

sample = " if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)} # key가 c이고 value가 i인 dictionary 생성

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

num_classes = len(char2idx)
input_dim = len(char2idx)
hidden_size = len(char2idx)  # output from the LSTM. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = len(sample) - 1
learning_rate = 0.1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# 원래 logits에 RNN에서 나오는 output을 그대로 넣으면 안됨
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(result, "true Y: ", y_data)
        # 결과를 문자열로
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction: ", ' '.join(result_str))