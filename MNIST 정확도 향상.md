## MNIST 정확도 향상

#### Base

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 0 ~ 9
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random.normal([784, nb_classes]))
b = tf.Variable(tf.random.normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs = 50
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        summary = tf.summary.merge_all()
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) 
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print("Accuarcy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
```

<img src="https://user-images.githubusercontent.com/58063806/92307099-e6a38900-efce-11ea-955a-310dfb78e7cf.JPG" width=30% />

#### NN for MNIST

**softmax대신 relu함수를 이용하고 3개의 layer를 통해 학습**

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 0 ~ 9
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W1 = tf.Variable(tf.random.normal([784, 256]))
# Xavier 초기값
# W1 = tf.get_variable("W1", shape=([784, 256]), initializer=tf.contrib.layers.xavier_initializer())
# He 초기값
# W1 = tf.get_variable("W1", shape=([784, 256]), initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.random.normal([256, 256]))
b2 = tf.Variable(tf.random.normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
W3 = tf.Variable(tf.random.normal([256, 10]))
b3 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L2, W3) + b3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs = 50
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        summary = tf.summary.merge_all()
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) 
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print("Accuarcy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
```

<img src="https://user-images.githubusercontent.com/58063806/92307101-e73c1f80-efce-11ea-9fed-2a5ee9849fc9.JPG" width=30% />

base보다 5%정도 정확도가 향상된 것을 볼 수 있음

(Xavier 초기값을 사용하면 0.4%, He 초기값은 0.5% 정도 정확도가 더 향상)

#### DEEP NN for MNIST

5개의 layer를 통해 학습

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 0 ~ 9
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W1 = tf.get_variable("W1", shape=([784, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.get_variable("W2", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
W3 = tf.get_variable("W3", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
W4 = tf.get_variable("W4", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
W5 = tf.get_variable("W5", shape=([512, 10]), initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs = 50
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        summary = tf.summary.merge_all()
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) 
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print("Accuarcy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
```

<img src="https://user-images.githubusercontent.com/58063806/92308012-7d734400-efd5-11ea-8968-2cff769b6fa1.JPG" width=30% />

#### Dropout for MNIST

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 0 ~ 9
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
keep_prob = tf.placeholder(tf.float32)
W1 = tf.get_variable("W1", shape=([784, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random.normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob) 
# 얼마의 layer를 남길 것 인가(keep_prob는 보통 train 시에는 0.5 ~ 0.7, test 시에는 1)
W2 = tf.get_variable("W2", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random.normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
W3 = tf.get_variable("W3", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.random.normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
W4 = tf.get_variable("W4", shape=([512, 512]), initializer=tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.random.normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
W5 = tf.get_variable("W5", shape=([512, 10]), initializer=tf.contrib.layers.variance_scaling_initializer())
b5 = tf.Variable(tf.random.normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs = 50
batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        summary = tf.summary.merge_all()
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) 
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print("Accuarcy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
```

<img src="https://user-images.githubusercontent.com/58063806/92308030-9f6cc680-efd5-11ea-9463-436ea960d1cf.JPG" width=30% />

 오버피팅 방지를 위해 Dropout을 함으로써 정확도가 올라가는 것을 볼 수 있음