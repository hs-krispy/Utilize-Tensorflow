# 동물의 특징에 따라서 어떤 종인가를 분류
import tensorflow as tf
import numpy as np
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32) # 데이터를 불러옴
x_data = xy[:, 0:-1] # 동물의 특징들
y_data = xy[:, [-1]] # 정답 레이블(어떤 종인가)
nb_classes = 7 # 0 ~ 6
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes) # 1과 0으로 이루어진 형태로 반환(차원이 하나 증가함)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # 차원을 조정(여기서는 17, 7)
W = tf.Variable(tf.random_normal([16, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot) # 교차 엔트로피 오차
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.argmax(hypothesis, 1) # 예측값, 각 행에서 가장 큰 값의 인덱스를 추출, argmax( , 0)이면 열에서 가장 큰 값의 인덱스
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # 예측값과 정답이 얼마나 일치하는가
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc= sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2f}".format(step, loss, acc))
    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()): # y_data를 1차원으로
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
