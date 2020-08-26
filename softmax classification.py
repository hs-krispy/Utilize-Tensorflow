# softmax classification - 다중 분류를 해결하기 위한 모델로 여러 개의 연산 결과를 정규화하여 모든 클래스의 확률값의 합을 1로 만듬
import tensorflow as tf
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]] # 원 핫 인코딩
X = tf.placeholder("float", shape=[None, 4])
Y = tf.placeholder("float", shape=[None, 3])
nb_classes = 3 # 원 핫 인코딩을 이용할 때는 y의 개수가 클래스의 개수
W = tf.Variable(tf.random_normal([4, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # 교차 엔트로피 오차
prediction = tf.argmax(hypothesis, 1) # 예측값
res = tf.argmax(Y, axis=1) # 정답
correct_prediction = tf.equal(prediction, res) # 얼마나 일치하는가
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("ans : ", sess.run(res, feed_dict={Y: y_data}))
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            loss, pred, acc = sess.run([cost, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step : ", step, "\tLoss : {}\t Prediction : {}\t Acc : {}".format(loss, pred, acc))
            # 반복횟수, 손실함수 값, 예측값, 정확도