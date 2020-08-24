## Tensorflow

기계학습을 위해서 구글에서 만든 오픈소스 라이브러리로 텐서플로우에서 **계산은 데이터 플로우 그래프**로 행해지는데 이때 그래프 **각각의 노드는 수식**을, **간선은 시스템을 따라 흘러가는 데이터(Tensor)**를 나타냄

**Tensor  - 텐서플로우의 기본 데이터 구조**로 보통 **다차원 배열**을 지칭

```python
A = tf.constant([1, 2, 3])
B = tf.constant([3, 4, 5])
```

tf.constant() - 상수 텐서를 생성

- dtype - 텐서 원소들의 타입
- shape - 결과값 텐서의 차원
- name - 텐서의 명칭

```python
W = tf.Variable(tf.random.normal([1]), name="weight")
# tf.random.normal() - 정규분포를 따르고 크기에 맞는 난수를 생성 
```

tf.Variable - 텐서 변수를 생성**(변수는 상수와 다르게 초기화를 해줘야 함)**

tf.global_variables_initializer() - 여러 **변수들을 한번에 초기화**시킴

```python
X = tf.placeholder(tf.float32)
sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5]})
# feed_dict를 이용해서 X에 값을 줌
```

**tf.placeholder() - 변수의 타입을 미리 설정해놓고 필요한 변수를 나중에 받아서 실행**

### 손실함수 값 구하기

**손실함수(평균제곱오차)**

<img src="https://user-images.githubusercontent.com/58063806/91023940-2747ed80-e632-11ea-8f6c-2438d88e83b9.PNG" width=50% />

```python
import tensorflow as tf
x_train = [1, 2, 3]
y_train = [1, 2, 3]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 추론한 값과 정답 레이블의 차를 제곱하고 평균을 냄
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 학습률 0.01로 경사하강법 적용
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict={X: x_train, Y: y_train})
    if step % 20 == 0:
        print(step, cost_val)
        if step == 2000:
            print("W: ", W_val, "b: ", b_val)
```

**tf.Session() - tensorflow 연산들을 실행하기 위한 클래스(operation 객체를 실행하고, tensor 객체를 평가하기 위한 환경을 제공하는 객체)**

<img src="https://user-images.githubusercontent.com/58063806/91024841-7e01f700-e633-11ea-94ad-4cb39a94fec0.PNG" width=20% />

<img src="https://user-images.githubusercontent.com/58063806/91024843-7e9a8d80-e633-11ea-9b5f-75bbc7073b79.PNG" width=20%/>

위의 결과를 보면 **학습을 반복해서 진행할수록 손실함수의 값이 작아지는 것**을 볼 수 있음

<img src="https://user-images.githubusercontent.com/58063806/91025363-3e87da80-e634-11ea-9f98-794db9ca7e5c.PNG" width=60% />

결과적으로 2000번 학습을 진행했을때 **최적의 가중치와 편향의 값**은 위와 같음 

