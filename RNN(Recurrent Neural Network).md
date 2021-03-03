## RNN(Recurrent Neural Network)

시퀀스 데이터를 모델링하기 위한 것으로 **hidden state(기억 - 지금까지의 입력데이터를 요약한 정보)** 를 가지며 새로운 입력이 들어올때마다 네트워크는 기억을 조금씩 수정하는 이러한 반복을 통해 아무리 긴 시퀀스라도 처리할 수 있도록 함

실제로는 시점이 충분히 길게되면 앞 쪽에 위치한 정보들이 손실되고 결과를 제대로 예측하지 못하게 되는 **장기 의존성 문제(the problem of Long-Term Dependencies)** 가 있다고함

#### 셀(cell) 

- RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드, 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로 이를 **메모리 셀** 또는 **RNN 셀**이라고 표현
- 각 셀은 0 ~ 1 사이의 값을 갖는 3개의 gate(write, read, keep)로 구성되고 이 값을 통해 cell의 정보를 저장할지 불러올지 유지할지 결정 **(신경망의 weights들과 마찬가지로 학습)**
- 은닉층의 메모리 셀은 **바로 이전 시점에서 은닉층의 메모리 셀에서 나온 값 (은닉 상태) 을 자신의 입력**으로 사용 **(재귀적)**

<img src="https://user-images.githubusercontent.com/58063806/109741781-1c899900-7c11-11eb-8476-9a4f6e3cd791.png" width=50% />

<img src="https://user-images.githubusercontent.com/58063806/109742198-003a2c00-7c12-11eb-91b3-3ab963c903a6.png" width=20% />

은닉층의 메모리 셀은 ht를 계산하기 위해 **입력층에서 입력값에 대한 가중치 Wx**와 **이전 시점 t - 1의 은닉 상태인 h(t-1)에 대한 가중치 Wh** 를 이용

<img src="https://user-images.githubusercontent.com/58063806/109742532-8eaead80-7c12-11eb-8f3f-0a62ea9a7909.png" width=40% />

**배치크기가 1이고 d, Dh가 모두 4인 경우**

<img src="https://user-images.githubusercontent.com/58063806/109743420-04ffdf80-7c14-11eb-93b8-3e2928cd5ce7.png" width=80%/>



```python
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
hidden_size = 2
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
print(x_data.shape) 
pp.pprint(x_data)
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
```

<img src="https://user-images.githubusercontent.com/58063806/95017234-38553700-0693-11eb-8b84-633aa153adb6.JPG" width=50%/>

**shape의 의미 - batch size(1), sequence_length(위의 경우에는 h, e, l, l, o 이므로 5)**

<img src="https://user-images.githubusercontent.com/58063806/95017236-39866400-0693-11eb-8a6f-47a18976259e.JPG" width=50% />

초기화된 weight들 **(hidden size가 2이므로 2 dimension들이 출력)**

```python
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()
hidden_size = 2
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]
x_data = np.array([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]], dtype=np.float32)
print(x_data.shape)
pp.pprint(x_data)
cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
```

다음과 같이 **batch size를 증가(한번에 데이터를 여러개)해서 사용 가능**

<img src="https://user-images.githubusercontent.com/58063806/95017461-64bd8300-0694-11eb-8560-d18b2c33fd5c.JPG" width=35%/>

<img src="https://user-images.githubusercontent.com/58063806/95017462-65eeb000-0694-11eb-9214-7d1e54c7a496.JPG" width=35% />

```python
from keras.models import Sequential
from keras.layers import SimpleRNN
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))
# model.add(SimpleRNN(3, input_length=2, input_dim=10))
plot_model(model, 'model.png', show_shapes=True)
```

hidden size : 3 - 메모리 셀이 다음 시점의 메모리 셀로 보내는 은닉 상태의 크기로 출력층으로 보내는 값의 크기와 동일 (RNN의 용량(capacity)을 늘린다고 보면 되며, 중소형 모델의 경우 보통 128, 256, 512, 1024 등의 값을 가진다고 함)

timesteps : 입력 시퀀스의 길이

input_dim : 입력의 크기

<img src="https://user-images.githubusercontent.com/58063806/109745105-b56ee300-7c16-11eb-9fb5-e4cbb9174ea0.png" width=50%/>

```python
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10)))
```

<img src="https://user-images.githubusercontent.com/58063806/109746068-29f65180-7c18-11eb-8a44-775413a044b1.png" width=50%/>

```python
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10), return_sequences=True))
```

<img src="https://user-images.githubusercontent.com/58063806/109746261-780b5500-7c18-11eb-8fb5-62190470506e.png" width=50% />

<img src="https://user-images.githubusercontent.com/58063806/109741848-4478fc80-7c11-11eb-977f-da5942c4db29.png" width=70% />

RNN은 위와 같이 입력과 출력의 길이에 따라 다양한 용도로 사용이 가능

##### EX)

one-to-many : 하나의 이미지 입력에 대해서 사진의 제목을 출력하는 이미지 캡셔닝에 사용

many-to-one : 입력 문서가 긍정인지 부정인지 판별하는 감성 분류, 스팸 메일 분류 등에 사용

many-to-many : 입력 문장으로부터 대답 문장을 출력하는 챗봇과 입력 문장으로부터 번역된 문장을 출력하는 번역기, 개체명 인식 등에 사용 

[내용 참고 및 이미지 출처](https://wikidocs.net/22886)