## RNN(Recurrent Neural Network)

시퀀스 데이터를 모델링하기 위한 것으로 **hidden state(기억 - 지금까지의 입력데이터를 요약한 정보)** 를 가지며 새로운 입력이 들어올때마다 네트워크는 기억을 조금씩 수정하는 이러한 반복을 통해 아무리 긴 시퀀스라도 처리할 수 있도록 함

#### 셀(cell) 

- RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드, 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로 이를 **메모리 셀** 또는 **RNN 셀**이라고 표현
- 각 셀은 0 ~ 1 사이의 값을 갖는 3개의 gate(write, read, keep)로 구성되고 이 값을 통해 cell의 정보를 저장할지 불러올지 유지할지 결정 **(신경망의 weights들과 마찬가지로 학습)**
- 은닉층의 메모리 셀은 **바로 이전 시점에서 은닉층의 메모리 셀에서 나온 값을 자신의 입력**으로 사용 **(재귀적)**

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