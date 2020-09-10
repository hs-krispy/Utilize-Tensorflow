##  Convolution and Pooling in MNIST

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
img = mnist.train.images[0].reshape(28, 28)
plt.imshow(img, cmap="gray")
img = img.reshape(-1, 28, 28, 1) # 28 x 28 (1가지 색)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01)) 
# 3 x 3 필터(1가지 색) 5개
```

<img src="https://user-images.githubusercontent.com/58063806/92679500-924c3080-f363-11ea-93b9-e01750ff0989.PNG" width=50% />

#### Convolution

```python
fig = plt.figure(figsize=(15, 20))
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding="SAME") 
# padding이 same이므로 입력과 출력의 크기는 같아야하지만(스트라이드가 1 x 1 일때) 스트라이드가 2 x 2이므로 (출력을 14 x 14로)
print(conv2d)
# Tensor("Conv2D:0", shape=(1, 14, 14, 5), dtype=float32) 출력
sess.run(tf.global_variables_initializer())
# 이미지를 출력하기 위함
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    cimage = fig.add_subplot(1, 5, i + 1)
	cimage.imshow(one_img.reshape(14, 14), cmap="gray")
```

<img src="https://user-images.githubusercontent.com/58063806/92680429-c9234600-f365-11ea-8752-bdf0678ea7e3.PNG" width=100% />

#### Pooling

``` python
fig2 = plt.figure(figsize=(15, 20))
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
# 풀링의 크기가 2 x 2 이므로 출력의 크기는 7 x 7
print(pool)
# Tensor("MaxPool:0", shape=(1, 7, 7, 5), dtype=float32) 출력 
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    pimage = fig2.add_subplot(1, 5, i + 1)
    pimage.imshow(one_img.reshape(7, 7), cmap="gray")
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/92680714-8d3cb080-f366-11ea-95ea-461f8e4dbd28.PNG" width=100% />

**이미지의 크기가 작아질수록 해상도가 작아지는 것을 알 수 있음**