## CycleGAN

generator, discriminator 구성을 위한 모듈

- 이미지를 받아 전치 컨볼루션 연산을 수행할 업샘플링 계층
- 컨볼루션 연산을 수행할 다운샘플링 계층
- 충분한 심층 모델을 갖게 하는 residual 계층
  - 기존의 네트워크는 층이 깊어지면 gradient vanishing, exploding, degradation(후반부에 학습이 이루어지지 않음)
  - 기존의 모델보다 optimize가 용이함
  - shortcut connection을 이용해 마지막에 input값을 더함으로써 gradient가 최소 1 이상이 되게하고 네트워크가 더 깊어져도 gradient vanishing 문제가 발생하지 않도록 함 
  



필요한 라이브러리

```python
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import time
from tensorflow.keras import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, ZeroPadding2D, Flatten, Conv2DTranspose, Concatenate
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
```

이미지의 픽셀값을 -1 ~ 1 사이로 scale

```python
def normalize(input_image, label=None):  
    input_image = tf.cast(input_image, tf.float32)
    input_image = (input_image / 127.5) - 1
    return input_image
```

학습에 사용할 데이터 로드 

```python
dataset, metadata = tfds.load('cycle_gan/vangogh2photo', with_info=True, as_supervised=True)
# dataset2, metadata2 = tfds.load('flic/small', with_info=True)
# train_A, train_B = dataset['trainA'], dataset['trainB'] 
# test_A, test_B = dataset['testA'], dataset['testB']
train_A = dataset['trainA']
test_A = dataset['testA'] 
face = np.load("train/train_data(face_256).npy")
train_B, test_B = train_test_split(face, test_size=0.3, random_state=42)
train_B = tf.data.Dataset.from_tensor_slices(train_B)
test_B = tf.data.Dataset.from_tensor_slices(test_B)
```

데이터셋에 normalize를 적용하고 buffer, batch size에 따라 shuffle과 batch를 수행

```python
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE

# num_parallel_calls - 데이터의 전처리를 병렬적으로 수행
# cache - 데이터셋을 캐시, 즉 메모리 또는 파일에 보관, 따라서 두번째 이터레이션부터는 캐시된 데이터를 사용
train_A = train_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_B = train_B.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_A = test_A.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_B = test_B.map(normalize, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

```python
inpA = next(iter(train_A))
inpB = next(iter(train_B))
# 예시 이미지 출력
plt.subplot(121)
plt.title("Train Set A")
plt.imshow(inpA[0]*0.5 + 0.5)
plt.subplot(122)
plt.title("Train Set B")
plt.imshow(inpB[0]*0.5 + 0.5)
```

<img src="https://user-images.githubusercontent.com/58063806/105847086-7d292300-6020-11eb-8a33-ff890942799a.png" width=40% />

conv 계층을 추가 (downsampling)

```python
def downsample(filters, size=3, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same', activation=LeakyReLU(), kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(BatchNormalization())

    return result
```

conv2transpose 계층을 추가 (Upsampling)

```python
def upsample(filters, size=3, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same', activation="relu", kernel_initializer=initializer, use_bias=False))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))

    return result
```

residual 계층 추가

```python
# residual 계층 - 정보 손실이 적고, 고해상도 처리가 가능
class ResnetIdentityBlock(Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = Conv2D(filters1, (1, 1))
        self.bn2a = BatchNormalization()

        self.conv2b = Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = BatchNormalization()

        self.conv2c = Conv2D(filters3, (1, 1))
        self.bn2c = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        # 마지막에 input값을 더함으로써 gradient가 최소 1 이상이 되게하고 gradient vanishing 문제 해결 
        x += input_tensor
        return tf.nn.relu(x)

    
block1 = ResnetIdentityBlock(3, [512, 512, 512])
block2 = ResnetIdentityBlock(3, [512, 512, 512])
block3 = ResnetIdentityBlock(3, [512, 512, 512])


resnet = [block1, block2, block3]
```



### generator architecture

```python
def Generator():
    down_stack = [downsample(64, 4, apply_batchnorm=False), downsample(128, 4), downsample(256, 4), downsample(512, 4)]
    up_stack = [upsample(256, 4), upsample(128, 4), upsample(64, 4), ]

    initializer = tf.random_normal_initializer(0., 0.02)
    # 마지막 층에서 activation을 tanh로 하면서 -1 ~ 1 사이의 값 return 
    last = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

	# 256 x 256 x 3 image
    inputs = Input(shape=[256, 256, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    for block in resnet:
        x = block(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        concat = Concatenate()
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)

 
# Downsampling을 진행 -> resnet-block 통과 -> Upsampling, Concatenate layer (반복마다) 순서로 진행
generator = Generator()
gen_output = generator(inpA, training=False)
# 0 ~ 1로 scaling
gen_output = (gen_output + 1) / 2
# 모델 architecture plot
plot_model(generator, 'generator.png', show_shapes=True)
```

<img src="https://user-images.githubusercontent.com/58063806/105847549-13f5df80-6021-11eb-8246-24dc1353e35f.png" width=60% />

### discriminator architecture

```python
def Discriminator():
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs
    g_filter = 64
    
    down_stack = [downsample(g_filter), downsample(g_filter * 2), downsample(g_filter * 4), downsample(g_filter * 8),]
    
    for down in down_stack:
        x = down(x)
	# 128, 64, 32, 16
    
    last = Conv2D(1, 4, strides=1, padding='same') 
    x = last(x)

    return Model(inputs=inputs, outputs=x)

discriminator = Discriminator()
dis_output = discriminator(inpA, training=False)
plot_model(discriminator, 'discriminator.png', show_shapes=True)
```



<img src="https://user-images.githubusercontent.com/58063806/105311686-d51af080-5c00-11eb-8b40-5a79a11af2ed.png" width=40% />

학습에 필요한 객체들

```python
discriminator_A = Discriminator()
discriminator_B = Discriminator()

generator_AB = Generator()
generator_BA = Generator()

# 손실함수 정의(softmax 함수를 거치지 않으면 from_logits=True)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True) 

# optimzier 정의, label 정의
optimizer = Adam(1e-4, beta_1=0.5)
discriminator_optimizer = Adam(1e-4, beta_1=0.5)
valid = np.ones((BATCH_SIZE, 16, 16, 1)).astype('float32')
fake = np.zeros((BATCH_SIZE, 16, 16, 1)).astype('float32')
```

판별기에서 기존 이미지와 가짜 이미지에 대한 오차의 합

```python
@tf.function
def discriminator_loss(disc_real_output, disc_generated_output):
    # 원래 이미지의 오차
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    # 생성된 가짜 이미지에 대한 오차
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss  
```

모델 학습 과정

```python
@tf.function
def train_batch(imgs_A, imgs_B):
    with tf.GradientTape() as g, tf.GradientTape() as d_tape:
        fake_B = generator_AB(imgs_A, training=True) # A 이미지에서 fake B 생성
        fake_A = generator_BA(imgs_B, training=True) # B 이미지에서 fake A 생성
        
        logits_real_A = discriminator_A(imgs_A, training=True)
        logits_fake_A = discriminator_A(fake_A, training=True)
        # 실제 A 이미지와 fake A 이미지에 대한 loss를 합한 판별기 A의 loss
        dA_loss = discriminator_loss(logits_real_A, logits_fake_A) 
        
        logits_real_B = discriminator_B(imgs_B, training=True) 
        logits_fake_B = discriminator_B(fake_B, training=True)
        # 실제 B 이미지와 fake B 이미지에 대한 loss를 합한 판별기 B의 loss
        dB_loss = discriminator_loss(logits_real_B, logits_fake_B) 
        
        # 판별기 loss 평균값
        d_loss = (dA_loss + dB_loss) / 2
        
        # Translate images back to original domain
        
        # fake B 이미지에서 다시 A 이미지로 재구성
        reconstr_A = generator_BA(fake_B, training=True) 
        # fake A 이미지에서 다시 B 이미지로 재구성
        reconstr_B = generator_AB(fake_A, training=True) 
        # 이때 재구성한 이미지가 기존의 A, B 이미지와 일관성을 유지해야(유사해야) 함
        
        # A 이미지를 생성기에 넣고 기존의 A 이미지와 비교할 이미지 생성
        id_A = generator_BA(imgs_A, training=True) 
        # B 이미지를 생성기에 넣고 기존의 B 이미지와 비교할 이미지 생성
        id_B = generator_AB(imgs_B, training=True) 


        gen_loss = tf.math.reduce_sum([
            # 기존의 생성기 loss
            1 * tf.math.reduce_mean(mean_squared_error(logits_fake_A, valid)),
            1 * tf.math.reduce_mean(mean_squared_error(logits_fake_B, valid)),
            # cycle consistency loss(주기 일관성 loss)
            10 * tf.math.reduce_mean(mean_squared_error(reconstr_A, imgs_A)),
            10 * tf.math.reduce_mean(mean_squared_error(reconstr_B, imgs_B)),
            0.1 * tf.math.reduce_mean(mean_squared_error(id_A, imgs_A)),
            0.1 * tf.math.reduce_mean(mean_squared_error(id_B, imgs_B)),
        ])
    
    # 판별기 A, B에 대해서 gradient를 계산
    gradients_of_d = d_tape.gradient(d_loss, discriminator_A.trainable_variables + discriminator_B.trainable_variables)
    # 판별기 A, B에 대해서 backpropagation으로 가중치를 update
    discriminator_optimizer.apply_gradients(zip(gradients_of_d, discriminator_A.trainable_variables + discriminator_B.trainable_variables))

    # 생성기 A, B에 대해서 gradient를 계산
    gradients_of_generator = g.gradient(gen_loss, generator_AB.trainable_variables + generator_BA.trainable_variables)
    # 생성기 A, B에 대해서 backpropagation으로 가중치를 update
    optimizer.apply_gradients(zip(gradients_of_generator, generator_AB.trainable_variables + generator_BA.trainable_variables))
    
    # 각 판별기와 생성기의 loss return
    return dA_loss, dB_loss, gen_loss
```

<img src="https://user-images.githubusercontent.com/58063806/106097088-09029280-617a-11eb-9c52-281784c60e91.png" width=80%/>

**generator 1 : G1(X) => Fake Y ** 

**generator 2 : G2(Y) => Fake X**

generator loss - 기존의 이미지와 생성기가 생성한 가짜 이미지 사이의 오차

EX) 

- generator 1 loss : Y와 Fake Y의 오차
- generator 2 loss : X와 Fake X의 오차
- cycle consistency loss 
  - **생성자에 의해 생성된 이미지를 기존 이미지로 복원했을 때 기존의 이미지와의 오차**
  - generator 1에서 생성한 Fake Y를 generator 2의 입력으로해서 나온 Fake X와 기존 X의 오차 
  - G2(G1(X)) => Fake X
  - generator 2에서 생성한 Fake X를 generator 1의 입력으로해서 나온 Fake Y와 기존 Y의 오차 
  - G1(G2(Y)) => Fake Y
- Identity loss : 생성기의 출력에 해당하는 값을 입력으로 넣으면 출력으로 입력과 유사한 이미지가 나와야 함
  - generator 1에 Y를 입력으로해서 나온 Fake Y와 기존 Y의 오차 
  - G1(Y) => Fake Y
  - generator 2에 X를 입력으로해서 나온 Fake X와 기존 X의 오차  
  - G2(X) => Fake X

#### Gradient_tape 

- 자동 미분(주어진 입력 변수에 대한 연산의 gradient를 계산하는 것)
- 해당 영역안에서 실행되는 모든 연산을 tape에 기록하고  [후진 방식 자동 미분(reverse mode differentiation)](https://en.wikipedia.org/wiki/Automatic_differentiation)을 사용해 tape에 기록된 연산의 gradient를 계산

모델 가중치를 저장할 체크포인트 지정

```python
checkpoint_dird_A = './training_checkpointsd_A'
checkpoint_prefixd_A = os.path.join(checkpoint_dird_A, "ckpt_{epoch}")

checkpoint_dird_B = './training_checkpointsd_B'
checkpoint_prefixd_B = os.path.join(checkpoint_dird_B, "ckpt_{epoch}")

checkpoint_dirg_AB = './training_checkpointsg_AB'
checkpoint_prefixg_AB = os.path.join(checkpoint_dirg_AB, "ckpt_{epoch}")

checkpoint_dirg_BA = './training_checkpointsg_BA'
checkpoint_prefixg_BA = os.path.join(checkpoint_dirg_BA, "ckpt_{epoch}")
```

학습과 결과 시각화

```python
def train(trainA_, trainB_, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        for batch_i, (imgs_A, imgs_B) in enumerate(zip(trainA_, trainB_)):
            # train img A, B에 대해 학습을 진행하고 loss 반환
            dA_loss, dB_loss, g_loss = train_batch(imgs_A, imgs_B)
            print ("%d [DA loss: %f, DB loss: %.2f%%] [G loss: %f]" % (epoch, dA_loss, dB_loss, g_loss))
            # 학습 중 1000번 단위로 test img A, B에 대해 결과 시각화
            if batch_i % 1000 == 0:
                test_imgA = next(iter(test_A))
                test_imgB = next(iter(test_B))
                print ('Time taken for epoch {} batch index {} is {} seconds\n'.format(epoch, batch_i, time.time()-start))
                print("discriminator A: ", dA_loss.numpy())
                print("discriminator B: ", dB_loss.numpy())
                print("generator: {}\n".format(g_loss))

                fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)
                gen_outputA = generator_AB(test_imgA, training=False)
                gen_outputB = generator_BA(test_imgB, training=False)
                axs[0,0].imshow(test_imgA[0]*0.5 + 0.5)
                axs[0,0].set_title("Generator A Input")
                axs[0,1].imshow(gen_outputA[0]*0.5 + 0.5)
                axs[0,1].set_title("Generator A Output")
                axs[1,0].imshow(test_imgB[0]*0.5 + 0.5)
                axs[1,0].set_title("Generator B Input")
                axs[1,1].imshow(gen_outputB[0]*0.5 + 0.5)
                axs[1,1].set_title("Generator B Output")
                plt.show()

# 각 판별기와 생성기의 가중치 저장 
                discriminator_A.save_weights(checkpoint_prefixd_A.format(epoch=epoch))
                discriminator_B.save_weights(checkpoint_prefixd_B.format(epoch=epoch))
                generator_AB.save_weights(checkpoint_prefixg_AB.format(epoch=epoch))
                generator_BA.save_weights(checkpoint_prefixg_BA.format(epoch=epoch))
```

<img src="https://user-images.githubusercontent.com/58063806/106099887-c1cad080-617e-11eb-98f8-7258b9c76db5.png" width=60%/>

학습이 종료된 후 모델을 저장

```python
discriminator_A.save_weights('discriminator_A.h5')
discriminator_B.save_weights('discriminator_B.h5')
generator_AB.save_weights('generator_AB.h5')
generator_BA.save_weights('generator_BA.h5')
```

저장된 가중치를 불러와 새로운 image에 적용

```python
discriminator_A.load_weights('./discriminator_A.h5')
discriminator_B.load_weights('./discriminator_B.h5')
generator_AB.load_weights('./generator_AB.h5')
generator_BA.load_weights('./generator_BA.h5')

from PIL import Image
img = Image.open("face_image/test_face.jpg")
img = img.resize((256, 256))
img_tensor = np.array(img)
img_tensor = normalize(img_tensor)
img_tensor = np.expand_dims(img_tensor, axis=0)
fake_B = generator_AB(inpA, training=False)
fake_A = generator_BA(img_tensor, training=False)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0,0].imshow(inpA[0]*0.5 + 0.5)
axs[0,0].set_title("Generator A Input")
axs[0,1].imshow(fake_B[0]*0.5 + 0.5)
axs[0,1].set_title("Generator A Output")
axs[1,0].imshow(img_tensor[0]*0.5 + 0.5)
axs[1,0].set_title("Generator B Input")
axs[1,1].imshow(fake_A[0]*0.5 + 0.5)
axs[1,1].set_title("Generator B Output")
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/106100462-bcba5100-617f-11eb-9ed8-0cc90304df6c.png" width=60% />