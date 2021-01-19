## DCGAN

##### Deep Convolutional Generative Adversarial Networks with Face dataset

### class 구성

```python
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, ZeroPadding2D, Flatten, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image


    def __init__(self, rows, cols, channels, z=100):
        # Input shape
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = z
        optimizer = Adam(0.0002, 0.5)
        self.G_losses = []
        self.D_losses = []
        self.x = []
        
        self.train_data = np.load("train/train_data(face).npy")
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        # generator가 생성한 가짜 이미지
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # discriminator가 가짜 이미지를 판별한 결과
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # 입력 노이즈와 그에 대한 discriminator의 판별 결과
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)   
```

클래스 내부에서 사용할 변수, 객체들을 초기화

```python
    def save_data(self, path):
        print(path)
        img_list = os.listdir(path)
        X = []
        for img_name in img_list:
            img_path = path + '/' + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_tensor = np.array(img)
            X.append(img_tensor)

        np.save("train/train_data(face).npy", X)
```

이미지들이 존재하는 디렉토리 경로를 받아와 image들을 numpy array 파일로 변환

```python
    def build_generator(self):
            model = Sequential()

            model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
            model.add(Reshape((8, 8, 128)))
            # UpSampling2D - 이미지의 열과 행을 두배로 증가
            model.add(UpSampling2D()) # 16, 16, 128
            model.add(Conv2DTranspose(128, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(UpSampling2D()) # 32, 32, 128
            model.add(Conv2DTranspose(128, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(UpSampling2D()) # 64, 64, 128
            model.add(Conv2DTranspose(256, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            # 64, 64, 256
            model.add(Conv2DTranspose(self.channels, kernel_size=3, padding="same"))
            model.add(Activation("tanh"))

            model.summary()
            # 노이즈
            noise = Input(shape=(self.latent_dim,))
            # 노이즈를 이용해 이미지 생성
            img = model(noise)
            # input - 노이즈, output - Convolution 과정을 거쳐 생성된 img 
            return Model(noise, img)
```

generator (생성기)를 구성

```python
   def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # 32, 32, 64
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # 16, 16, 64
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # 8, 8, 128
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # 8, 8, 256
        model.add(Flatten())
        # 8 * 8 * 256
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        # image
        img = Input(shape=self.img_shape)
        # image를 이용해 얻어진 결과 (0 ~ 1 사이의 값)
        validity = model(img)
		# input - image, output - Convolution과 여러 과정을 거쳐서 얻은 image에 대한 값 
        return Model(img, validity)
```

discriminator (판별기)를 구성

```python
    def train(self, epochs, batch_size=256, save_interval=50):
        X_train = self.train_data
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        batchCount = int(X_train.shape[0] / batch_size)
        
        # 레이블 값 스무딩 (더 나은 학습을 위함)
        valid = np.ones((batch_size, 1)) * 0.9
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('-'*15, 'Epoch %d' % epoch, '-'*15)
            g_loss_mean = 0
            d_loss_mean = 0
            for _ in range(batchCount):

            	# batch_size 만큼 img들을 선택
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # 0 ~ 1 사이의 값을 갖는 노이즈
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # 노이즈로 얻어진 가짜 이미지
                gen_imgs = self.generator.predict(noise)
                
                # discriminator의 input, label을 정의
                imgs = np.concatenate((imgs, gen_imgs))
                label = np.concatenate((valid, fake))
                # discriminator 학습
                d_loss = self.discriminator.train_on_batch(imgs, label)
                # 1 에폭에서 발생하는 discriminator loss의 합
                d_loss_mean += d_loss[0]

                # generator 학습 (노이즈를 통해 얻어지는 이미지를 진짜라고 학습)
                g_loss = self.combined.train_on_batch(noise, valid)
                # 1 에폭에서 발생하는 generator loss의 합 
                g_loss_mean += g_loss
                
                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                    
            if epoch % save_interval == 0:
                # 1 에폭 당 각 loss의 평균값
                g_loss_mean /= batchCount
                d_loss_mean /= batchCount
                self.G_losses.append(g_loss_mean)
                self.D_losses.append(d_loss_mean)
                self.x.append(epoch)
                self.save_imgs(epoch)
```

학습 과정을 구성

```python
    def save_imgs(self, epoch):
        r, c = 5, 5
        # seed를 고정
        np.random.seed(seed=100)
        # 25개의 노이즈
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(7, 7))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, ])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("new_face/face_%d.png" % epoch)
        plt.close()
```

지정된 조건마다 이미지를 저장하는 과정을 구성 

```python
def show_loss(self):
    plt.title("Generator and Discriminator Loss During Training") 
    plt.plot(self.x, self.G_losses,label="Generator") 
    plt.plot(self.x, self.D_losses,label="Discriminator") 
    plt.xlabel("epoch") 
    plt.ylabel("Loss") 
    plt.legend() 
    plt.show()
```

학습을 마치고 generator와 discriminator의 loss 추이를 시각화

```python
dcgan = DCGAN(64, 64, 3)
dcgan.train(epochs=300, batch_size=64, save_interval=30)
dcgan.show_loss()
```

### 생성된 이미지

**Epoch 0**

<img src="https://user-images.githubusercontent.com/58063806/104996175-bedb2c00-5a6a-11eb-998f-41f2194049e0.png" width=50% />

**Epoch 30**

<img src="https://user-images.githubusercontent.com/58063806/104996178-c1d61c80-5a6a-11eb-828e-141ce98ad67a.png" width=50%/>

**Epoch 60**

<img src="https://user-images.githubusercontent.com/58063806/104996181-c39fe000-5a6a-11eb-8b85-677b2a8a446d.png" width=50% />

**Epoch 150**

<img src="https://user-images.githubusercontent.com/58063806/104996466-45900900-5a6b-11eb-9a78-641ee962dd7d.png" width=50% />

**Epoch 270**

<img src="https://user-images.githubusercontent.com/58063806/105001795-a7ed0780-5a73-11eb-8f11-3f23c2dd013f.png" width=50% />



### Loss 변화 시각화

<img src="https://user-images.githubusercontent.com/58063806/105002868-4332ac80-5a75-11eb-8caa-ff545e7c2332.png" width=60% />

- g_loss는 감소하고 d_loss는 증가하는 추세를 보임 
- 점점 더 가짜 이미지가 진짜 이미지와 유사해지고 있는 모습을 볼 수 있음
- 두 loss가 0.5에 수렴하는 것이 이상적
- epoch을 더 크게 할 필요가 있음