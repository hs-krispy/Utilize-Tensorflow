## Conditional GAN

#### Dataset

<img src="https://user-images.githubusercontent.com/58063806/107040361-78bfef80-6802-11eb-9cc9-f5508d7dab94.png" width=30% />

class 1을 10만개 생성

#### library load & combine model 구성

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, 
    ZeroPadding2D, Flatten, Conv2DTranspose, concatenate, Embedding, multiply
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop

data = pd.read_csv("train.csv")
y = data[data.columns[-1]]
data = data[data.columns[1:-1]]
scaler = MinMaxScaler()
scale_data = scaler.fit_transform(data)
data = pd.DataFrame(data=scale_data, columns=data.columns)

class CGAN():
    def __init__(self, rows, cols, z=100):
        # Input shape
        self.img_rows = rows
        self.img_cols = cols
        self.latent_dim = z
        # class의 개수(해당 dataset은 3개)
        self.label_num = 3
        optimizer = RMSprop(lr=0.00005)
        self.G_losses = []
        self.D_losses = []
        self.x = []

        self.train_data = data
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        # label 정보
        label = Input(shape=(1,))
        # noise와 label 정보를 같이 입력
        fake = self.generator([z, label])
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        # 생성기에서 생성된 fake data와 label 정보를 같이 입력
        valid = self.discriminator([fake, label])
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
```

#### generator layer

```python
    def build_generator(self):
            model = Sequential()

            # input_dim - data의 feature 개수와 동일
            model.add(Dense(1024, activation="relu", input_dim=self.latent_dim))
            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dropout(0.5))
            model.add(Dense(data.shape[1], activation='sigmoid'))

            model.summary()

            noise = Input(shape=(self.latent_dim,))
            label = Input(shape=(1,), dtype='int32')
            # label을 noise와 같은 차원으로 만듬
            label_embedding = Flatten()(Embedding(self.label_num + 1, self.latent_dim)(label))
            # element-wise multiplication
            model_input = multiply([noise, label_embedding])
            fake = model(model_input)

            # input - noise, label 
            # output - fake data
            return Model([noise, label], fake)
```

**example of Input**

<img src="https://user-images.githubusercontent.com/58063806/106865493-9e6bcc80-670e-11eb-98e6-320c127f404f.png" width=50%/>

#### discriminator layer

```python
    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(128, activation="relu", input_dim=data.shape[1]))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
		
        # generated(fake) data
        fake = Input(shape=(data.shape[1]), )
        label = Input(shape=(1,), dtype='int32')
		
        # label을 feature와 동일한 차원으로 만듬
        label_embedding = Flatten()(Embedding(self.label_num + 1, data.shape[1])(label))
        # element-wise multiplication 
        model_input = multiply([img, label_embedding])
        validity = model(model_input)
		
        # input - generated(fake) data, label
        # output - fake에 대해 판별한 0 ~ 1 사이의 값
        return Model([fake, label], validity)
```

#### 전체적인 학습과정

```python
    def train(self, epochs, batch_size=256, save_interval=50):
        global d_loss, g_loss
        X_train = self.train_data
        batchCount = int(X_train.shape[0] / batch_size)

        # Adversarial ground truths
        valid = np.ones(batch_size) * 0.9
        fake = np.zeros(batch_size)

        for epoch in range(epochs):
            print('-' * 15, 'Epoch %d' % epoch, '-' * 15)
            for _ in range(batchCount):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Select a random half of data & label
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_data = X_train.loc[idx, :]
                labels = y.loc[idx]
                # Sample noise and generate a batch of new data
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_data = self.generator.predict([noise, labels])
                # Train the discriminator (real classified as ones and generated as zeros)]
                d_loss_real = self.discriminator.train_on_batch([real_data, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_data, labels], fake)
                # 기존 data와 fake data 학습에 대한 loss 평균
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Train Generator
                # ---------------------
				
                # fake data 생성을 위한 sample label
                sampled_labels = np.random.randint(0, 3, batch_size).reshape(-1, 1)
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # 특정 주기마다 기존 data와 fake data의 분포를 저장
            if epoch % save_interval == 0:
                self.save_dist(epoch, d_loss[0], g_loss)
```

#### 기존 data와 fake data의 분포 비교 

```python
    def save_dist(self, epoch, d_loss, g_loss):
        noise = np.random.normal(0, 1, (100000, self.latent_dim)) # noise 생성
        labels = np.ones(100000).reshape(-1, 1) # 특정한 label 지정 (여기서는 1)
        if d_loss <= 0.8 and g_loss <= 0.8:
            fake_data = self.generator.predict([noise, labels])
            fake_data = pd.DataFrame(fake_data, columns=data.columns)

            # fake_data에 대한 label
            add_label = [1] * 100000
            add_label = pd.DataFrame(add_label)
            y_resampled = np.ravel(add_label)

            X = data
            X_resampled = fake_data

            # 기존 data와 fake data의 분포를 시각화하기 위해 2차원으로 축소
            pca = PCA(n_components=2, random_state=42)
            # 차원축소된 기존 data 
            X_vis = pca.fit_transform(X)
			# 차원축소된 fake data
            X_res_vis = pca.transform(X_resampled)

            plt.figure(figsize=(15, 10))
            plt.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], color="r", label="Class #1", alpha=0.5)
            plt.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1], color="g", label="Generated Class #1", alpha=0.5)
            plt.tight_layout(pad=3)
            plt.legend()
            plt.savefig('CGAN/{}_epoch_plot.jpg'.format(epoch))


cgan = CGAN(data.shape[0], data.shape[1])
cgan.train(epochs=100, batch_size=128, save_interval=10)
```

