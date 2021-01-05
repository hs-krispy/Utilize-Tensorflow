## GAN

- GAN(Generative Adversarial Network)은 생성기(generator)와 판별기(discriminator)로 구성
- 생성기는 데이터를 위조하며 판별기는 진짜 데이터에 기반을 두고 위조된 데이터가 얼마나 진짜 같은지를 판별
- 생성기와 판별기는 교대로 훈련하고, 각각의 목표는 경사하강을 통해 최적화된 손실 함수로 표현 **(적대적 훈련 - 생성 모델은 지속적으로 위변조 능력을 향상, 판별 모델은 위변조 인식 기능을 향상)**
  - 판별기에서 생성기가 생성한 위조 데이터와 훈련 데이터셋을 받아서 지도학습을 통해 가짜와 진짜를 구분하기 위해 학습, 데이터가 훈련 데이터셋 분포라면 판별기는 입력 데이터가 진짜일 때는 1에 가깝게, 가짜일 때는 0에 가깝게 만듬
  - 생성기는 판별기의 결과가 1에 가깝게 하도록 만듬 
  - 위의 두 단계가 순차적으로 반복

<img src="https://user-images.githubusercontent.com/58063806/103435075-3af10980-4c4d-11eb-8200-d5f5b302abdb.png" width=70% />

- **실제 데이터의 분포와 모델이 생성한 데이터의 분포 사이의 차이를 줄이는 것**

<img src="https://user-images.githubusercontent.com/58063806/103655223-496f5600-4faa-11eb-8e29-f14b2966b5e5.png" width=70% />

검은색 점선 - 실제 데이터의 분포

파란색 점선 - 판별기 분포 (계속 학습을 진행하다보면 나중에는 0.5로 수렴)

초록색 선 - 위조 데이터의 분포 (실제 데이터의 분포와 유사하도록 해야함)

[이미지 출처](https://yamalab.tistory.com/98)



### MNIST with GAN

```python
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

randomDim = 10
adam = Adam(lr=0.0002, beta_1=0.5)
dLosses = []
gLosses = []
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(60000, 784)


# 생성기와 판별기의 에폭 당 손실값을 시각화
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# 생성기에 의해 생성된 MNIST 이미지
def saveGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')

        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)


def train(epochs=1, batchSize=128):
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs + 1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batchCount):
            # 랜덤 입력 노이즈와 이미지
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # 가짜 MNIST 이미지 생성
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # 생성된 것과 실제 이미지의 레이블
            yDis = np.zeros(2 * batchSize)
            # 편파적 레이블 평활화 (의도적으로 1이 아닌 0.9로 설정, 너무 확실하게 설정하는 것을 방지)
            yDis[:batchSize] = 0.9

            # 판별기 훈련
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # 생성기 훈련
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        dLosses.append(dloss)
        gLosses.append(gloss)
		
        if e == 1 or e % 20 == 0:
            saveGeneratedImages(e)


generator = Sequential()
generator.add(Dense(256, input_dim=randomDim))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(784, activation='tanh'))

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# 생성기와 판별기를 결합해 GAN을 구성
discriminator.trainable = False # 판별기의 가중치를 동결 (변경되지 않게 함)
ganInput = Input(shape=(randomDim, ))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

train(40, 128)
```

- GAN의 경우, 판별기를 학습시키때 마다 생성기가 생성하는 데이터가 변화하게 됨
- 처음부터 모든 데이터가 존재하고 이를 한번에 학습시키는 fit과는 다르게, 한번씩 업데이트를 할때마다 모델이 변화하므로 train_on_batch를 사용

**epoch 1**

<img src="https://user-images.githubusercontent.com/58063806/103513071-36c32700-4ead-11eb-8f34-dbac6ff343cb.png" width=60% />

**epoch 20**

<img src="https://user-images.githubusercontent.com/58063806/103514981-19905780-4eb1-11eb-894f-a1d0e0e2f0fb.png" width=60%/>

**epoch 40**

<img src="https://user-images.githubusercontent.com/58063806/103516419-dc799480-4eb3-11eb-82ca-7299a2d0006d.png" width=60% />

위의 결과들로 볼 때 epoch이 증가함에 따라 생성기에서 생성된 숫자 이미지가 실제와 점점 더 유사해지는 것을 볼 수 있음 