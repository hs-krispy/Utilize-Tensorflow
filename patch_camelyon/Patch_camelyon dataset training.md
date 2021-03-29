## Patch_camelyon dataset training

- Train set - 262,144

- Validation set - 32,768

- Test set - 32,768
- 림프절 절편의 조직 병리학적 스캔에서 추출한 327,680 개의 컬러 이미지 (96 x 96px)로 구성
- 전이성 조직의 존재를 판별 (binary classification)

**Sample images**

<img src="https://user-images.githubusercontent.com/58063806/112849953-85ffa900-90e4-11eb-90ce-ab2e7f94eccf.png" width=50% />

### Train parameter

- AUTOTUNE = tf.data.experimental.AUTOTUNE
- buffer_size = 1000
- batch_size = 64
- epochs = 100
- Optimizer = Adam(lr=0.001)
- Loss = binary_crossentropy

#### LeNet

**Model layer & params**

<img src="https://user-images.githubusercontent.com/58063806/112850248-cced9e80-90e4-11eb-9a65-7a48f343d934.png" width=100% />



**result**

- **학습에 15시간 43분 소요 (RTX 2070 GPU 가용 시)**
- **Test accuracy : 83.35%**



#### DenseNet

**Model layer & params**

<img src="https://user-images.githubusercontent.com/58063806/112850561-16d68480-90e5-11eb-9a9f-f61039eb42cb.png" width=80% />

**result**

- **학습에 5시간 50분 소요 (RTX 2070 GPU 가용 시)**
- **Test accuracy : 84.59%**



***기존에  LeNet을 사용했을 때에 비해 DenseNet을 사용했을때 파라미터의 개수가 50배 넘게 줄어든 것과 더불어 학습시간도 10시간 가량 감소했고 test accuracy 또한 1% 이상 상승하는 모습을 보임***

