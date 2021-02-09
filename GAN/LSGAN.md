## LSGAN

#### (Least Squares Generative Adversarial Networks)

- 기존의 GAN에서 binary crossentropy를 대신해 mse를 손실함수로 사용
- 기존의 data와 멀리 떨어져있는 data에 대해 penalty를 부여 (실제 데이터의 분포와 유사하도록 만드는데 도움)
- 기존의 GAN에 비해 학습이 더 안정적(기울기 손실 문제를 해결)이고 좋은 성능을 보임
- mse를 손실함수로 사용하므로 discriminator의 output에 sigmoid를 적용하지 않음
- Batch Normalization 없이도 비교적 안정적으로 학습
- mode collapse 문제를 줄임

#### mode collapse

<img src="https://user-images.githubusercontent.com/58063806/107367565-cf8b3900-6b22-11eb-85c0-1f7ca0c32ee7.png" width=60% />

[이미지 출처](https://arxiv.org/pdf/1611.04076.pdf)

- 위의 그림과 target의 분포를 학습하는 과정에서 특정 부분의 분포만 학습되는 문제

  - EX) MNIST의 학습과정 중 1에 해당하는 이미지에 대해서만 학습

- GAN으로 만든 generated image의 질을 정량적으로 측정할 수 있는 방법이 마땅치 않음

- GAN의 training loss를 계산하는 것이 intractable하기 때문에 발생

  

개인적으로 실험해본 결과

Generator에만 Batch Normalization을 적용하는 것이 가장 좋은 결과를 보임