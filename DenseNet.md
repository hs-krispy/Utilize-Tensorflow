## DenseNet

- 정보의 흐름을 최대로 하는것을 보장하기 위해서, 모든 layer들을 연결
- 각각의 layer들은 모든 이전 layer들로 부터 추가적인 input을 받음
- Dense connection은 regularize 효과가 있어, overffiting을 감소시킴
- 기존 방법들에 비해 feature map에 대한 불필요한 학습이 필요가 없기 때문에, parameter의 수가 적음
- 각각의 layer들은 loss function과 input signal로 부터의 gradient에 직접 접근할 수 있어서, 더 쉬운 학습이 가능



#### model architecture

<img src="https://user-images.githubusercontent.com/58063806/111280364-15d23b80-867f-11eb-94b7-991da8b49b28.png" width=90% />



#### Dense connectivity

<img src="https://user-images.githubusercontent.com/58063806/111280119-cbe95580-867e-11eb-87e0-1869e3e9da52.png" width=55% />

- 후속 계층들은 이전의 feature map들과 concatenate



#### Composite function

- BN(Batch Normalization) - ReLU - Conv 순서로 이루어진 **full pre-activation 구조**를 사용 



#### Transition Layer

- feature map의 크기를 다운샘플링 해주는 layer
- BN - ReLU - 1 x 1 Conv - 2 x 2 AvgPooling 순서로 이루어짐
- 마지막 Dense Block을 제외한 나머지 Dense Block 뒤에 위치
- theta의 값(0 ~ 1)으로 feature map의 개수를 조정
  - 논문에서는 0.5의 값을 사용 (feature map의 개수를 절반으로)  



#### Growth rate(k)

- 네트워크 성장률이라고 하며 각 계층이 글로벌 상태에 기여하는 새로운 정보의 양을 조절
- 상대적으로 적은 성장률로 좋은 결과를 얻을 수 있음



#### Bottleneck layers

<img src="https://user-images.githubusercontent.com/58063806/111284003-0b19a580-8683-11eb-97e3-4f5505888bb4.png" width=40% />

- 입력 feature map의 수를 줄이기 위해 3 x 3 Conv 전에 1 x 1 Conv을 추가
- 1 x 1 Conv에서 4k의 피처 맵을 생성하도록 함



내용 참고 및 이미지 출처

[https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)

[https://hoya012.github.io/blog/DenseNet-Tutorial-1/](https://hoya012.github.io/blog/DenseNet-Tutorial-1/)