## OverSampling

데이터 클래스의 비율이 너무 차이가 나면 단순하게 우세한 클래스를 택하는 모형의 정확도가 높아지고 그로 인해 모형의 성능판별이 어려워짐 (정확도가 높아도 데이터의 개수가 적은 클래스의 재현율은 급격히 작아지는 현상이 발생)

이러한 비대칭 데이터 문제를 해결하기 위해 oversampling을 사용하여 데이터 클래스의 비율을 맞추면 정밀도가 향상됨

- #### RandomOverSampler

  - 소수 클래스의 데이터를 반복해서 추가
  - 소수 클래스 데이터들의 사본을 추가하는 것이기 때문에 오버피팅 발생 가능성을 높임

  <img src="https://user-images.githubusercontent.com/58063806/103352807-98b91080-4aea-11eb-82ed-029a663b2ef5.png" width=50% />

- #### SMOTE

  - 먼저 소수 클래스에서 임의의 데이터를 선택하고 해당 데이터에 대한 k- 최근 접 이웃을 설정, 그런 다음 임의의 데이터와 무작위로 선택된 k- 최근 접 이웃 사이에 합성 데이터를 생성 (소수 클래스의 데이터가 지정된 비율이 될 때 까지 반복) 
  - <img src="https://user-images.githubusercontent.com/58063806/103411943-3d545480-4bb6-11eb-99da-c91f63ac240e.png" width=50% />

  - [이미지 출처](https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5)

  <img src="https://user-images.githubusercontent.com/58063806/103411926-21e94980-4bb6-11eb-95f8-26f2c32e979c.png" width=50% />

- #### SMOTE-NC

  - 기존의 SMOTE는 연속적인 데이터에만 적용이 가능
  - SMOTE-NC는 범주형 데이터가 포함되어 있는 데이터에 적용가능
  - 범주형 피처가 몇 번째 열인지 명시하고 사용

- #### BorderlineSMOTE

  - SMOTE의 변형으로 두 클래스 간의 결정 경계를 따라 합성 데이터를 생성
  - 클래스 사이의 경계선 근처의 데이터들에 대해 합성 데이터를 생성함으로써 분류하기 더 어려운 부분에 집중
  - 소수 클래스에 속한 모든 데이터에 대해서 k- 최근접 이웃을 설정하고 설정된 **k-** **최근접 이웃들 중 절반 이상이 다수 클래스**인 데이터를 **DANGER**라고 하며 이는 **borderline**에 있는 **(분류하기 어려운 부분) 데이터**를 의미
  - DANGER 데이터들에 대해 k- 최근접 이웃들을 다시 설정 (이때는 소수 클래스에 속해 있는 데이터로만 설정)
  - 기존의 소수 클래스 데이터와 위에서 설정한 k- 최근접 이웃과 차이를 랜덤비율로 더함으로써 합성 데이터를 생성

  <img src="https://user-images.githubusercontent.com/58063806/103353087-3a406200-4aeb-11eb-94d7-82cebafa5846.png" width=50% />

- #### SVMSMOTE

  - Borderline-SMOTE의 변형으로 다른 SMOTE들과 차이점은 합성 데이터를 생성할 때 KNN과 SVM을 통합해서 사용한다는 것
  - Borderline-SMOTE에 비해 클래스 중복 영역에서 더 많은 데이터가 합성

  <img src="https://user-images.githubusercontent.com/58063806/103353172-75db2c00-4aeb-11eb-98cb-459366025551.png" width=50% />

- #### KMeansSMOTE

  - K-Means 알고리즘을 이용해 데이터를 군집화 시킴
  - 소수 클래스 데이터가 많이 속해 있는 클러스터에서 더 많은 합성 데이터를 생성하게 함
  - <img src="C:\Users\0864h\AppData\Roaming\Typora\typora-user-images\image-20201230221011262.png" width=60% />
  - [이미지 출처](https://www.researchgate.net/figure/K-means-SMOTE-oversamples-safe-areas-and-combats-within-class-imbalance-3_fig2_320821366)

  <img src="https://user-images.githubusercontent.com/58063806/103353305-dd917700-4aeb-11eb-8d0f-dfdc5f2b1bab.png" width=50% />

- #### ADASYN

  - SMOTE의 변형으로 데이터 밀도에 따라 합성 데이터를 생성
  - 소수 클래스가 덜 밀집된 부분에서 합성 데이터를 많이 생성하고 그렇지 않은 부분에서는 합성 데이터를 많이 생성하지 않음 (합성 데이터 생성은 소수 클래스의 밀도에 반비례)
  - 데이터가 **덜 밀집되어 있는 부분은 이상치 데이터**일 가능성이 존재하고 이런 경우에 **모델의 성능이 저하**됨 (이상치 값을 제거하고 사용하는 것이 바람직함)
  
  <img src="https://user-images.githubusercontent.com/58063806/103353202-92776400-4aeb-11eb-9d46-76c82e165492.png" width=50% />
