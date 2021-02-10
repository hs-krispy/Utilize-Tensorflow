## LSGAN

#### (Least Squares Generative Adversarial Networks)

- 기존의 GAN에서 binary crossentropy를 대신해 mse를 손실함수로 사용
- 기존의 data와 멀리 떨어져있는 data에 대해 penalty를 부여 (실제 데이터의 분포와 유사하도록 만드는데 도움)
- 기존의 GAN에 비해 학습이 더 안정적(기울기 손실 문제를 해결)이고 좋은 성능을 보임
- mse를 손실함수로 사용하므로 discriminator의 output에 sigmoid를 적용하지 않음
- Batch Normalization 없이도 비교적 안정적으로 학습

  - 개인적으로 실험해본 결과 **Generator에만 Batch Normalization을 적용하는 것이 가장 좋은 결과**를 보임

- mode collapse 문제를 줄임
- optimizer로는 RMSprop 사용 권장 (Adam에 비해 학습이 더 안정적으로 이루어짐)
  - 일반적으로 극한값이 존재하지 않는 nonstationary 문제에서는 momentum 계열보다 RMSprop의 성능이 더 좋다고 함

<img src="https://user-images.githubusercontent.com/58063806/107394277-0b33fc00-6b3f-11eb-8e65-9ea3409908f9.png" width=80% />

<img src="https://user-images.githubusercontent.com/58063806/107394380-2868ca80-6b3f-11eb-961e-32950587cec5.png" width=80% />

<img src="https://user-images.githubusercontent.com/58063806/107394454-41717b80-6b3f-11eb-9cf5-4611dc46b132.png" width=80% />

<img src="https://user-images.githubusercontent.com/58063806/107394531-577f3c00-6b3f-11eb-8a95-21de68a83c8d.png" width=80% />

<img src="https://user-images.githubusercontent.com/58063806/107394804-957c6000-6b3f-11eb-9778-b0cc9dd46e2a.png" width=80% />

실험결과 실제로 Optimizer로 RMSprop를 사용했을 때 학습이 더 안정적이고 좋은 결과가 나오는 것을 볼 수있음 

#### 생성된 data를 이용한 성능측정 & 비교

기존 data의 class 분포

<img src="https://user-images.githubusercontent.com/58063806/107393993-bbedcb80-6b3e-11eb-8071-938f5b3de0c7.png" width=15% />

LSGAN + CGAN을 이용해 class 1의 data를 100000개 생성

```python
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=500, n_jobs=-1, random_state=42)
sampler = SMOTE(random_state=42)
scaler = MinMaxScaler()
data = pd.read_csv('train.csv')
# 기존 data shape - (320000, 18)
additional_data = pd.read_csv('additional_data.csv')
y = data['class']
print(y.value_counts())
data.drop(columns=["id", "class"], inplace=True)
data = scaler.fit_transform(data)


def cross_val(x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = []
    for train_idx, valid_idx in skf.split(x, y):
        train_x, valid_x = x[train_idx], x[valid_idx]
        train_y, valid_y = y[train_idx], y[valid_idx]
        # train_x, train_y = sampler.fit_resample(train_x, train_y)
        
        # LSGAN 생성 데이터
        # train_x, valid_x = x.iloc[train_idx], x.iloc[valid_idx]
        # train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]
        # select = np.random.randint(0, 100000, 80000)
        # train_x = pd.concat([train_x, additional_data.iloc[select]], axis=0, ignore_index=True)
        # train_y = pd.concat([train_y, additional_y.iloc[select]], axis=0, ignore_index=True)
        # train_x, train_y = shuffle(train_x, train_y)
		train_y = np.ravel(train_y)
        valid_y = np.ravel(valid_y)
        evals = [(valid_x, valid_y)]
        clf.fit(train_x, train_y, early_stopping_rounds=30, eval_set=evals)
        accuracy = clf.score(valid_x, valid_y)
        score.append(accuracy)
    print(np.mean(score))

    
data = pd.DataFrame(data, columns=additional_data.columns)
y = pd.DataFrame(y)
cross_val(data, y)
```

**Accuracy score & 각 fold 별 class 비율**

```python
기존 data - 0.9274437500000001
```

<img src="https://user-images.githubusercontent.com/58063806/107471728-ea0bf380-6bb0-11eb-8702-e9e06e330bb1.png" width=15% />

```python
SMOTE를 이용해 oversampling한 data - 0.9099031249999999
```

<img src="https://user-images.githubusercontent.com/58063806/107475516-9f41aa00-6bb7-11eb-8f6b-11b5113bd392.png" width=15% />

```python
LSGAN 생성 데이터를 이용해 oversampling한 data - 0.9271750000000001
```

<img src="https://user-images.githubusercontent.com/58063806/107474765-4ae9fa80-6bb6-11eb-973d-de043fb07472.png" width=15% />

- 해당 조건에서는 기존의 data의 acc가 가장 높았고 그 다음으로 LSGAN, SMOTE 순 

- LSGAN의 학습 정도와 data에 따라 결과가 상이할 것 같지만 그래도 LSGAN을 이용한 oversampling이 어느 정도 효과가 있는 것을 확인

#### mode collapse

<img src="https://user-images.githubusercontent.com/58063806/107367565-cf8b3900-6b22-11eb-85c0-1f7ca0c32ee7.png" width=60% />

[이미지 출처](https://arxiv.org/pdf/1611.04076.pdf)

- 위의 그림과 target의 분포를 학습하는 과정에서 특정 부분의 분포만 학습되는 문제

  - EX) MNIST의 학습과정 중 1에 해당하는 이미지에 대해서만 학습

- GAN으로 만든 generated image의 질을 정량적으로 측정할 수 있는 방법이 마땅치 않음

- GAN의 training loss를 계산하는 것이 intractable하기 때문에 발생

  

