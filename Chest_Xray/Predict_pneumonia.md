## Predict_pneumonia

[Download dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

정상적인 폐와 폐렴이 걸린 폐의 이미지를 분류



#### 데이터 살펴보기

각 데이터 셋의 크기

<img src="https://user-images.githubusercontent.com/58063806/102012850-ef56e880-3d8f-11eb-8bbb-74f981196d37.png" width=20% />

각 데이터 셋에서 정상과 폐렴의 데이터 개수

<img src="https://user-images.githubusercontent.com/58063806/102012876-0bf32080-3d90-11eb-8de5-1dc6e7c8cb49.png" width=30% />

각 데이터 셋의 정상과 폐렴 데이터 이미지 (왼쪽부터 train, val, test set, 상단이 정상인 경우)

<img src="https://user-images.githubusercontent.com/58063806/102012901-32b15700-3d90-11eb-8d90-4635caffc971.png" width=90% />

#### 데이터 증폭

```python
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, brightness_range=0.5, zoom_range=0.5)
train_data = generator.flow(X_train, y_train, batch_size=batch_size)
val_data = generator.flow(X_val, y_val, batch_size=batch_size)
test_data = generator.flow(X_test, y_test, batch_size=batch_size)
```

keras의 ImageDataGenerator를 이용해서 데이터의 개수를 증폭

- rotation_range - 지정된 각도 범위내에서 임의로 원본이미지를 회전
- width_shift_range - 지정된 수평방향 범위내에서 임의로 원본이미지를 이동
- height_shift_range - 지정된 수직방향 이동 범위내에서 임의로 원본이미지를 이동
- zoom_range - 지정된 확대/축소 범위내에서 임의로 원본이미지를 확대/축소 (위의 경우는 0.5 ~ 1.5)
- brightness_range - 지정된 밝기 범위내에서 임의로 원본이미지의 밝기를 변화 (위의 경우는 0.5 ~ 1.5)
- [이 외 parameters](https://keras.io/ko/preprocessing/image/)
- evaluate_generator()를 통해서 데이터 생성기 상에서 모델을 평가

**새로 생성된 이미지**

```python
fig = plt.figure(figsize=(30, 30))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1)
    ax.imshow(train_data.next()[0][i].astype('uint8'), cmap="gray")

plt.savefig("generated image.jpg")
```

<img src="https://user-images.githubusercontent.com/58063806/102013618-a81f2680-3d94-11eb-98f7-9c3e6424b073.png" width=80% />

#### callbacks

```python
callbacks = [EarlyStopping(monitor='loss', patience=5), 
             ModelCheckpoint(filepath='best_model.h5', verbose=1, monitor='loss', save_best_only=True)]

history = model.fit_generator(train_data, steps_per_epoch=len(X_train) // batch_size, epochs=epoch, verbose=1, validation_data=val_data, callbacks=callbacks)
# history = model.fit(train_data, steps_per_epoch=len(X_train) // batch_size, epochs=epoch, verbose=1, class_weight={0:3.0, 1:1.0}, validation_data=(X_val, y_val), callbacks=callbacks)
```

EarlyStopping - **기준 지표(위에서는 loss)를 대상**으로 **지정된 epoch(위에서는 5) 동안 개선이 없으면 학습을 종료**시킴

ModelCheckpoint - 기준 지표의 값(위에서는 loss)이 이전 epoch에 비해 개선된 경우 filepath에 해당하는 경로에 모델을 저장함 (그러므로 학습이 중지되었을때 기준 지표에서 가장 좋은 성능의 모델을 반환가능)

### Image transform

**Bilateral filter**

```python
img = image.load_img(img_path, color_mode="grayscale", target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = cv2.bilateralFilter(img_tensor, d=-1, sigmaColor=15, sigmaSpace=15)
        img_tensor = img_tensor.reshape(150, 150, 1)
```

이미지를 좀 더 부드럽게 만들고 동시에 폐의 크기와 혼탁한 부분에 대한 경계를 좀 더 잘 나타낼 수 있도록 이미지를 스무딩 해주면서 경계는 보존하는 Bilateral filter를 적용

<img src="https://user-images.githubusercontent.com/58063806/102227143-91133c80-3f2c-11eb-9868-e4c7d0a2296d.png" width=90% />

parameter에 따라 차이는 있지만 전반적으로 원본 이미지에 비해 경계가 잘 나뉘어 진 것을 볼 수 있음

**Crop**

```python
img = image.load_img(img_path, color_mode="grayscale", target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor[40: 130, 50: 130]
```

원본 이미지에서 폐렴을 판별하는데에 필요하다고 생각되는 임의의 영역만 잘라냄

<img src="https://user-images.githubusercontent.com/58063806/102228760-66c27e80-3f2e-11eb-82d1-fa0415b607ab.png" width=90% />