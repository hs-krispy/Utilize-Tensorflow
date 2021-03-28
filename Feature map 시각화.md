## Feature map 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = Sequential()
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, 3, activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.build(input_shape=(None, 150, 150, 3))
model.summary()

img = load_img("이미지 경로", target_size=(150, 150))
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
# (1, 150, 150, 3)

plt.imshow(img_tensor[0])
plt.show()
```

<img src="https://user-images.githubusercontent.com/58063806/112747279-473bf700-8fef-11eb-8729-52e9dbbe01c3.png" width=40%/>

```python
from tensorflow.keras.models import Model
layer_outputs = [layer.output for layer in model.layers[:8]]

# 입력 이미지를 넣으면 각 활성화 layer를 지난 후의 활성화 값을 반환
activation_model = Model(model.input, layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
# (1, 148, 148, 32)

plt.matshow(first_layer_activation[0, :, :, 19], cmap="viridis")
# 첫 번째 활성화 값 중 20번째 채널 시각화
```

<img src="https://user-images.githubusercontent.com/58063806/112747329-b31e5f80-8fef-11eb-80c0-9f9e17413846.png" width=40% />

```python
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

    scale = 1./size

    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect="auto", cmap="viridis")

plt.show()
```

**하위층에서 추출한 필터들**

<img src="https://user-images.githubusercontent.com/58063806/112747993-f7abfa00-8ff3-11eb-9528-8dfdbc5a106a.png" width=90%/>

- 초기 사진에 있는 거의 대부분의 정보가 유지

**상위층에서 추출한 필터들**

<img src="https://user-images.githubusercontent.com/58063806/112748094-95072e00-8ff4-11eb-9c42-70bb21be9fe8.png" width=90% />

- 상위층으로 갈수록 **활성화는 점점 추상적으로 되고 시각적으로 이해하기 어려워짐**

- '고양이 귀', '고양이 눈' 과 같이 **고수준의 개념을 인코딩**하기 시작

- 이미지의 **시각적 콘텐츠에 관한 정보가 줄어들고 이미지의 클래스에 관한 정보가 점점 증가** 

- 활성화 층이 깊어짐에 따라 활성화되지 않는 필터들이 생김

  > 필터에 인코딩된 패턴이 입력 이미지에 나타나지 않음 