## data API

- 학습시킬 데이터의 크기가 클때 메모리를 효과적으로 사용할 수 있도록 함

#### from_tensor_slice

data를 TensorSliceDataset으로 변환 **(data, labels)의 형태**

```python
from tensorflow.data import Dataset

train_data = Dataset.from_tensor_slices((X_train, y_train))
```

#### map, cache

**data preprocessing을 위함**

```python
def preprocess(image, labels):
    image = tf.cast(image, tf.float32)
    image /= 255.

    return image, labels

train_data = train_data.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
```

> num_parallel_calls - 병렬 처리를 위한 parameter
>
> 수동으로 값을 설정하거나 tf.data.experimental.AUTOTUNE으로 값을 설정하면  런타임이 실행 시에 동적으로 값을 조정

cache - 더 빠른 preprocessing을 위함

>  file open, data read와 같은 작업들은 첫 번째 epoch 동안에만 실행되고 다음 epoch부터는 cache된 data를 재사용

#### shuffle, prefetch, batch

```python
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
```

shuffle - data를 얼마나 shuffling 할지 (전체 data 크기 이상의 값으로 설정해야 전체 data가 모두 shuffling 됨)

batch - 해당 size 만큼 data를 가져옴

prefetch -  이전의 data가 학습 중일때 다음 data를 미리 load (마찬가지로 tf.data.experimental.AUTOTUNE로 값을 설정하면 런타임이 동적으로 값을 설정)

#### repeat

dataset을 계속 반복해서 생성

#### take, skip

```python
train_data.take(n)
new_data = train_data.skip(n)
```

take - n개의 data만 불러옴 (train_data[:n])

skip - train_data의 앞 n개의 데이터를 스킵 (new_data = train_data[n:])

#### flow_from_directory

- 제너레이터 생성 (iterator 처럼 작동)
- 폴더명에 맞춰 자동으로 labeling
- 전체 데이터를 한번에 불러오는 것이 아니므로 메모리 관리에 효율적

```python
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode="binary")
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode="binary")

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50, verbose=1)
```

제너레이터는 배치를 무한정 만들어내기 때문에 fit_generator 시에 **steps_per_epoch으로 한 epoch에 batch를 몇 번 불러올 것**인지 설정 (validation_steps도 같은 맥락)

> fit_generator는 multiprocessing이 가능