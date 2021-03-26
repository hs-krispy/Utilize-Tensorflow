import os
import tensorflow as tf
from utils.learning_env_setting import dir_setting, continue_setting, get_classification_metrics
from utils.dataset_utils import get_ds
from utils.train_validation_test import train, validation, test
from utils.cp_utils import save_metrics_model, metric_visualizer
from utils.basic_utils import resetter, training_reporter
from models import LeNet5
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# GPU 적정량만 사용
os.environ['TF_FORGE_GPU_ALLOW_GROWTH'] = 'true'

exp_name = "LeNet5_train1"
CONTINUE_LEARNING = False

train_ratio = 0.8
train_batch_size, test_batch_size = 128, 128

epochs = 10
save_interval = 2
learning_rate = 0.01

model = LeNet5()
optimizer = SGD(learning_rate=learning_rate)

loss_object = SparseCategoricalCrossentropy()
path_dict = dir_setting(exp_name, CONTINUE_LEARNING)
model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)
train_ds, validation_ds, test_ds = get_ds(train_ratio, train_batch_size, test_batch_size, "mnist")
metric_objects = get_classification_metrics()

for epoch in range(start_epoch, epochs):
    train(train_ds, model, loss_object, optimizer, metric_objects)
    validation(validation_ds, model, loss_object, metric_objects)

    training_reporter(epoch, losses_accs, metric_objects)
    save_metrics_model(epoch, model, losses_accs, path_dict, save_interval)

    metric_visualizer(losses_accs, path_dict['cp_path'])
    resetter(metric_objects)

test(test_ds, model, loss_object, metric_objects, path_dict)