import os
import shutil
import tensorflow as tf
import numpy as np
from termcolor import colored
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

# 디렉토리 세팅
def dir_setting(dir_name, CONTINUE_LEARNING):
    # os.getcwd() - 현재 디렉토리 경로
    cp_path = os.path.join(os.getcwd(), dir_name)
    confusion_matrix = os.path.join(cp_path, "confusion_matrix")
    model_path = os.path.join(cp_path, "model")

    if CONTINUE_LEARNING == False and os.path.isdir(cp_path):
        shutil.rmtree(cp_path)

    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)
        os.makedirs(confusion_matrix, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    path_dict = {'cp_path': cp_path,
                 'confusion_path': confusion_matrix,
                 'model_path': model_path}

    return path_dict


# load loss, acc metrics & make dictonary
def get_classification_metrics():
    train_loss = Mean()
    train_acc = SparseCategoricalAccuracy()

    validation_loss = Mean()
    validation_acc = SparseCategoricalAccuracy()

    test_loss = Mean()
    test_acc = SparseCategoricalAccuracy()

    metric_object = {}
    metric_object['train_loss'] = train_loss
    metric_object['train_acc'] = train_acc
    metric_object['validation_loss'] = validation_loss
    metric_object['validation_acc'] = validation_acc
    metric_object['test_loss'] = test_loss
    metric_object['test_acc'] = test_acc

    return metric_object


# CONTINUE_LEARNING이 True면 모델이 종료된 마지막 epoch의 정보들을 load 아니면 처음부터
def continue_setting(CONTINUE_LEARNING, path_dict, model=None):
    if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0:
        CONTINUE_LEARNING = False
        print(colored("CONTINUE LEARNING flag has been converted to FALSE", "cyan"))

    if CONTINUE_LEARNING:
        epoch_list = os.listdir(path_dict['model_path'])
        epoch_list = [int(epoch.split('_')[1]) for epoch in epoch_list]
        last_epoch = max(epoch_list)

        model_path = path_dict['model_path'] + '\epoch_' + str(last_epoch)
        model = tf.keras.models.load_model(model_path)

        losses_accs_path = path_dict['cp_path']
        losses_accs_np = np.load(losses_accs_path + "loss 파일이름")
        losses_accs = {}
        for k, v in losses_accs_np.items():
            losses_accs[k] = list(v)

        start_epoch = last_epoch + 1

    else:
        model = model
        start_epoch = 0
        losses_accs = {'train_losses': [], 'train_accs': [], 'validation_losses': [], 'validation_accs': []}

    return model, losses_accs, start_epoch
