import os
import numpy as np
import matplotlib.pyplot as plt
from utils.learning_env_setting import dir_setting


# model과 loss, acc 저장
def save_metrics_model(epoch, model, losses_accs, path_dict, save_interval):
    if epoch % save_interval == 0:
        model.save(os.path.join(path_dict['model_path'], 'epoch_' + str(epoch)))
        np.savez_compressed(os.path.join(path_dict['cp_path'], 'losses_accs'), train_losses=losses_accs['train_losses'],
                            train_accs=losses_accs['train_accs'],
                            validation_losses=losses_accs['validation_losses'],
                            validation_accs=losses_accs['validation_accs'])


# loss, acc 변화 과정 시각화
def metric_visualizer(losses_accs, save_path):
    fig, ax = plt.subplots(figsize=(35, 15))
    ax2 = ax.twinx()

    epoch_range = np.arange(1, 1 + len(losses_accs['train_losses']))
    ax.plot(epoch_range, losses_accs['train_losses'], color="tab:blue", linestyle=":", linewidth=2, label="Train Loss")
    ax.plot(epoch_range, losses_accs['validation_losses'], color="tab:blue", label="Validation Loss")
    ax2.plot(epoch_range, losses_accs['train_accs'], color="tab:orange", linestyle=":", linewidth=2,
             label="Train Accuaracy")
    ax2.plot(epoch_range, losses_accs['validation_accs'], color="tab:orange", label="Validation Accuaracy")
    ax.legend(bbox_to_anchor=(1, 0.5), loc="upper right", frameon=False)
    ax2.legend(bbox_to_anchor=(1, 0.5), loc="lower right", frameon=False)

    plt.savefig(save_path + '/losses_accs_visualization.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()
