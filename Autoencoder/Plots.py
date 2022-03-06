import matplotlib.pyplot as plt


def plot_loss(train, val, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train, 'b', color='deepskyblue', label=label1)
    plt.plot(epochs, val, 'b', color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.legend()
    plt.savefig(f'{save_dir}/bce_and_mse_loss.pdf')
    plt.close()


def plot_auc(train_auc, val_auc, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train_auc, color='deepskyblue', label=label1)
    plt.plot(epochs, val_auc, color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(f'{save_dir}/auc.pdf')
    plt.close()


def plot_r2(train_r2, val_r2, num_epochs, save_dir):
    epochs = [e for e in range(1, num_epochs + 1)]
    label1 = 'Train'
    label2 = 'Validation'
    plt.figure(2)
    plt.plot(epochs, train_r2, color='deepskyblue', label=label1)
    plt.plot(epochs, val_r2, color='orange', label=label2)
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.savefig(f'{save_dir}/r2.pdf')
    plt.close()