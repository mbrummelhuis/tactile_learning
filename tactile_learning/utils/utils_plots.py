import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class LearningPlotter:
    def __init__(
        self,
        save_dir=None,
        max_epochs=None,
        plot_during_training=False,
        name="train_plot.png",
    ):
        self._max_epochs = max_epochs
        self._save_dir = save_dir
        self._name = name
        self.plot_during_training = plot_during_training

        if plot_during_training:
            plt.ion()
            self._fig, self._axs = plt.subplots(1, 2, figsize=(12, 4))

    def update(
        self,
        train_loss, val_loss,
        train_acc, val_acc
    ):
        for ax in self._axs.flat:
            ax.clear()

        # convert lists to arrays
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        train_acc = np.array(train_acc)
        val_acc = np.array(val_acc)

        if train_acc.size == 0:
            train_acc = np.zeros_like(val_acc)

        max_epochs = self._max_epochs if self._max_epochs is not None else train_loss.shape[0]
        n_epochs = train_loss.shape[0]
        r_epochs = np.arange(1, n_epochs+1)

        for loss, color in zip([train_loss, val_loss], ['r', 'b']):
            lo_bound = np.clip(loss.mean(axis=1) - loss.std(axis=1), loss.min(axis=1), loss.max(axis=1))
            up_bound = np.clip(loss.mean(axis=1) + loss.std(axis=1), loss.min(axis=1), loss.max(axis=1))
            self._axs[0].plot(r_epochs, loss.mean(axis=1), color=color, alpha=1.0)
            self._axs[0].fill_between(r_epochs, lo_bound, up_bound, color=color, alpha=0.25)

        self._axs[0].set_yscale('log')
        self._axs[0].set_xlabel('Epoch')
        self._axs[0].set_ylabel('Loss')

        for acc, color in zip([train_acc, val_acc], ['r', 'b']):
            lo_bound = np.clip(acc.mean(axis=1) - acc.std(axis=1), acc.min(axis=1), acc.max(axis=1))
            up_bound = np.clip(acc.mean(axis=1) + acc.std(axis=1), acc.min(axis=1), acc.max(axis=1))
            self._axs[1].plot(r_epochs, acc.mean(axis=1), color=color, alpha=1.0)
            self._axs[1].fill_between(r_epochs, lo_bound, up_bound, color=color, alpha=0.25, label='_nolegend_')

        self._axs[1].set_xlabel('Epoch')
        self._axs[1].set_ylabel('Accuracy')
        plt.legend(['Train', 'Val'])

        if self._save_dir is not None:
            save_file = os.path.join(self._save_dir, self._name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        for ax in self._axs.flat:
            ax.relim()
            ax.set_xlim([0, max_epochs])
            ax.autoscale_view(True, True, True)

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        train_loss, val_loss,
        train_acc, val_acc
    ):
        if not self.plot_during_training:
            self._fig, self._axs = plt.subplots(1, 2, figsize=(12, 4))

        self.update(
            train_loss, val_loss,
            train_acc, val_acc
        )
        plt.show()


if __name__ == '__main__':
    pass
