import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class LearningPlotter:
    def __init__(
        self,
        save_dir=None,
        max_epochs=None,
        name="train_plot.png",
        plot_during_training=False,
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


class ClassificationPlotter:
    def __init__(
        self,
        class_names,
        save_dir=None,
        name="error_plot.png",
        plot_during_training=False,
        normalize=True
    ):
        self.class_names = class_names
        self.save_dir = save_dir
        self.name = name
        self.plot_during_training = plot_during_training
        self.normalize = normalize

        if plot_during_training:
            plt.ion()
            plt.figure()
            self._fig = plt.gcf()
            self._fig.set_size_inches((12, 12), forward=False)

    def update(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):

        self._fig.gca().clear()

        cm = metrics['conf_mat']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=90, fontsize=12)
        plt.yticks(tick_marks, self.class_names, fontsize=12)

        fmt = '.2f' if self.normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)

        plt.tight_layout()
        plt.xlabel('Target class', fontsize=16, fontweight='bold')
        plt.ylabel('Predicted class', fontsize=16, fontweight='bold')

        if self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        if not self.plot_during_training:
            plt.figure()
            self._fig = plt.gcf()
            self._fig.set_size_inches((12, 12), forward=False)

        self.update(pred_df, targ_df, metrics)
        plt.show(block=True)


class RegressionPlotter:
    def __init__(
        self,
        task_params,
        save_dir=None,
        name="error_plot.png",
        plot_during_training=False,
        plot_interp=True
    ):
        self.target_label_names = list(filter(None, task_params['target_label_names']))
        self.save_dir = save_dir
        self.name = name
        self.plot_during_training = plot_during_training
        self.plot_interp = plot_interp

        self.n_plots = len(self.target_label_names)
        self.n_rows = int(np.ceil(self.n_plots/3))
        self.n_cols = np.minimum(self.n_plots, 3)

        if self.n_plots == 2 or self.n_plots == 5:
            self.target_label_names.insert(2, None)

        if plot_during_training:
            plt.ion()
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols,
                                                figsize=(4*self.n_cols, 3.5*self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)

    def update(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        for ax in self._axs.flat:
            ax.clear()

        n_smooth = int(pred_df.shape[0] / 20)

        err_df = metrics['err']  ## BUG does not handle None default

        for ax, label_name in zip(self._axs.flat, self.target_label_names):
            if label_name:

                targ_df = targ_df.sort_values(by=label_name)

                pred_df = pred_df.assign(temp=targ_df[label_name])
                pred_df = pred_df.sort_values(by='temp')
                pred_df = pred_df.drop('temp', axis=1)

                if isinstance(err_df, pd.DataFrame):
                    err_df = err_df.assign(temp=targ_df[label_name])
                    err_df = err_df.sort_values(by='temp')
                    err_df = err_df.drop('temp', axis=1)

                    ax.scatter(
                        targ_df[label_name].astype(float),
                        pred_df[label_name].astype(float),
                        s=1, c=err_df[label_name], cmap="inferno"
                    )

                else:
                    ax.scatter(
                        targ_df[label_name].astype(float),
                        pred_df[label_name].astype(float), s=1, c='k'
                    )

                if self.plot_interp:
                    ax.plot(
                        targ_df[label_name].astype(float).rolling(n_smooth).mean(),
                        pred_df[label_name].astype(float).rolling(n_smooth).mean(),
                        linewidth=2, c='r'
                    )

                ax.set(xlabel=f"target {label_name}", ylabel=f"predicted {label_name}")
                xlim = (
                    np.round(targ_df[label_name].astype(float).min()),
                    np.round(targ_df[label_name].astype(float).max())
                )
                xticks = ax.get_xticks()
                ax.set_xticks(xticks), ax.set_yticks(xticks)
                ax.set_xlim(*xlim), ax.set_ylim(*xlim)

                if isinstance(err_df, pd.DataFrame):
                    ax.text(0.05, 0.9, 'MAE = {:.4f}'.format(err_df[label_name].mean()), transform=ax.transAxes)
                ax.grid(True)

            else:
                ax.axis('off')

        if self.save_dir is not None:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        plt.pause(0.01)

    def final_plot(
        self,
        pred_df,
        targ_df,
        metrics=None,
    ):
        if not self.plot_during_training:
            plt.ion()
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols,
                                                figsize=(4*self.n_cols, 3.5*self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)

        self.update(pred_df, targ_df, metrics)
        # plt.show()


if __name__ == '__main__':
    pass
