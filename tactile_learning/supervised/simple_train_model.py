"""
python train_model.py -t surface_3d
python train_model.py -t edge_2d
python train_model.py -t edge_3d
python train_model.py -t edge_5d
python train_model.py -t surface_3d edge_2d edge_3d edge_5d
"""

from tactile_learning.supervised.image_generator import ImageDataGenerator

import os
import time
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def simple_train_model(
    prediction_mode,
    model,
    label_encoder,
    train_data_dirs,
    val_data_dirs,
    csv_row_to_label,
    learning_params,
    image_processing_params,
    augmentation_params,
    save_dir,
    device='cpu'
):

    # tensorboard writer for tracking vars
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_runs'))

    # set generators and loaders
    train_generator = ImageDataGenerator(
        data_dirs=train_data_dirs,
        csv_row_to_label=csv_row_to_label,
        **{**image_processing_params, **augmentation_params}
    )
    val_generator = ImageDataGenerator(
        data_dirs=val_data_dirs,
        csv_row_to_label=csv_row_to_label,
        **image_processing_params
    )

    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # define optimizer and loss
    if prediction_mode == 'classification':
        loss = nn.CrossEntropyLoss()
    elif prediction_mode == 'regression':
        loss = nn.MSELoss()
    else:
        raise Warning("Incorrect prediction mode provided, falling back on MSEloss")
        loss = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params['lr'],
        weight_decay=learning_params['adam_decay']
    )

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        verbose=True
    )

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run_epoch(loader, n_batches, training=True):

        epoch_batch_loss = []
        epoch_batch_acc = []

        for batch in loader:

            # get inputs
            inputs, labels_dict = batch['images'], batch['labels']

            # wrap them in a Variable object
            inputs = Variable(inputs).float().to(device)

            # get labels
            labels = label_encoder.encode_label(labels_dict)

            # set the parameter gradients to zero
            if training:
                optimizer.zero_grad()

            # forward pass, backward pass, optimize
            outputs = model(inputs)
            loss_size = loss(outputs, labels)
            epoch_batch_loss.append(loss_size.item())

            if prediction_mode == 'classification':
                epoch_batch_acc.append((outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item())
            else:
                epoch_batch_acc.append(0.0)

            if training:
                loss_size.backward()
                optimizer.step()

        return epoch_batch_loss, epoch_batch_acc

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    # for saving best model
    lowest_val_loss = np.inf

    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            train_epoch_loss, train_epoch_acc = run_epoch(
                train_loader, n_train_batches, training=True
            )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc = run_epoch(
                val_loader, n_val_batches, training=False
            )
            model.train()

            # append loss and acc
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            # print metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch))
            print("Train Loss: {:.6f}".format(np.mean(train_epoch_loss)))
            print("Train Acc:  {:.6f}".format(np.mean(train_epoch_acc)))
            print("Val Loss:   {:.6f}".format(np.mean(val_epoch_loss)))
            print("Val Acc:    {:.6f}".format(np.mean(val_epoch_acc)))

            # write vals to tensorboard
            writer.add_scalar('loss/train', np.mean(train_epoch_loss), epoch)
            writer.add_scalar('loss/val', np.mean(val_epoch_loss), epoch)
            writer.add_scalar('accuracy/train', np.mean(train_epoch_acc), epoch)
            writer.add_scalar('accuracy/val', np.mean(val_epoch_acc), epoch)
            writer.add_scalar('learning_rate', get_lr(optimizer), epoch)

            # track weights on tensorboard
            for name, weight in model.named_parameters():
                full_name = f'{os.path.basename(os.path.normpath(save_dir))}/{name}'
                writer.add_histogram(full_name, weight, epoch)
                writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)

            # save the model with lowest val loss
            if np.mean(val_epoch_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_epoch_loss)

                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'best_model.pth')
                )

            # decay the lr
            lr_scheduler.step(np.mean(val_epoch_loss))

            # update epoch progress bar
            pbar.update(1)

    total_training_time = time.time() - training_start_time
    print("Training finished, took {:.6f}s".format(total_training_time))

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, 'final_model.pth')
    )


if __name__ == "__main__":
    pass
