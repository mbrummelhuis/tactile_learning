import numpy as np
import os
import pandas as pd
import torch

from tactile_learning.supervised.image_generator import numpy_collate


class PinDataGenerator(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dirs,
        csv_row_to_label,
    ):

        # check if data dirs are lists
        assert isinstance(data_dirs, list), "data_dirs should be a list!"

        self._csv_row_to_label = csv_row_to_label

        # load csv file
        self._label_df = self.load_data_dirs(data_dirs)

    def load_data_dirs(self, data_dirs):

        # add collumn for which dir data is stored in
        df_list = []
        for data_dir in data_dirs:
            df = pd.read_csv(os.path.join(data_dir, 'targets.csv'))

            # check for a processed image dir first
            keypoints_dir = os.path.join(data_dir, 'extracted_pins')

            df['keypoints_dir'] = keypoints_dir
            df_list.append(df)

        # concat all df
        full_df = pd.concat(df_list)

        return full_df

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._label_df)))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        row = self._label_df.iloc[index]
        keypoints_filename = os.path.join(row['keypoints_dir'], row['keypoints_filename'])
        keypoints = np.load(keypoints_filename)

        # get label
        target = self._csv_row_to_label(row)
        sample = {'inputs': keypoints, 'labels': target}

        return sample


def demo_pin_generation(
    data_dirs,
    csv_row_to_label,
    learning_params
):

    # Configure dataloaders
    generator = PinDataGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu'],
        collate_fn=numpy_collate
    )

    # iterate through
    for (i_batch, sample_batched) in enumerate(loader, 0):

        # shape = (batch, n_frames, width, height)
        inputs = sample_batched['inputs']
        labels = sample_batched['labels']

        for i in range(inputs.shape[0]):
            for key, item in labels.items():
                print(key, ': ', item[i])

            print('')
            print('Extracted Keypoints: ', inputs.shape)
            print('')
