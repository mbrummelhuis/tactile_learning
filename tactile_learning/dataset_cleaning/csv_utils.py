import os
import pathlib
import pandas as pd
import cv2
import numpy as np

from tactile_learning.utils.utils_learning import make_dir


def check_images_exist(base_dir):

    # load target df
    targets_df = pd.read_csv(os.path.join(base_dir, 'targets.csv'))

    images_not_found = []

    for row in targets_df.iterrows():
        img_name = row[1]['sensor_image']
        print(f'checking {base_dir}: {img_name}')

        infile = os.path.join(base_dir, 'images', img_name)
        image = cv2.imread(infile)

        if image is None:
            images_not_found.append(img_name)
            print('Image not found')

    if images_not_found == []:
        print('Dataset is complete')
    else:
        print('Images not found: ', images_not_found)


def adjust_csv(base_dir, dry_run=True):

    def adjust_filename(video_filename):
        video_filename = pathlib.Path(video_filename).stem
        id = video_filename.split('_')[1]
        return f'image_{id}.png'

    target_df = pd.read_csv(os.path.join(base_dir, 'targets_video.csv'))
    target_df['sensor_image'] = target_df.sensor_video.apply(adjust_filename)
    target_df.drop('sensor_video', axis=1, inplace=True)
    print(target_df)

    if not dry_run:
        target_df.to_csv(os.path.join(base_dir, 'targets.csv'), index=False)


def partition_dataset(base_dir, dry_run=True):

    # make deterministic
    np.random.seed(0)

    # define split
    indir_name = "data"
    outdir_names = ["train", "val"]
    split = 0.8

    # load target df
    targets_df = pd.read_csv(os.path.join(base_dir, indir_name, 'targets.csv'))

    # Select data
    inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
    inds = [inds_true, ~inds_true]

    # iterate over split
    for outdir_name, ind in zip(outdir_names, inds):

        indir = os.path.join(base_dir, indir_name)
        outdir = os.path.join(base_dir, outdir_name)

        # check save dir exists
        if not dry_run:
            make_dir(outdir)
            image_dir = os.path.join(outdir, "images")
            os.makedirs(image_dir, exist_ok=True)

        for img_name in targets_df[ind].raw_image_name:
            print(f'processed {outdir_name}: {img_name}')
            infile = os.path.join(indir, 'images', img_name)
            outfile = os.path.join(outdir, 'images', img_name)
            img = cv2.imread(infile)

            if not dry_run:
                cv2.imwrite(outfile, img)

        # save targets
        if not dry_run:
            targets_df[ind].to_csv(os.path.join(outdir, 'targets.csv'), index=False)


if __name__ == '__main__':

    data_path = os.path.join(
        "/home/alex/tactile_datasets/braille_classification/tactip_331_25mm/alphabet/train"
    )
    dry_run = True

    check_images_exist(base_dir=data_path)
