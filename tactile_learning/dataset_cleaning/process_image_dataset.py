import os
from glob import glob
import pathlib
import cv2

from tactile_image_processing.image_transforms import process_image
from tactile_learning.utils_learning import make_dir, save_json_obj

data_path = os.path.join(
    # "/home/alex/tactile_datasets/tactile_servo_control/tactip_127"
    "/home/alex/tactile_datasets/braille_classification/tactip_331_25mm"
)


def process_dataset(
    base_dir,
    image_processing_params,
    dry_run=True
):

    image_dir = os.path.join(
        base_dir,
        'images'
    )

    all_image_files = [y for x in os.walk(
        image_dir
    ) for y in glob(os.path.join(x[0], '*.png'))]

    cv2.namedWindow("proccessed_image")

    # make new dir for saving images
    new_dir_name = os.path.join(
        base_dir,
        'processed_images',
    )

    if not dry_run:
        make_dir(new_dir_name)

    for image_file in all_image_files:

        # process image
        raw_image = cv2.imread(image_file)

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=True,
            **image_processing_params
        )

        # create new filename for saving
        image_path = pathlib.Path(image_file)
        filename = image_path.stem
        new_image_filename = os.path.join(
            new_dir_name,
            filename + '.png'
        )

        # save the new image
        if not dry_run:
            cv2.imwrite(new_image_filename, processed_image)

        # show image
        cv2.imshow("proccessed_image", processed_image)
        k = cv2.waitKey(1)
        if k == 27:    # Esc key to stop
            exit()


if __name__ == '__main__':

    image_processing_params = {
        'dims': (128, 128),
        'bbox': [125, 67, 485, 427],
        'thresh': [11, -30],
        'stdiz': False,
        'normlz': False,
        'circle_mask_radius': 165,
    }

    dry_run = False

    # tasks = ['edge_2d', 'edge_3d', 'edge_5d', 'surface_3d']
    tasks = ['alphabet', 'arrows']
    sets = ['train', 'val']

    for task in tasks:
        for set in sets:

            base_dir = os.path.join(
                data_path,
                task,
                set,
            )

            if not dry_run:
                save_json_obj(image_processing_params, os.path.join(base_dir, 'image_processing_params'))

            process_dataset(
                base_dir,
                image_processing_params=image_processing_params,
                dry_run=dry_run
            )
