# Code got from: https://github.com/majrie/visualize_augmentation
# All augmentations options: https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto

from object_detection.core import preprocessor
import functools
import os
from object_detection import inputs
from object_detection.core import standard_fields as fields
from matplotlib import pyplot as mp
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.float32)  # .astype(np.uint8)


# lot of augmentations have probabilities < 1 will not happen if repeated only once.
number_of_repeats = 20

image2 = Image.open(
    "/home/alvaro/Downloads/test-001/test/signer0_sample6_color_30.jpg")
image_np = load_image_into_numpy_array(image2)

save_to_disk = True
directory = 'visualize_augmentation'
preprocessing_list = [
    (preprocessor.random_horizontal_flip, {}),
    (preprocessor.random_image_scale,
     {'min_scale_ratio': 0.8}),
    (preprocessor.random_rgb_to_gray, {}),
    (preprocessor.random_adjust_brightness,
     {'max_delta': 0.25}),
    (preprocessor.random_adjust_contrast, {'max_delta': 1.3}),
    (preprocessor.random_adjust_hue, {'max_delta': 0.04}),
    (preprocessor.random_adjust_saturation, {}),
    (preprocessor.random_distort_color, {}),
    (preprocessor.random_jitter_boxes, {}),
    (preprocessor.random_crop_image, {}),
    (preprocessor.random_absolute_pad_image, {}),
    (preprocessor.random_black_patches, {'probability': .35, 'size_to_image_ratio': 0.05}),

    #   (preprocessor.random_vertical_flip, {'probability': 1}),
    #   (preprocessor.random_rotation90, {'probability': 1}),
    #   (preprocessor.random_pixel_value_scale, {}),

    # (preprocessor.random_pad_image, {}),
    # (preprocessor.random_crop_pad_image, {}),
    # (preprocessor.random_crop_to_aspect_ratio, {}),
    # (preprocessor.random_pad_to_aspect_ratio, {}),
    # (preprocessor.random_resize_method, {}),
    # (preprocessor.random_square_crop_by_scale, {}),
    # (preprocessor.random_patch_gaussian, {}),
    # (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    # (preprocessor.subtract_channel_mean, {'probability': 1}),
    # (preprocessor.random_self_concat_image, {}),
    # (preprocessor.ssd_random_crop, {}),
    # (preprocessor.random_jpeg_quality, {}),
    # (preprocessor.ssd_random_crop_pad, {'probability': 1}),
    # (preprocessor.ssd_random_crop_fixed_aspect_ratio, {'probability': 1}),
    # (preprocessor.ssd_random_crop_pad_fixed_aspect_ratio, {'probability': 1}),
    # (preprocessor.convert_class_logits_to_softmax, {'probability': 1}),
]

for preprocessing_technique in preprocessing_list:
    for i in range(number_of_repeats):

        tf.compat.v1.reset_default_graph()
        if preprocessing_technique is not None:
            print(str(preprocessing_technique[0].__name__))
        else:
            print('Image without augmentation: ')
        if preprocessing_technique is not None:
            data_augmentation_options = [preprocessing_technique]
        else:
            data_augmentation_options = []
        data_augmentation_fn = functools.partial(
            inputs.augment_input_data,
            data_augmentation_options=data_augmentation_options)

        tensor_dict = {
            fields.InputDataFields.image:
            tf.constant(image_np.astype(np.float32)),
            fields.InputDataFields.groundtruth_boxes:
            tf.constant(
                np.array([[0.572265625, 0.4267578125, 0.1015625, 0.11132812]], np.float32)),
            fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1.0], np.float32))
        }

        augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)

        plt.figure()
        plt.imshow(
            augmented_tensor_dict[fields.InputDataFields.image].numpy().astype(int))
        plt.show()

        if save_to_disk:
            if not os.path.exists(directory):
                os.makedirs(directory)
            if preprocessing_technique is not None:
                mp.savefig(directory + '/augmentation_'+str(
                    preprocessing_technique[0].__name__)+'_'+str(i)+'.png', dpi=300,  bbox_inches='tight')
            else:
                mp.savefig(directory + '/no_augmentation.png',
                           dpi=300,  bbox_inches='tight')
        plt.close('all')
