import glob
import functools
import random
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as mp
from matplotlib import pyplot as plt

from object_detection.core import standard_fields as fields
from object_detection.core import preprocessor


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.float32)  # .astype(np.uint8)


def save_image(image, image_name):
    plt.figure()
    plt.imshow(image)
    plt.show()

    mp.savefig(image_name, dpi=300,  bbox_inches='tight')

method = tf.image.ResizeMethod.BILINEAR

per_channel_pad_value = (0, 0, 0)

dataset_path = "/home/alvaro/Downloads/autonomy_train_img/"
images = glob.glob(dataset_path+"*.jpg")
img_index = random.randint(0, len(images))

image2 = Image.open(images[img_index])
image_np = load_image_into_numpy_array(image2)


keep_aspect_fn = functools.partial(
    preprocessor.resize_to_range,
    min_dimension=640,
    max_dimension=640,
    method=method,
    pad_to_max_dimension=True,
    per_channel_pad_value=per_channel_pad_value)

fixed_shape_fn = functools.partial(
    preprocessor.resize_image,
    new_height=640,
    new_width=640,
    method=method)

keep_aspect_image = keep_aspect_fn(image=tf.constant(image_np.astype(np.float32)))
fixed_shape_image = fixed_shape_fn(image=tf.constant(image_np.astype(np.float32)))

save_image(keep_aspect_image[0].numpy().astype(int), 'keep_aspect_ratio_resizer.png')
save_image(fixed_shape_image[0].numpy().astype(int), 'fixed_shape_resizer.png')
