import os
from itertools import combinations

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from io import BytesIO

def read_tfrecord(example_proto):
    feature_dict = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64)
    }

    features = tf.io.parse_single_example(
        example_proto, features=feature_dict)

    width = tf.cast(features['image/height'], tf.int32)
    height = tf.cast(features['image/width'], tf.int32)
    label = tf.cast(features['image/object/class/label'], tf.int32)
    img_encoded = tf.cast(features['image/encoded'], tf.string)
    
    filename = tf.cast(features['image/filename'], tf.string)
    # triangle_data.append(tf.squeeze(tf.reshape(
    #     features[triangle_stream].values, (1, 13))))

    return width, height, img_encoded, label, filename


def get_image(img, width, height):
    image = tf.image.decode_jpeg(img, channels=3)
    image = tf.image.resize(image, [width, height])
    # image = tf.reshape(image, tf.stack([height, width, 3]))
    # image = tf.reshape(image, [1, height, width, 3])
    # image = tf.image.per_image_standardization(image)
    image = tf.cast(image, dtype='uint8')
    return image


def load_dataset(tf_record_path):
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)
    parsed_dataset = raw_dataset.map(read_tfrecord)
    return parsed_dataset


def prepare_for_training(ds, shuffle_buffer_size=20):
    # ds.cache() # I can remove this to don't use cache or use cocodata.tfcache

    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(
        10).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def filter_func(hands, face, triangle_data, centroids, video, label, video_name, triangle_stream_arr):
    return tf.math.greater(label, 206)


def load_data_tfrecord(tfrecord_path):
    dataset = load_dataset(tfrecord_path)
    # dataset = dataset.filter(filter_func)

    dataset = prepare_for_training(dataset)
    return dataset


tf_record_path = tf.io.gfile.glob(
    '/home/alvaro/Downloads/object_detection_dataset/train_records-002/train_autonomy/*.tfrecord')

def count_data_items(tfrecord):
    count = 0
    for fn in tfrecord:
        for _ in tf.compat.v1.python_io.tf_record_iterator(fn):
            count += 1

    return count

num_imgs = count_data_items(tf_record_path)

dataset = load_data_tfrecord(tf_record_path)

for width, height, img_encoded, label, filename in dataset:
    encoded_img = img_encoded[0].numpy()
    im = Image.open(BytesIO(encoded_img))
    im.save(f'/home/alvaro/Downloads/autonomy_test_img/{filename[0].numpy().decode("utf-8")}')
