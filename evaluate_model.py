import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import time
import glob
from itertools import chain
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

base_path = '/home/alvaro/Documentos/projeto libras/frame dataset/validation/'
image_path_list = [glob.glob(base_path+folder+'/*.jpg') for folder in os.listdir(base_path)] 
final_list = list(chain(*image_path_list))

print('Loading model... ', end='')

model = load_model('./model/three_stream_cnn.h5')

def get_label_map():
	train_datagen = ImageDataGenerator()
	generator = train_datagen.flow_from_directory(
		'/home/alvaro/Documentos/projeto libras/frame dataset/validation', batch_size=1)
	y_true_labels = generator.classes
	y_class_name = generator.filenames

	labels = {}

	count = 0
	for class_number in y_true_labels:
		if labels.get(class_number) == None:
			labels[y_class_name[count].split('/')[0]] = class_number

		count += 1

	return labels

def get_image_and_label(image_path, label_map):
    image = np.array(Image.open(image_path))
    label, folder = get_label(image_path, label_map)
    hand_search_path = '/home/alvaro/Documentos/projeto libras/frame dataset/validation_hand/'+folder+'_hand/'
    hands = get_hands_img(hand_search_path, image_path)
    stacked_img = np.stack([tf.image.resize(image, [224,224]), tf.image.resize(
        hands[0], [224, 224]), tf.image.resize(hands[1], [224, 224])])

    return stacked_img, label


def get_label(image_path, label_map):
    splitted_path = image_path.split('/')
    folder = splitted_path[len(splitted_path) - 2]
    label = label_map[folder]
    return label, folder


def get_hands_img(search_path, image_path):
    hands = []    
    image_name = image_path.split('/')[-1].split('.jpg')[0]

    for i in range(2):
        hand_image_name = image_name+'hand_'+str(i)+'.jpg'
        hands.append(np.array(Image.open(search_path+hand_image_name)))

    return hands

def evaluate_model_without_generator():
    final_pred = []
    final_labels = []
    label_map = get_label_map()

    for start in range(0, len(final_list), 128):
        print('Remaining images', len(final_list) - start)
        end = min(start + 128, 3767)
        images_and_labels = np.array([get_image_and_label(image_path, label_map) for image_path in final_list[start:end]])
        read_imgs = images_and_labels[:, 0]
        labels = images_and_labels[:, 1]
        final_image = []
        for image in read_imgs:
            final_image.append(image)

        final_image = np.array(final_image)

        img1 = final_image[:,0,:,:]    
        img2 = final_image[:,1,:,:]
        img3 = final_image[:,2,:,:]
        pred = model.predict([img1, img2, img3])
        final_pred.extend([np.argmax(p) for p in pred]) 
        final_labels.extend(labels)

    acc = accuracy_score(final_labels, final_pred)
    return acc

evaluate_model_without_generator()

