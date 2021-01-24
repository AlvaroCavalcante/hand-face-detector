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

PATH_TO_CFG = "./pipeline.config"
PATH_TO_CKPT = "./checkpoint"
PATH_TO_LABELS = './label_map.pbtxt'
IMAGE_PATHS = '/home/alvaro/Documentos/projeto libras/frame dataset/20 FPS hand/test/'

save_hand_image = False

base_path = '/home/alvaro/Documentos/projeto libras/frame dataset/validation/'
image_path_list = [glob.glob(base_path+folder+'/*.jpg') for folder in os.listdir(base_path)] 
final_list = list(chain(*image_path_list))

print('Loading model... ', end='')

model = load_model('./model/three_stream_cnn.h5')

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-5')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

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


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def get_save_path(word, image_name, count):
    save_path = '/home/alvaro/Documentos/projeto libras/frame dataset/20 FPS hand/test_hand/'
    if word+'_hand' not in os.listdir(save_path):
        os.mkdir(save_path+word+'_hand')

    save_path += word+'_hand/' + \
        image_name.split('.jpg')[0]+'hand_'+str(count) + '.jpg'
    return save_path


def detect_hand(image):
    image = cv2.resize(image, dsize=(224,224)) #image.reshape((224,224,3))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image.copy()

    bouding_boxes = viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    im_width, im_height = image.shape[1], image.shape[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    count = 0
    hands = []

    for box in bouding_boxes:
        if count == 2:
            break

        xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
        xmin, xmax, ymin, ymax = int(
            xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

        hands.append(image[ymin:ymax, xmin:xmax, :])
        count += 1

    while count < 2:
        black_img = np.zeros((224, 224, 3))
        hands.append(black_img)

        count += 1

    stacked_img = np.stack([image, tf.image.resize(
        hands[0], [224, 224]), tf.image.resize(hands[1], [224, 224])])
    return stacked_img

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))

def custom_generator():
    datagen = ImageDataGenerator()
    aug_iter = datagen.flow_from_directory('/home/alvaro/Documentos/projeto libras/frame dataset/validation',
                                       target_size=[224, 224], batch_size=5, class_mode='sparse')

    while True:
        output = aug_iter.next()
        final_image = []
        for image in output[0]:
            final_image.append(detect_hand(image))

        final_image = np.array(final_image)

        img1 = final_image[:,0,:,:]    
        img2 = final_image[:,1,:,:]
        img3 = final_image[:,2,:,:]
        pred = model.predict([img1, img2, img3])
        # yield np.array([img1, img2, img3])# , tf.cast(output[1], tf.int32)
        yield img1, img2, img3

def get_image_and_label(image_path, label_map):
    image = np.array(Image.open(image_path))
    label = get_label(image_path, label_map)
    return image, label

def get_label(image_path, label_map):
    splitted_path = image_path.split('/')
    folder = splitted_path[len(splitted_path) - 2]
    label = label_map[folder]
    return label


class CustomGenerator(keras.utils.Sequence):
    def __init__(self, img_path_list, batch_size):
        self.image_path_list = img_path_list 
        self.batch_size = batch_size
        self.label_map = get_label_map()

    def __len__(self):
        return int(3767/ self.batch_size)

    def __getitem__(self, idx):
        for start in range(0, len(self.image_path_list), self.batch_size):
            end = min(start + self.batch_size, 3767)
            images_and_labels = np.array([get_image_and_label(image_path, self.label_map) for image_path in self.image_path_list[start:end]])
            final_image = []
            read_imgs = images_and_labels[:, 0]
            labels = images_and_labels[:, 1]
            for image in read_imgs:
                final_image.append(detect_hand(image))

            final_image = np.array(final_image).astype('float32')

            img1 = final_image[:,0,:,:]    
            img2 = final_image[:,1,:,:]
            img3 = final_image[:,2,:,:]

            return [img1, img2, img3], labels

def evaluate_model_without_generator():
    final_pred = []
    final_labels = []
    label_map = get_label_map()

    for start in range(0, len(final_list), 32):
        print('Remaining images', len(final_list) - start)
        end = min(start + 32, 3767)
        images_and_labels = np.array([get_image_and_label(image_path, label_map) for image_path in final_list[start:end]])
        read_imgs = images_and_labels[:, 0]
        labels = images_and_labels[:, 1]
        final_image = []
        for image in read_imgs:
            final_image.append(detect_hand(image))

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

custom = CustomGenerator(final_list, 32)
test_acc = model.evaluate(custom)

