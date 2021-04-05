import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import time
import glob 
import math

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTHONPATH"] = '/home/alvaro/Área de Trabalho/models/research'

PATH_TO_CFG = "./pipeline.config"
PATH_TO_CKPT = "./checkpoint"
PATH_TO_LABELS = './label_map.pbtxt'
IMAGE_PATHS = '/home/alvaro/Downloads/20 FPS/test/'

print('Loading model... ', end='')

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-5')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def get_angle(opposite, adjacent_1, adjacent_2):
    cos_value = ((adjacent_1**2 + adjacent_2**2) - opposite**2) / (2*(adjacent_1*adjacent_2))
    rad = math.acos(cos_value)

    degrees = rad * 180 / math.pi 

    return degrees

def get_save_path(word, image_name, count):
    save_path = './hand_images/'
    if word+'_hand' not in os.listdir(save_path):
        os.mkdir(save_path+word+'_hand')

    save_path += word+'_hand/'+image_name.split('.jpg')[0]+'hand_'+str(count) +'.jpg'
    return save_path

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

for word in os.listdir(IMAGE_PATHS):
    for image_name in os.listdir(IMAGE_PATHS+word):

        image_path = IMAGE_PATHS+word+'/'+image_name
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

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

        im_width, im_height = image_np.shape[1], image_np.shape[0]
        final_im_width, final_im_height = image_np.shape[1], image_np.shape[0] # 224, 224

        from face_detection import detect_face
        face, centroid_face = detect_face(image_np)

        centroid_detection = cv2.circle(image_np_with_detections, centroid_face, radius=5, color=(0, 0, 255), thickness=5)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  
        count = 0
        centroids = []

        for box in bouding_boxes:
            if count == 2:
                break

            xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']

            f_xmin, f_xmax, f_ymin, f_ymax = int(xmin * final_im_width), int(xmax * final_im_width), int(ymin * final_im_height), int(ymax * final_im_height)
            xmin, xmax, ymin, ymax = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

            centroids.append((int((f_xmin+f_xmax)/2), int((f_ymin+f_ymax)/2)))
            centroid_detection = cv2.circle(centroid_detection, centroids[count], radius=5, color=(0, 0, 255), thickness=5)

            save_path = get_save_path(word, image_name, count)
            cv2.imwrite(save_path, image_np[ymin:ymax, xmin:xmax, :])
            count +=1

        while count < 2:
            black_img = np.zeros((224,224, 3))
            save_path = get_save_path(word, image_name, count)
            cv2.imwrite(save_path, black_img)
            count +=1 

        cv2.line(centroid_detection, (centroids[1][0], centroids[1][1]), (centroids[0][0], centroids[0][1]), (0, 255, 0), thickness=5)
        cv2.line(centroid_detection, (centroids[1][0], centroids[1][1]), (centroid_face[0], centroid_face[1]), (0, 255, 0), thickness=5)
        cv2.line(centroid_detection, (centroids[0][0], centroids[0][1]), (centroid_face[0], centroid_face[1]), (0, 255, 0), thickness=5)

        cv2.imwrite('centroids.jpg', centroid_detection)

        if len(centroids) == 2:
            distance_1 = math.sqrt((centroids[1][0]-centroids[0][0])**2+(centroids[1][1]-centroids[0][1])**2)
            distance_2 = math.sqrt((centroid_face[0]-centroids[0][0])**2+(centroid_face[1]-centroids[0][1])**2)
            distance_3 = math.sqrt((centroids[1][0]-centroid_face[0])**2+(centroids[1][1]-centroid_face[1])**2)

            perimeter = distance_1 + distance_2 + distance_3
            semi_perimeter = perimeter / 2

            area = math.sqrt((
                semi_perimeter * (semi_perimeter - distance_1) * (semi_perimeter - distance_2) * (semi_perimeter - distance_3)))

            ang_inter_a = get_angle(distance_3, distance_1, distance_2) 
            ang_inter_b = get_angle(distance_1, distance_2, distance_3)
            ang_inter_c = 180.0 - (ang_inter_a + ang_inter_b)