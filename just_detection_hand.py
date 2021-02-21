import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import time
import glob 
import math
from random import randint

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = "./pipeline.config"
PATH_TO_CKPT = "./checkpoint"
PATH_TO_LABELS = './label_map.pbtxt'
IMAGE_PATHS = '/home/alvaro/Documentos/projeto libras/frame dataset/validation/'

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


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_video(path):
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Não foi possível carregar o vídeo')
        sys.exit()
    
    return video

video = load_video('/home/alvaro/Área de Trabalho/WLASL/start_kit/raw_videos/0ScBo3iRUhY.mkv')
  
def detect_obj(frame):
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    original_frame = frame.copy()

    bouding_boxes = viz_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

    return frame

def show_frame(frame):
    cv2.imshow('detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_initial_frame(video, frame_number):
    for f in range(frame_number + 1):
        loaded, frame = video.read()
        if f == frame_number:
            return loaded, frame

loaded, frame = get_initial_frame(video, 35)
fps_rate = []
count = 1

while True:
    loaded, frame = video.read()
    if not loaded:
        break

    timer = cv2.getTickCount()

    frame = detect_obj(frame)
    
    fps_rate.append(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    mean = sum(fps_rate) / count
    cv2.putText(frame, 'FPS: ' + str(fps_rate[count-1]), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
    cv2.putText(frame, 'Mean FPS: ' + str(mean), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)

    cv2.imshow('Tracking', frame)
    count += 1
    if cv2.waitKey(1) & 0XFF == 27:
        break

