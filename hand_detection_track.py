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

video = load_video('/home/alvaro/Documentos/projeto libras/excluded_videos/banco.mp4')
  
def detect_obj(frame, video):
    print('detecting object from detector')
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

    im_width, im_height = frame.shape[1], frame.shape[0]

    for box in bouding_boxes:     
        xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
        xmin, xmax, ymin, ymax = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
        return frame, original_frame, (xmin, ymin, xmax, ymax)
    
    loaded, frame = video.read()

    if loaded:
        return detect_obj(frame, video)
    else:
        print('End of the video')
        sys.exit()


loaded, frame = video.read()
tracker = cv2.TrackerCSRT_create()
colors = (randint(0,255), randint(0,255), randint(0,255))

frame, original_frame, bbox = detect_obj(frame, video)
ok = tracker.init(original_frame, bbox)

while True:
    loaded, frame = video.read()
    if not loaded:
        break

    timer = cv2.getTickCount()

    loaded, bbox = tracker.update(frame)

    if loaded:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), colors, 2)
    else:
        frame, original_frame, bbox = detect_obj(frame, video)
        tracker.init(original_frame, bbox)
    
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, 'FPS: ' + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)

    cv2.imshow('Tracking', frame)
    k = cv2.waitKey(1) & 0XFF
    if k == 27:
        break

