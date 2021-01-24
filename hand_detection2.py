import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import time
import glob 

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


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def get_save_path(word, image_name, count):
    save_path = '/home/alvaro/Documentos/projeto libras/frame dataset/validation_hand/'
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

        bouding_boxes = result = viz_utils.visualize_boxes_and_labels_on_image_array(
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
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  
        count = 0

        for box in bouding_boxes:
            if count == 2:
                break

            xmin, xmax, ymin, ymax = box['xmin'], box['xmax'], box['ymin'], box['ymax']
            xmin, xmax, ymin, ymax = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

            save_path = get_save_path(word, image_name, count)
            cv2.imwrite(save_path, image_np[ymin:ymax, xmin:xmax, :])
            count +=1

        while count < 2:
            black_img = np.zeros((224,224, 3))
            save_path = get_save_path(word, image_name, count)
            cv2.imwrite(save_path, black_img)
            count +=1 
        
