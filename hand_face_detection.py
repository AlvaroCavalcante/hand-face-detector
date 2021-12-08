import tensorflow as tf
import numpy as np
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# tf.config.optimizer.set_jit(True)

def infer_images(image, label_map_path, draw_image=False):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                            use_display_name=True)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    image_np_with_detections = image_np.copy()

    if draw_image:
      viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    return detections['detection_boxes'], detections['detection_classes'], detections['detection_scores'], image_np_with_detections

detect_fn = tf.saved_model.load('/home/alvaro/Downloads/exported_model_2/saved_model/')

cap = cv2.VideoCapture(0)

count = 0
fps = str(0)
start_time = time.time()

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # output = frame.copy()
    frame = cv2.resize(frame, (512, 512)).astype('uint8')
    _, _, _, img_np = infer_images(frame, '/home/alvaro/Documentos/body-detection/utils/label_map.pbtxt', True)
    frame = np.divide(frame, 255)

    count += 1
    if (time.time() - start_time) > 1:
        fps = int(count / (time.time() - start_time))
        print("detection FPS: {}".format(str(fps)))
        count = 0
        start_time = time.time()

    cv2.imshow("Frame", cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break