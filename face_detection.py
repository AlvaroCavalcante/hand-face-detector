import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt 

def detect_face(image):
    PATH_TO_CKPT = './model/frozen_inference_graph.pb'
    PATH_TO_LABELS = './proto/label_map.pbtxt'
    NUM_CLASSES = 1

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load Tensorflow model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # Actual detection.
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Start video stream
    # cap = WebcamVideoStream(0).start()
    # fps = FPS().start()

    original_image = image.copy()

    expanded_frame = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num_c) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: expanded_frame})

    bouding_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.40)[0] # zero position

    im_width, im_height = image.shape[1], image.shape[0]
    final_im_width, final_im_height = 224, 224

    xmin, xmax, ymin, ymax = bouding_boxes['xmin'], bouding_boxes['xmax'], bouding_boxes['ymin'], bouding_boxes['ymax']
    xmin, xmax, ymin, ymax = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)

    f_xmin, f_xmax, f_ymin, f_ymax = int(xmin * final_im_width), int(xmax * final_im_width), int(ymin * final_im_height), int(ymax * final_im_height)
    centroid = (int((f_xmin+f_xmax)/2), int((f_ymin+f_ymax)/2))

    return original_image[ymin:ymax, xmin:xmax, :], centroid
    # cv2.imwrite('./faces/face.jpg', original_image[ymin:ymax, xmin:xmax, :])
    # cv2.imwrite('./results/test.jpg', image)

    
if __name__ == '__main__':
    image = cv2.imread('/home/alvaro/Downloads/20 FPS/test/hospital/VID_20201011_1445172020-10-18 10:45:24.919016.jpg')
    detect_face(image)
