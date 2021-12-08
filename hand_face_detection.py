from os import name
import tensorflow as tf
import numpy as np
import cv2
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import argparse

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')

def infer_images(image, label_map_path, detect_fn):
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


def main(args):
    detect_fn = tf.saved_model.load(args.saved_model_path)

    source = args.source_path if args.source_path else 0
    cap = cv2.VideoCapture(source)

    count = 0
    fps = str(0)
    start_time = time.time()
    fps_hist = []

    while True:
        got_frame, frame = cap.read()

        if not got_frame:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512)).astype('uint8')
        _, _, _, img_np = infer_images(frame, args.label_map_path, detect_fn)
        frame = np.divide(frame, 255)

        count += 1
        if (time.time() - start_time) > 1:
            fps = int(count / (time.time() - start_time))
            print('detection FPS: {}'.format(str(fps)))
            count = 0
            start_time = time.time()
            fps_hist.append(fps)

        cv2.imshow('Frame', cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print(fps_hist)
    print('Mean FPS: ', str(sum(fps_hist) / len(fps_hist)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', type=str)
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--label_map_path', type=str, default='./utils/label_map.pbtxt')
    args = parser.parse_args()

    main(args)