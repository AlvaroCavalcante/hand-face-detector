import time

import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

from utils import label_map_util

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


def draw_boxes_on_img(image_np_with_detections, scores, classes, boxes, heigth, width):
    category_index = label_map_util.create_category_index_from_labelmap('./utils/label_map.pbtxt',
                                                                        use_display_name=True)

    hand_counter = 2
    face_counter = 1
    for i in np.where(scores > .5)[0]:
        class_name = category_index[classes[i]].get('name')

        if face_counter == 0 and hand_counter == 0:
            return image_np_with_detections
        elif class_name == 'face' and face_counter == 0:
            continue
        elif class_name == 'hand' and hand_counter == 0:
            continue

        class_name = 'hand_' + \
            str(hand_counter) if class_name == 'hand' else class_name

        xmin, xmax, ymin, ymax = boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2]
        xmin, xmax, ymin, ymax = int(
            xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)

        color = (0, 255, 0) if class_name == 'face' else (255, 0, 0)
        cv2.rectangle(image_np_with_detections,
                      (xmin, ymin), (xmax, ymax), color, 2)

        if class_name == 'face':
            face_counter -= 1
        else:
            hand_counter -= 1

    return image_np_with_detections


def show_bboxes_on_img(detections, output_img):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    heigth, width = output_img.shape[0], output_img.shape[1]

    image_np_with_detections = draw_boxes_on_img(
        output_img,
        detections['detection_scores'],
        detections['detection_classes'],
        detections['detection_boxes'],
        heigth, width)

    return image_np_with_detections


def start_benchmark(model_path, videos_path, width, height, show_img=False):
    print('Loading object detection model...')
    detect_fn = tf.saved_model.load(model_path)
    print('Model loaded')

    pbar = tqdm(total=len(videos_path))
    inference_times = []

    for video in videos_path:
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = frame.copy()

            input_tensor = get_input_tensor(frame, width, height)

            start_inference_time = time.time()
            detections = detect_fn(input_tensor)
            final_inference_time = time.time() - start_inference_time
            inference_times.append(final_inference_time)

            if show_img:
                output_img = show_bboxes_on_img(detections, output_img)
                cv2.imshow('Frame', cv2.cvtColor(
                    output_img, cv2.COLOR_BGR2RGB))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        pbar.update(1)
        cap.release()

    inference_times.pop(0)
    print('Average inference time: ', np.mean(inference_times))
    print('Inference std: ', np.std(inference_times))


def get_input_tensor(frame, width, height):
    frame = cv2.resize(frame, (width, height)).astype('uint8')

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor


if __name__ == '__main__':
    model_path = './utils/models/saved_model_ssd_fpn_320x320_autsl'
    videos_path = tf.io.gfile.glob(
        '/home/alvaro/Documents/AUTSL_VIDEO_DATA/test/test/*.mp4')

    IMG_WIDTH = 320
    IMG_HEIGHT = 320

    start_benchmark(
        model_path, 
        videos_path,
        IMG_WIDTH,
        IMG_HEIGHT,
        show_img=False
    )
