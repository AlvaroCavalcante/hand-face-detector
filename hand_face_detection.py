import tensorflow as tf
import numpy as np
import cv2
import time
from utils import label_map_util
import argparse
from itertools import combinations

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')

def draw_lines_and_centroids(bouding_boxes, img):
    centroids = []
    for bbox in bouding_boxes:
        xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
        centroid = (int((xmin+xmax)/2), int((ymin+ymax)/2))
        centroids.append(centroid)

    for cc in combinations(centroids, 2):
        centroid_1 = cc[0]
        centroid_2 = cc[1]
        cv2.line(img, (centroid_1[0], centroid_1[1]), (centroid_2[0], centroid_2[1]), (0, 255, 0), thickness=5)

    return img

def draw_boxes_on_img(image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                            use_display_name=True)

    output_bboxes = []
    for i in range(len(scores)):
        class_name = category_index[classes[i]].get('name')
        if scores[i] <= 0.4:
            continue

        xmin, xmax, ymin, ymax = boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2]
        xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)
        output_bboxes.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})
        
        color = (0,255,0) if class_name == 'face' else (255,0,0)
        cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), color, 2)

    return image_np_with_detections, output_bboxes

def infer_images(image, output_img, label_map_path, detect_fn, heigth, width):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections, bouding_boxes = draw_boxes_on_img(
        output_img, 
        label_map_path, 
        detections['detection_scores'], 
        detections['detection_classes'], 
        detections['detection_boxes'],
        heigth, width)

    return image_np_with_detections, bouding_boxes

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
        output_img = frame.copy()
        heigth, width = output_img.shape[0], output_img.shape[1]

        frame = cv2.resize(frame, (320, 320)).astype('uint8')
        output_img, bouding_boxes = infer_images(frame, output_img, args.label_map_path, detect_fn, heigth, width)

        if args.compute_features:
            output_img = draw_lines_and_centroids(bouding_boxes, output_img)

        count += 1
        if (time.time() - start_time) > 1:
            fps = int(count / (time.time() - start_time))
            print('detection FPS: {}'.format(str(fps)))
            count = 0
            start_time = time.time()
            fps_hist.append(fps)

        cv2.imshow('Frame', cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print(fps_hist)
    print('Mean FPS: ', str(sum(fps_hist) / len(fps_hist)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', type=str, default='/home/alvaro/Downloads/ssd_v2/saved_model')
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--label_map_path', type=str, default='./utils/label_map.pbtxt')
    parser.add_argument('--compute_features', type=str, default=True)
    args = parser.parse_args()

    main(args)