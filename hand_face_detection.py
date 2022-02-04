import tensorflow as tf
import numpy as np
import cv2
import time
from utils import label_map_util
import argparse
from itertools import combinations
import math

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('GPU not found')

def get_centroids(bouding_boxes):
    centroids = []
    for bbox in bouding_boxes:
        xmin, xmax, ymin, ymax = bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax']
        centroid = (int((xmin+xmax)/2), int((ymin+ymax)/2))
        centroids.append(centroid)
    
    return centroids

def get_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    cos_value = ((adjacent_1**2 + adjacent_2**2) - opposite**2) / (2*(adjacent_1*adjacent_2))
    rad = math.acos(cos_value)

    degrees = rad * 180 / math.pi 

    return degrees

def compute_triangle_features(euclidian_distances):
    d1, d2, d3 = euclidian_distances[0], euclidian_distances[1], euclidian_distances[2]
    triangle_features = {}

    triangle_features['perimeter'] = d1 + d2 + d3
    triangle_features['semi_perimeter'] = triangle_features['perimeter'] / 2
    triangle_features['area'] = math.sqrt(   # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
        (triangle_features['semi_perimeter'] * (triangle_features['semi_perimeter'] - d1) * (
        triangle_features['semi_perimeter'] - d2) * (triangle_features['semi_perimeter'] - d3)))

    triangle_features['ang_inter_a'] = get_angle(d3, d1, d2) 
    triangle_features['ang_inter_b'] = get_angle(d1, d2, d3)
    triangle_features['ang_inter_c'] = 180.0 - (triangle_features['ang_inter_a'] + triangle_features['ang_inter_b'])

    # teorema dos Ângulos externos https://pt.wikipedia.org/wiki/Teorema_dos_%C3%A2ngulos_externos
    triangle_features['ang_ext_a'] = triangle_features['ang_inter_b'] + triangle_features['ang_inter_c']
    triangle_features['ang_ext_b'] = triangle_features['ang_inter_a'] + triangle_features['ang_inter_c']
    triangle_features['ang_ext_c'] = triangle_features['ang_inter_b'] + triangle_features['ang_inter_a']

    return triangle_features

def compute_features_and_draw_lines(bouding_boxes, img):
    centroids = get_centroids(bouding_boxes)
    euclidian_distances = []

    for cc in combinations(centroids, 2):
        centroid_1 = cc[0]
        centroid_2 = cc[1]
        cv2.line(img, (centroid_1[0], centroid_1[1]), (centroid_2[0], centroid_2[1]), (0, 255, 0), thickness=5)
        distance = math.sqrt((centroid_1[0]-centroid_2[0])**2+(centroid_1[1]-centroid_2[1])**2)
        euclidian_distances.append(distance)

    triangle_features = {}
    if len(centroids) == 3:
        triangle_features = compute_triangle_features(euclidian_distances)

    return img, euclidian_distances, triangle_features

def draw_boxes_on_img(image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width, single_person):
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path,
                                                            use_display_name=True)

    output_bboxes = []
    hand_counter = 2
    face_counter = 1
    for i in np.where(scores > .4)[0]:
        class_name = category_index[classes[i]].get('name')

        if single_person:
            if face_counter == 0 and hand_counter == 0:
                return image_np_with_detections, output_bboxes      
            if class_name == 'face' and face_counter == 0:
                continue
            elif class_name == 'hand' and hand_counter == 0:
                continue

        xmin, xmax, ymin, ymax = boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2]
        xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * heigth), int(ymax * heigth)
        output_bboxes.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax})
        
        color = (0,255,0) if class_name == 'face' else (255,0,0)
        cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), color, 2)

        if class_name == 'face':
            face_counter -= 1
        else:
            hand_counter -= 1

    return image_np_with_detections, output_bboxes

def infer_images(image, output_img, label_map_path, detect_fn, heigth, width, single_person):
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
        heigth, width, single_person)

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
        output_img, bouding_boxes = infer_images(frame, output_img, args.label_map_path, detect_fn, heigth, width, args.single_person)

        if args.compute_features:
            output_img, distances, triangle_features = compute_features_and_draw_lines(bouding_boxes, output_img)

        count += 1
        if (time.time() - start_time) > 1:
            fps = int(count / (time.time() - start_time))
            print('detection FPS: {}'.format(str(fps)))
            count = 0
            start_time = time.time()
            fps_hist.append(fps)

        if not args.use_docker:
            cv2.imshow('Frame', cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    print(fps_hist)
    print('Mean FPS: ', str(sum(fps_hist) / len(fps_hist)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_path', type=str, default='./utils/models/ssd_v2/saved_model')
    parser.add_argument('--source_path', type=str)
    parser.add_argument('--label_map_path', type=str, default='./utils/label_map.pbtxt')
    parser.add_argument('--compute_features', type=bool, default=True)
    parser.add_argument('--use_docker', type=bool, default=False)
    parser.add_argument('--single_person', type=bool, default=True)
    args = parser.parse_args()

    main(args)