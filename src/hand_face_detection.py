import argparse
import math
import time

import tensorflow as tf
import numpy as np
import cv2

from utils import label_map_util


def set_device(device: str):
    print("Setting device...")

    if device == "cpu":
        print("Running model on CPU")
        tf.config.set_visible_devices([], "GPU")
        visible_devices = tf.config.get_visible_devices()
        for devices in visible_devices:
            assert devices.device_type != "GPU"
    else:
        print("Running model on GPU")
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            print("GPU not found")


def get_centroids(bouding_boxes):
    centroids = {}
    for class_name in bouding_boxes:
        bbox = bouding_boxes.get(class_name)
        if not bbox:
            continue
        xmin, xmax, ymin, ymax = bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]

        centroid = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        centroids[class_name] = centroid

    return centroids


def get_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    cos_value = ((adjacent_1**2 + adjacent_2**2) - opposite**2) / (
        2 * (adjacent_1 * adjacent_2)
    )
    rad = math.acos(cos_value)

    degrees = rad * 180 / math.pi

    return degrees


def compute_centroids_distances(centroids, img):
    try:
        d1 = math.sqrt(
            (centroids["hand_1"][0] - centroids["face"][0]) ** 2
            + (centroids["hand_1"][1] - centroids["face"][1]) ** 2
        )

        cv2.line(
            img,
            (centroids["hand_1"][0], centroids["hand_1"][1]),
            (centroids["face"][0], centroids["face"][1]),
            (0, 255, 0),
            thickness=5,
        )

        d2 = math.sqrt(
            (centroids["hand_2"][0] - centroids["face"][0]) ** 2
            + (centroids["hand_2"][1] - centroids["face"][1]) ** 2
        )

        cv2.line(
            img,
            (centroids["hand_2"][0], centroids["hand_2"][1]),
            (centroids["face"][0], centroids["face"][1]),
            (0, 255, 0),
            thickness=5,
        )

        d3 = math.sqrt(
            (centroids["hand_1"][0] - centroids["hand_2"][0]) ** 2
            + (centroids["hand_1"][1] - centroids["hand_2"][1]) ** 2
        )

        cv2.line(
            img,
            (centroids["hand_1"][0], centroids["hand_1"][1]),
            (centroids["hand_2"][0], centroids["hand_2"][1]),
            (0, 255, 0),
            thickness=5,
        )

        return d1, d2, d3
    except Exception as e:
        print("Error to calculate centroids distances")
        print(e)


def get_normalized_angle(opposite, adjacent_1, adjacent_2):
    # lei dos cossenos: https://pt.khanacademy.org/math/trigonometry/trig-with-general-triangles/law-of-cosines/v/law-of-cosines-missing-angle
    try:
        cos_value = ((adjacent_1**2 + adjacent_2**2) - opposite**2) / (
            2 * (adjacent_1 * adjacent_2) + 1e-10
        )
        rad = math.acos(cos_value)

        degrees = rad / math.pi  # rad * 180 to remove normalization [0 - 1]

        return degrees
    except Exception as e:
        print("Error to calculate normalized angle")
        print(e)


def draw_lines_and_text(img, d1, d2, d3, centroids):
    pos = (centroids["hand_1"][0] + centroids["face"][0]) // 2, (
        centroids["hand_1"][1] + centroids["face"][1]
    ) // 2

    cv2.putText(
        img,
        str(round(d1, 3)),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color=(216, 139, 37),
        thickness=2,
    )

    pos = (centroids["hand_2"][0] + centroids["face"][0]) // 2, (
        centroids["hand_2"][1] + centroids["face"][1]
    ) // 2
    cv2.putText(
        img,
        str(round(d2, 3)),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color=(216, 139, 37),
        thickness=2,
    )

    pos = (centroids["hand_2"][0] + centroids["hand_1"][0]) // 2, (
        centroids["hand_2"][1] + centroids["hand_1"][1]
    ) // 2
    cv2.putText(
        img,
        str(round(d3, 3)),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color=(216, 139, 37),
        thickness=2,
    )


def compute_triangle_features(centroids, img):
    triangle_features = {}
    try:
        d1, d2, d3 = compute_centroids_distances(centroids, img)
        perimeter = d1 + d2 + d3
        triangle_features["perimeter"] = perimeter
        triangle_features["semi_perimeter"] = triangle_features["perimeter"] / 2

        triangle_features[
            "area"
        ] = math.sqrt(  # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
            (
                triangle_features["semi_perimeter"]
                * (triangle_features["semi_perimeter"] - d1)
                * (triangle_features["semi_perimeter"] - d2)
                * (triangle_features["semi_perimeter"] - d3)
            )
        )

        # avoid 0 division
        triangle_features["height"] = 2 * triangle_features["area"] / (d3 + 1e-10)

        d1, d2, d3 = d1 / perimeter, d2 / perimeter, d3 / perimeter

        draw_lines_and_text(img, d1, d2, d3, centroids)

        triangle_features.update({"distance_1": d1, "distance_2": d2, "distance_3": d3})

        triangle_features[
            "norm_area"
        ] = math.sqrt(  # Fórmula de Heron https://www.todamateria.com.br/area-do-triangulo/
            (0.5 * (0.5 - d1) * (0.5 - d2) * (0.5 - d3))
        )

        triangle_features["norm_height"] = (
            2 * triangle_features["norm_area"] / (d3 + 1e-10)
        )

        pos = (30, 30)
        cv2.putText(
            img,
            "A:" + str(round(triangle_features["norm_area"], 5)),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(0, 0, 255),
            thickness=5,
        )

        pos = (30, 80)
        cv2.putText(
            img,
            "H:" + str(round(triangle_features["norm_height"], 5)),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(0, 0, 255),
            thickness=5,
        )

        triangle_features["ang_inter_a"] = get_normalized_angle(d3, d1, d2)
        triangle_features["ang_inter_b"] = get_normalized_angle(d1, d2, d3)
        triangle_features["ang_inter_c"] = 1 - (
            triangle_features["ang_inter_a"] + triangle_features["ang_inter_b"]
        )

        pos = (centroids["face"][0] + 20, centroids["face"][1] + 20)
        cv2.putText(
            img,
            str(round(triangle_features["ang_inter_a"], 2)),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(0, 0, 255),
            thickness=2,
        )

        pos = (centroids["hand_2"][0] + 20, centroids["hand_2"][1] + 20)
        cv2.putText(
            img,
            str(round(triangle_features["ang_inter_b"], 2)),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(0, 0, 255),
            thickness=2,
        )

        pos = (centroids["hand_1"][0] + 20, centroids["hand_1"][1] + 20)
        cv2.putText(
            img,
            str(round(triangle_features["ang_inter_c"], 2)),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(0, 0, 255),
            thickness=2,
        )

        # teorema dos Ângulos externos https://pt.wikipedia.org/wiki/Teorema_dos_%C3%A2ngulos_externos
        triangle_features["ang_ext_a"] = (
            triangle_features["ang_inter_b"] + triangle_features["ang_inter_c"]
        )
        triangle_features["ang_ext_b"] = (
            triangle_features["ang_inter_a"] + triangle_features["ang_inter_c"]
        )
        triangle_features["ang_ext_c"] = (
            triangle_features["ang_inter_b"] + triangle_features["ang_inter_a"]
        )
    except Exception as e:
        print("Error to calculate triangle features")
        print(e)

    return triangle_features


def compute_features_and_draw_lines(bouding_boxes, img):
    triangle_features = {}
    centroids = get_centroids(bouding_boxes)
    if len(centroids) == 3:
        triangle_features = compute_triangle_features(centroids, img)

    return img, triangle_features


def draw_boxes_on_img(
    image_np_with_detections, label_map_path, scores, classes, boxes, heigth, width
):
    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path, use_display_name=True
    )

    output_bboxes = {"face": None, "hand_1": None, "hand_2": None}
    hand_counter = 2
    face_counter = 1
    for i in np.where(scores > 0.5)[0]:
        class_name = category_index[classes[i]].get("name")

        if face_counter == 0 and hand_counter == 0:
            return image_np_with_detections, output_bboxes
        elif class_name == "face" and face_counter == 0:
            continue
        elif class_name == "hand" and hand_counter == 0:
            continue

        class_name = "hand_" + str(hand_counter) if class_name == "hand" else class_name

        xmin, xmax, ymin, ymax = boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2]
        xmin, xmax, ymin, ymax = (
            int(xmin * width),
            int(xmax * width),
            int(ymin * heigth),
            int(ymax * heigth),
        )

        output_bboxes[class_name] = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

        color = (0, 255, 0) if class_name == "face" else (255, 0, 0)
        cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), color, 2)

        if class_name == "face":
            face_counter -= 1
        else:
            hand_counter -= 1

    return image_np_with_detections, output_bboxes


def infer_images(image, output_img, label_map_path, detect_fn, heigth, width):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    start = time.time()
    detections = detect_fn(input_tensor)
    inference_time = time.time() - start

    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    image_np_with_detections, bouding_boxes = draw_boxes_on_img(
        output_img,
        label_map_path,
        detections["detection_scores"],
        detections["detection_classes"],
        detections["detection_boxes"],
        heigth,
        width,
    )

    return image_np_with_detections, bouding_boxes, inference_time


def main(args):
    set_device(args.device)
    print("Loading object detection model...")
    detect_fn = tf.saved_model.load(args.saved_model_path)
    print("Model loaded")

    source = args.source_path if args.source_path else 0
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    result = None
    is_first_detection = True

    inference_speed = []
    start_time = time.time()

    while True:
        got_frame, frame = cap.read()

        if not got_frame:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_img = frame.copy()
        heigth, width = output_img.shape[0], output_img.shape[1]

        if not result:
            result = cv2.VideoWriter(
                "output.avi", fourcc, 30.0, (int(width), int(heigth))
            )

        frame = cv2.resize(frame, (int(args.img_res), int(args.img_res))).astype(
            "uint8"
        )

        output_img, _, inference_time = infer_images(
            frame, output_img, args.label_map_path, detect_fn, heigth, width
        )

        inference_speed.append(inference_time)
        if is_first_detection:
            inference_speed.pop(0)
            is_first_detection = False

        # if args.compute_features:
        #     output_img, triangle_features = compute_features_and_draw_lines(
        #         bouding_boxes, output_img)

        if (time.time() - start_time) > 1:
            mean_inf_speed_ms = 1e3 * np.mean(inference_speed)
            print("inference ms: {}".format(round(mean_inf_speed_ms, 2)))
            print("inference FPS: {}".format(round(1000 / mean_inf_speed_ms, 2)))
            start_time = time.time()

        if args.show_results:
            color_frame = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            cv2.putText(
                color_frame,
                f"FPS: {str(round(1000 / mean_inf_speed_ms, 2))}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color=(216, 139, 37),
                thickness=2,
            )
            cv2.imshow("Frame", color_frame)
            result.write(color_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    result.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument(
        "--label_map_path", type=str, default="src/utils/label_map.pbtxt"
    )
    # parser.add_argument('--compute_features', type=bool, default=True)
    parser.add_argument("--show_results", type=bool, default=True)
    parser.add_argument("--img_res", type=str, default="512")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
