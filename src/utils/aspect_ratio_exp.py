"""
This script is used for generating the best aspect ratios for the anchor boxes of object detectors.
"""

import sys
import os
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.cluster import KMeans

# Code got from here: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/generate_ssd_anchor_box_aspect_ratios_using_k_means_clustering.ipynb

def xml_to_boxes(path, rescale_width=None, rescale_height=None):
    """Extracts bounding-box widths and heights from ground-truth dataset.

    Args:
    path : Path to .xml annotation files for your dataset.
    rescale_width : Scaling factor to rescale width of bounding box.
    rescale_height : Scaling factor to rescale height of bounding box.

    Returns:
    bboxes : A numpy array with pairs of box dimensions as [width, height].
    """

    xml_list = []
    filenames = os.listdir(os.path.join(path))
    filenames = [os.path.join(path, f)
                 for f in filenames if (f.endswith('.xml'))]
    for xml_file in filenames:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            bbox_width = int(bndbox.find('xmax').text) - \
                int(bndbox.find('xmin').text)
            bbox_height = int(bndbox.find('ymax').text) - \
                int(bndbox.find('ymin').text)
            if rescale_width and rescale_height:
                size = root.find('size')
                bbox_width = bbox_width * \
                    (rescale_width / int(size.find('width').text))
                bbox_height = bbox_height * \
                    (rescale_height / int(size.find('height').text))
            xml_list.append([bbox_width, bbox_height])
    bboxes = np.array(xml_list)
    return bboxes


def average_iou(bboxes, anchors):
    """Calculates the Intersection over Union (IoU) between bounding boxes and
    anchors.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    anchors : Array of aspect ratios [n, 2] format.

    Returns:
    avg_iou_perc : A Float value, average of IOU scores from each aspect ratio
    """
    intersection_width = np.minimum(anchors[:, [0]], bboxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], bboxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(bboxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    avg_iou_perc = np.mean(
        np.max(intersection_area / union_area, axis=1)) * 100

    return avg_iou_perc


def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    """Calculate the centroid of bounding boxes clusters using Kmeans algorithm.

    Args:
    bboxes : Array of bounding boxes in [width, height] format.
    kmeans_max_iter : Maximum number of iterations to find centroids.
    num_aspect_ratios : Number of centroids to optimize kmeans.

    Returns:
    aspect_ratios : Centroids of cluster (optmised for dataset).
    avg_iou_prec : Average score of bboxes intersecting with new aspect ratios.
    """

    assert len(bboxes), "You must provide bounding boxes"

    normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))

    # Using kmeans to find centroids of the width/height clusters
    kmeans = KMeans(
        init='random', n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
    kmeans.fit(X=normalized_bboxes)
    ar = kmeans.cluster_centers_

    assert len(
        ar), "Unable to find k-means centroid, try increasing kmeans_max_iter."

    avg_iou_perc = average_iou(normalized_bboxes, ar)

    if not np.isfinite(avg_iou_perc):
        sys.exit("Failed to get aspect ratios due to numerical errors in k-means")

    aspect_ratios = [w/h for w, h in ar]

    return aspect_ratios, avg_iou_perc


num_aspect_ratios = 2 # can be [2,3,4,5,6]

# Tune the iterations based on the size and distribution of your dataset
# You can check avg_iou_prec every 100 iterations to see how centroids converge
kmeans_max_iter = 800

# These should match the training pipeline config ('fixed_shape_resizer' param)
width = 512
height = 512

# Get the ground-truth bounding boxes for our dataset
bboxes = xml_to_boxes(path='/home/alvaro/Downloads/new_object_detection/train', rescale_width=width, rescale_height=height)

aspect_ratios, avg_iou_perc =  kmeans_aspect_ratios(
                                      bboxes=bboxes,
                                      kmeans_max_iter=kmeans_max_iter,
                                      num_aspect_ratios=num_aspect_ratios)

aspect_ratios = sorted(aspect_ratios)

print('Aspect ratios generated:', [round(ar,2) for ar in aspect_ratios])
print('Average IOU with anchors:', avg_iou_perc)
