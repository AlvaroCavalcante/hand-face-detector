"""
This script was used to visualize bounding boxes of different aspect ratios and compare them.
"""

import numpy as np
import cv2


black_img = np.zeros((350, 760, 3), dtype=np.uint8)

def get_aspect_ratio_box(img, aspect_ratio, pos, color):
    xmin, xmax = pos[0], pos[1]

    ymin = 100
    ymax = 100 + int((100 * aspect_ratio))

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    return img

blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)

black_img = get_aspect_ratio_box(black_img, 0.5, (20, 120), red)
black_img = get_aspect_ratio_box(black_img, 0.67, (140, 240), green)

black_img = get_aspect_ratio_box(black_img, 1, (260, 360), red)
black_img = get_aspect_ratio_box(black_img, 0.91, (380, 480), green)


black_img = get_aspect_ratio_box(black_img, 2, (500, 600), red)
black_img = get_aspect_ratio_box(black_img, 1.26, (620, 720), green)

cv2.imwrite('box.jpg', cv2.cvtColor(black_img, cv2.COLOR_BGR2RGB))
