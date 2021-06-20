import cv2
import numpy as np
import os
from non_max_suppression import non_max_suppression_fast as nms

def start_algorithm(flann_index, tree_value, search_params):
    index_params = dict(algorithm=flann_index, trees=tree_value)
    return cv2.FlannBasedMatcher(index_params, search_params)

def add_sample(path, sift, bow_kmeans_trainer):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is not None:
        bow_kmeans_trainer.add(descriptors)

def descriptor_extractor(img, sift, bow_extractor):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)

def image_pyramid(img, scale_factor=1.25, min_size=(100, 40),
            max_size=(600, 600)):
    h, w = img.shape
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            yield img
        w /= scale_factor
        h /= scale_factor
        img = cv2.resize(img, (int(w), int(h)),
                         interpolation=cv2.INTER_AREA)

def sliding_window(img, step=10, window_size=(160, 160)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y+window_h, x:x+window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield (x, y, roi)
