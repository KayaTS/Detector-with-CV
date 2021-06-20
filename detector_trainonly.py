import cv2
import numpy as np
import os
from dedector import *
from non_max_suppression import non_max_suppression_fast


KDTREE = 1
tree_value = 15
search = {}
sift = cv2.xfeatures2d.SIFT_create()
flann = start_algorithm(KDTREE, tree_value, search)

bow_kmtrainer = cv2.BOWKMeansTrainer(30)

pos_img_folder = '../d - yedek'
neg_img_folder = '../g - Kopya - Kopya'

for i in range(10):
    pos = pos_img_folder + '/a (%d).jpg' % (i+1)
    print(i+1)
    neg = neg_img_folder + '/gray (%d).jpg' % (i+1)
    add_sample(pos, sift, bow_kmtrainer)
    add_sample(neg, sift, bow_kmtrainer)

bow_ext = cv2.BOWImgDescriptorExtractor(sift, flann)
vis_voc = bow_kmtrainer.cluster()
bow_ext.setVocabulary(vis_voc)

data = []
labels = []
for i in range(1000):
    pos = pos_img_folder + '/a (%d).jpg' % (i+1)
    neg = neg_img_folder + '/gray (%d).jpg' % (i+1)
    print(i+1)
    pos_img = cv2.imread(pos, cv2.IMREAD_GRAYSCALE)
    neg_img = cv2.imread(neg, cv2.IMREAD_GRAYSCALE)
    pos_descriptors = descriptor_extractor(pos_img, sift, bow_ext)
    neg_descriptors = descriptor_extractor(neg_img, sift, bow_ext)
    if pos_descriptors is not None:
        data.extend(pos_descriptors)
        labels.append(1)
    if neg_descriptors is not None:
        data.extend(neg_descriptors)
        labels.append(-1)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(10)
svm.train(np.array(data), cv2.ml.ROW_SAMPLE,
          np.array(labels))

svm.save('svm_data2.dat')
