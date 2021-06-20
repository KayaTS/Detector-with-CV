import cv2
import numpy as np
import os
from dedector import *
from non_max_suppression import non_max_suppression_fast

svm = cv2.ml.SVM_create()
svm = cv2.ml.SVM_load('svm_data1.dat')

KDTREE = 1
tree_value = 15
search = {}
#xfeatures2d.
sift = cv2.SIFT_create()
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

testi = 0
testix = 1
test_img_folder = 'testresults/results/demo'
if not os.path.isdir(test_img_folder):
    os.mkdir(test_img_folder)

for t in range(20):
    test_img_path ='test/all/b (%d).jpg' % (testix + 1)
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pos_rects = []
    for resized in image_pyramid(gray_img):
        for x, y, roi in sliding_window(resized):
            descriptors = descriptor_extractor(roi, sift, bow_ext)
            if descriptors is None:
                continue
            prediction = svm.predict(descriptors)
            if prediction[1][0][0] == 1.0:
                raw_prediction = svm.predict(
                    descriptors, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
                score = -raw_prediction[1][0][0]
                if score > 1.80:
                    h, w = roi.shape
                    scale = gray_img.shape[0] / float(resized.shape[0])
                    pos_rects.append([int(x * scale),
                                      int(y * scale),
                                      int((x+w) * scale),
                                      int((y+h) * scale),
                                      score])
    pos_rects = non_max_suppression_fast(np.array(pos_rects), 0.02)
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 255, 255), 2)
        text = '%.2f' % score
        cv2.putText(img, text, (int(x0), int(y0) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite( test_img_folder + '/g (%d).jpg' % (testix+1), img)
    testix = testix + 1
cv2.waitKey(0)


