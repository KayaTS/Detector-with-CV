import cv2
import numpy as np
import os

''' ~Video copy
videoCapture = cv2.VideoCapture('MyInputVid.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

success, frame = videoCapture.read()
while success:  # Loop until there are no more frames.
    videoWriter.write(frame)
    success, frame = videoCapture.read()
'''

''' ~Camera record
cameraCapture = cv2.VideoCapture(0)
fps = 30  # An assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'CamOutputVideo.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1  # 10 seconds of frames
while numFramesRemaining > 0:
    if frame is not None:
        videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
'''