#!/usr/bin/env python

import cv2
import time

video = cv2.VideoCapture(0);

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

num_frames = 120;

print("Capturing {0} frames".format(num_frames))

start = time.time()

timeuout = time.time() + 10
fps_start_time = 0
fps = 0
fps_output = []
while time.time()<timeuout:
    ret, frame = video.read()

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1/time_diff
    fps_start_time = fps_end_time
    fps_output.append(fps)

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (5,30), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key==81 or key ==113:
        break

print("MAX FPS: ", max(fps_output))
video.release()