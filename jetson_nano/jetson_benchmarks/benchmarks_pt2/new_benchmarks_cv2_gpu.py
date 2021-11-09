from os import pathconf_names
import numpy as np
import argparse
import imutils
import time
import cv2
import statistics as st
import os

os.environ["DISPLAY"] = ':0'
path_classes = "networks/FCN-ResNet18-Cityscapes-512x256/classes.txt"
CLASSES = open(path_classes).read().strip().split('\n')
path_colors = "networks/FCN-ResNet18-Cityscapes-512x256/colors.txt"
COLORS_1 = open(path_colors).read().strip().split("\n")
COLORS_1 = [np.array(c.split(",")).astype("int") for c in COLORS_1]
CL_tmp = []
for color in COLORS_1:
    CL_tmp.append(np.array(color, dtype="uint8"))
COLORS_1 = np.array(CL_tmp)    

path_to_onnx = "networks/FCN-ResNet18-Cityscapes-512x256/fcn_resnet18.onnx"
print("[INFO] loading model...")
net = cv2.dnn.readNetFromONNX(path_to_onnx)

vs = cv2.VideoCapture("video/360p_30fps.mp4")
total_num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

time_elap = []
fps_output = []

timeStamp = time.time()
fpsFilter=0
font = cv2.FONT_HERSHEY_SIMPLEX

timeuout = time.time() + 10
frames = 0
while frames <= total_num_frames:
    (grabbed, frame) = vs.read()

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    frame = gpu_frame.download()

    frames += 1 
    print("Current frame: ", frames, " on: ", total_num_frames)
    if not grabbed:
        break
    frame = imutils.resize(frame, width=1280)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 256), 0, swapRB=True, crop=False)
    net.setInput(blob)
    
    start = time.time()
    #inference
    output = net.forward()
    end = time.time()
    
    #compute the fps inference 
    elap = (end - start)
    time_elap.append(elap)

    (numClasses, height, width) = output.shape[1:4]
    classMap = np.argmax(output[0], axis=0)
    mask = COLORS_1[classMap]

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")

    dt = time.time()-timeStamp
    timeStamp = time.time()
    fps = 1/dt
    fpsFilter=.9*fpsFilter+.1*fps
    fps_output.append(fpsFilter)
    cv2.putText(output, str(round(fpsFilter,1))+ ' fps', (0,30), font, 1,(0,0,255), 2)
    cv2.imshow("Frame", output) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("[INFO] FPS INFERENCE: {:.2f}".format(1/(sum(time_elap)/len(time_elap))))
print("[INFO] FPS OUTPUT: {:.2f}".format(max(fps_output)))
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()