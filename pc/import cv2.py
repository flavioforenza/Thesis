import numpy as np
import argparse
import cv2
import imutils
import onnx
import time

from imutils.video import FPS

path_to_onnx = "data/networks/FCN-ResNet18-Cityscapes-512x256/fcn_resnet18.onnx"

model = onnx.load(path_to_onnx)

def get_fps(network_file, weights, input_lst, classNames):
    for single_input in input_lst:
        #single_input = ""
        # construct the argument parse 
        parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')

        parser.add_argument("--video", type=str, default=single_input, help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", type=str, default=network_file, help='Path to text network file: ''MobileNetSSD_deploy.prototxt for Caffe model or ')
        parser.add_argument("--weights", type=str, default=weights, help='Path to weights: ''MobileNetSSD_deploy.caffemodel for Caffe model or ')
        parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        # Next, open the video file or capture device depending what we choose, also load the model Caffe model.
        # Open video file or capture device. 
        if args.video != "":
            cap = cv2.VideoCapture(args.video)
            fps = FPS().start()
        else:
            frameWidth = 1280
            frameHeight = 720
            cap = cv2.VideoCapture(0)
            cap.set(3, frameWidth)
            cap.set(4, frameHeight)
            cap.set(10,150)
            fps = FPS().start()

        #Load the Caffe model 
        #net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        net_onnx = cv2.dnn.readNetFromONNX(path_to_onnx)

        timeuout = time.time() + 10


        while time.time()<=timeuout:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_resized = cv2.resize(frame,(100,100))
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
            #net.setInput(blob)
            #detections = net.forward()

            net_onnx.setInput(blob)
            detections_onnx = net_onnx.forward()

            #Size of frame resize (300x300)
            cols = frame_resized.shape[1] 
            rows = frame_resized.shape[0]

            #For get the class and location of object detected, 
            # There is a fix index for class, location and confidence
            # value in @detections array .

            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) >= 0:  # Break with ESC 
                break
            fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cap.release()
        cv2.destroyAllWindows()
        

input_list = [
	"data/video/4k_60fps.mp4"
			]

#lables of network
classNames = {
    0:'void',
    1:'ego_vehicle',
    2:'ground',
    3:'road',
    4:'sidewalk',
    5:'building',
    6:'wall',
    7:'fence',
    8:'pole',
    9:'traffic_light',
    10:'traffic_sign',
    11:'vegetation',
    12:'terrain',
    13:'sky',
    14:'person',
    15:'car',
    16:'truck',
    17:'bus',
    18:'train',
    19:'motorcycle',
    20:'bicycle'
}

netwotrk = "data/networks/ped-100/deploy.prototxt"

weights = "data/networks/ped-100/snapshot_iter_70800.caffemodel"

get_fps(netwotrk,weights, input_list, classNames)