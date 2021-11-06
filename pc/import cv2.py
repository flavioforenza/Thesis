import numpy as np
import argparse
import cv2
import imutils
import onnx
import time

from imutils.video import FPS

path_to_onnx = "data/networks/FCN-ResNet18-Cityscapes-512x256/fcn_resnet18.onnx"

def get_fps(network_file, weights, input_lst, classNames):
    for single_input in input_lst:
        # single_input = ""
        # construct the argument parse 
        parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')

        parser.add_argument("--video", type=str, default=single_input, help="path to video file. If empty, camera's stream will be used")
        parser.add_argument("--prototxt", type=str, default=network_file, help='Path to text network file: ''MobileNetSSD_deploy.prototxt for Caffe model or ')
        parser.add_argument("--weights", type=str, default=weights, help='Path to weights: ''MobileNetSSD_deploy.caffemodel for Caffe model or ')
        parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        if args.video != "": #video.mp4
            cap = cv2.VideoCapture(args.video)
            total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total video's frames: ", total_num_frames)
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            if int(major_ver)  < 3 :
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            else :
                fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS INFPUT:{0}".format(fps))
        else: #webcam
            frameWidth = 1280
            frameHeight = 720
            cap = cv2.VideoCapture(0)
            total_num_frames = 120;   
        
        #net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        net_onnx = cv2.dnn.readNetFromONNX(path_to_onnx)
        
        timeuout = time.time() + 10
        fps_start_time = 0
        fps = 0
        lst_frame = []
        frame_count = 0

        while time.time()<timeuout:
            ret, frame = cap.read()
            frame_count +=1

            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = 1/time_diff
            lst_frame.append(fps)
            fps_start_time = fps_end_time

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (h, w), 127.5)

            net_onnx.setInput(blob)     

            detections = net_onnx.forward()

            #start fps inference count 
            fps_inference = FPS().start()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    #do something
                    fps_inference.update()#fps INFERENCE         
                    cv2.putText(frame, "FPS: {:.2f}".format(fps), (5,30), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0), 2)#FPS OUTPUT
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1)
                    if key==81 or key ==113:
                        break

        fps_inference.stop()
        print("FPS INFERENCE: {:.2f}".format(fps_inference.fps()))
        print("FPS OUTPUT: ", max(lst_frame))
        cap.release()
        cv2.destroyAllWindows()
        

input_list = [
	"data/video/1080p_60fps.mp4"
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