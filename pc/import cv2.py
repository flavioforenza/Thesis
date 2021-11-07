import numpy as np
import argparse
import cv2
import imutils
import time
import os 

from imutils.video import FPS

def get_fps(path_to_onnx, input_lst):
    for single_input in input_lst:
        single_input = ""
        # construct the argument parse 
        parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')

        parser.add_argument("--video", type=str, default=single_input, help="path to video file. If empty, camera's stream will be used")
        #parser.add_argument("--prototxt", type=str, default=network_file, help='Path to text network file: ''MobileNetSSD_deploy.prototxt for Caffe model or ')
        #parser.add_argument("--weights", type=str, default=weights, help='Path to weights: ''MobileNetSSD_deploy.caffemodel for Caffe model or ')
        parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold to filter out weak detections")
        args = parser.parse_args()

        if args.video != "": #video.mp4
            cap = cv2.VideoCapture(args.video)
            total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total video's frames: ", total_num_frames)
            (major_ver, _, _) = (cv2.__version__).split('.')
            if int(major_ver)  < 3 :
                fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            else :
                fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS INFPUT:{0}".format(fps))
        else: #webcam
            frameWidth = 1280
            frameHeight = 720
            cap = cv2.VideoCapture(1)
            total_num_frames = 120;
            fps_web = 30   
        
        #net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
        net_onnx = cv2.dnn.readNetFromONNX(path_to_onnx)
        
        timeuout = time.time() + 10
        fps_start_time = 0
        fps = 0
        lst_frame = []
        frame_count = 0

        while time.time()<timeuout:
            _, frame = cap.read()
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
            obj = 0

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args.thr:
                    #do something 
                    obj +=1    
                    
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
	"data/video/4k_60485=fps.mp4"
			]

list_accepted_folders = ["ResNet18-Cityscapes", "ResNet18-Pascal-VOC"]
dir_path = [path for path in os.walk("data/networks/")]
for i in range(1, len(dir_path)): #start from first
    current_path = dir_path[i]   
    for accepted in list_accepted_folders:
        if accepted in current_path[0]:
            #get the onnx file
            for file in current_path[2]:
                if '.onnx' in file and '.engine' not in file:
                    get_fps(current_path[0]+"/"+file, input_list)