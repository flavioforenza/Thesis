from os import pathconf_names
import numpy as np
import argparse
import imutils as im
import time
import cv2
import statistics as st
import os
from numpy.lib.function_base import kaiser
import pandas as pd


resolutions ={
    "FCN-ResNet18-Cityscapes-512x256": (512,256),
    "FCN-ResNet18-Cityscapes-1024x512": (1024,512),
    "FCN-ResNet18-Cityscapes-2048x1024": (2048,1024),
    "FCN-ResNet18-Pascal-VOC-320x320": (320,320),
    "FCN-ResNet18-Pascal-VOC-512x320": (512,320)
}

def get_fps(network, path_to_onnx, video):
    os.environ["DISPLAY"] = ':0'
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromONNX(path_to_onnx)

    #comment this two lines to execute code only with cpu
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    time_elap = []
    fps_output_list = []

    timeStamp = time.time()    
    fpsFilter=0
    font = cv2.FONT_HERSHEY_SIMPLEX

    frames = 0

    if video[-1] == "0": #for webcam0
        width=1280
        height=720  
        flip=2
        dispW = 640
        dispH = 480
        camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        total_num_frames = 300
        fps_input = 30
        vs = cv2.VideoCapture(camSet)
    elif video[-1] == "1": #for webcam1 (logitech)
        total_num_frames = 300
        fps_input = 30
        vs = cv2.VideoCapture(1)
    else: #for video.mp4
        vs = cv2.VideoCapture(video)
        total_num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        if str(total_num_frames)[0]=="3":
            fps_input = 30
        else:
            fps_input = 60
        
    while frames <= total_num_frames:
        (grabbed, frame) = vs.read()

        frames += 1 
        print("Current frame: ", frames, " on: ", total_num_frames, " network:", network, " Source: ", video)
        if not grabbed:
            break
        if video[-1] != "4":
            frame = im.resize(frame, width=1280)

        h,w = resolutions[network]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (h, w), 0, swapRB=True, crop=False)
        net.setInput(blob)
        
        start = time.time()
        #inference
        output = net.forward()
        end = time.time()
        
        #compute the fps inference 
        elap = (end - start)
        time_elap.append(elap)

        #(numClasses, height, width) = output.shape[1:4]
        classMap = np.argmax(output[0], axis=0)
        np.random.seed(42)
        CLASSES = open("data/networks/"+network+"/classes.txt").read().strip().split("\n")
        COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3), dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        mask = COLORS[classMap]
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        classMap = cv2.resize(classMap, (frame.shape[1], frame.shape[0]),interpolation=cv2.INTER_NEAREST)
        output = ((0.4 * frame) + (0.6 * mask)).astype("uint8") 

        dt = time.time()-timeStamp
        timeStamp = time.time()
        fps = 1/dt
        fpsFilter=.9*fpsFilter+.1*fps
        fps_output_list.append(fpsFilter)

        cv2.putText(output, str(round(fpsFilter,1))+ ' fps', (0,30), font, 1,(0,0,255), 2)
        cv2.imshow("Frame", output) 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    try:
        fps_inference = 1/(sum(time_elap)/len(time_elap))
    except:
        print("Division by zero")
    print("[INFO] FPS INFERENCE: {:.2f}".format(fps_inference))
    fps_output = max(fps_output_list)
    print("[INFO] FPS OUTPUT: {:.2f}".format(fps_output))
    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()
    return fps_input, fps_output, fps_inference

lst_input = [
	"video/240p_60fps.mp4",
	"video/360p_30fps.mp4",
	"video/480p_30fps.mp4",
	"video/720p_30fps.mp4",
	"video/1080p_30fps.mp4",
	"video/1080p_60fps.mp4",
	"0", 
	"1"
    ]

lst_networks = []
lst_path_onnx = []

list_accepted_folders = ["ResNet18-Cityscapes", "ResNet18-Pascal-VOC"]
dir_path = [path for path in os.walk("data/networks/")]
for i in range(1, len(dir_path)): #start from first
    current_path = dir_path[i]   
    for accepted in list_accepted_folders:
        if accepted in current_path[0]:
            #get the onnx file
            for file in current_path[2]:
                if '.onnx' in file and '.engine' not in file:
                    lst_networks.append(current_path[0].replace('data/networks/',""))
                    lst_path_onnx.append(file)

dataframe_display = pd.DataFrame(columns=["Input", "Output", "Network"], index=lst_input)   
for i in range (0, len(lst_networks)):
    j=3
    input_source =lst_input[j]
    #stream on display
    input, output , network_inference = get_fps(lst_networks[i],"data/networks/"+lst_networks[i]+"/"+lst_path_onnx[i], "data/"+input_source)
    if os.path.isfile(r"semantic_segmentation_cv2/network: {} display.csv".format(lst_networks[i])):
        dataframe = pd.read_csv("semantic_segmentation_cv2/network: {} display.csv".format(lst_networks[i]), index_col=0)
        dataframe.iloc[dataframe.index.get_loc(input_source), dataframe.columns.get_loc("Input")] = input
        dataframe.iloc[dataframe.index.get_loc(input_source), dataframe.columns.get_loc("Output")] = output
        dataframe.iloc[dataframe.index.get_loc(input_source), dataframe.columns.get_loc("Network")] = network_inference
        dataframe.to_csv(r"semantic_segmentation_cv2/network: {} display.csv".format(lst_networks[i]))
    else:
        dataframe_display = pd.DataFrame(np.nan, index=lst_input, columns=["Input", "Output", "Network"]) 
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Input")] = input
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Output")] = output
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Network")] = network_inference
        dataframe_display.to_csv(r"semantic_segmentation_cv2/network: {} display.csv".format(lst_networks[i]))
    