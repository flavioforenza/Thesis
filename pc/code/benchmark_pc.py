import numpy as np
import cv2
import time
import os 
import pandas as pd
from pyvirtualdisplay import Display


from imutils.video import FPS

def get_fps(network, path_to_onnx, single_input):
    disp = Display(visible=False, size=(100, 60))
    disp.start()
    path_network = 'data/networks/' + network
    if ".mp4" in single_input: #video.mp4
        single_input = "data/" + single_input
        cap = cv2.VideoCapture(single_input)
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total video's frames: ", total_num_frames)
        (major_ver, _, _) = (cv2.__version__).split('.')
        if int(major_ver)  < 3 :
            fps_input = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps_input = cap.get(cv2.CAP_PROP_FPS)
    else: #webcam
        frameWidth = 1280
        frameHeight = 720
        cap = cv2.VideoCapture(int(single_input))
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        cap.set(10,150)
        total_num_frames = 120;
        fps_input = 30   

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
    net_onnx = cv2.dnn.readNetFromONNX(path_network+'/'+path_to_onnx)
    
    timeuout = time.time() + 10
    fps_start_time = 0
    fps = 0
    lst_frame = []
    frame_count = 0

    while time.time()<timeuout:
        _, frame = cap.read()

        try:
            frame.shape[:2]
        except:
            print("Sbagliato")

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
            if confidence > 0.5:
                #do something 
                obj +=1    
                
        fps_inference.update()#fps INFERENCE

        cv2.putText(frame, "FPS: {:.2f}".format(fps), (5,30), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0), 2)#FPS OUTPUT
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key==81 or key ==113:
            break

    fps_inference.stop()
    print("FPS INPUT: {0}".format(fps_input))
    print("FPS OUTPUT: ", max(lst_frame))
    print("FPS INFERENCE: {:.2f}".format(fps_inference.fps()))

    cap.release()
    cv2.destroyAllWindows()

    return fps_input, max(lst_frame), fps_inference.fps()

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
    for input_source in lst_input:
        #stream on display
        input, output , network_inference = get_fps(lst_networks[i], lst_path_onnx[i], input_source)
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Input")] = input
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Output")] = output
        dataframe_display.iloc[dataframe_display.index.get_loc(input_source), dataframe_display.columns.get_loc("Network")] = network_inference


    #save results
    dataframe_display.to_csv("semantic_seg_bench/network: " +  lst_networks[i] + " display")





    






                    
