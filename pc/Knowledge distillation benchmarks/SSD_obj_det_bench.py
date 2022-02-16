import os
import numpy as np
import time
import cv2
import pandas as pd
from statistics import mean

#print(cv2.__version__)

def get_fps(network, path_to_onnx, input_list):
    dataframe = pd.DataFrame(columns=["Input", "Output", "Video"], index=input_list)

    for video in input_list:
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromONNX(path_to_onnx)

        time_elap = []
        fps_output_list = []

        timeStamp = time.time()
        fpsFilter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        frames = 0

        if video[-1] == "0":  # for webcam0 (integrated)
            # width = 1280
            # height = 720
            # flip = 2
            # dispW = 640
            # dispH = 480
            # camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=' + str(
            #     flip) + ' ! video/x-raw, width=' + str(dispW) + ', height=' + str(
            #     dispH) + ', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
            total_num_frames = 300
            fps_input = 30
            cap = cv2.VideoCapture(0)

            #cap = cv2.VideoCapture(camSet)
        elif video[-1] == "1":  # for webcam1 (logitech)
            total_num_frames = 300
            fps_input = 30
            cap = cv2.VideoCapture(1)
        else:  # for video.mp4
            cap = cv2.VideoCapture(video)
            total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if str(total_num_frames)[0] == "3":
                fps_input = 30
            else:
                fps_input = 60

        #fps_inference = FPS().start()
        while frames <= total_num_frames:
            (grabbed, frame) = cap.read()
            frames += 1
            print("Current frame: ", frames, " on: ", total_num_frames, " network:", network, " Source: ", video)
            if not grabbed:
                break
            #if video[-1] != "4":
            #    frame = im.resize(frame, width=1280)

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)

            start = time.time()
            # inference
            detections = net.forward()
            end = time.time()

            # compute the fps inference
            elap = (end - start)
            time_elap.append(elap)

            CLASSES = ["background", "bicycle", "bus", "car", "motorcycle", "person", "traffic light", "traffic sign",
                       "truck"]
            COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

            for i in np.arange(detections.shape[2]):
                confidence = detections[0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            dt = time.time() - timeStamp
            timeStamp = time.time()
            fps = 1 / dt
            fpsFilter = .9 * fpsFilter + .1 * fps
            fps_output_list.append(fpsFilter)

            cv2.putText(detections, str(round(fpsFilter, 1)) + ' fps', (0, 30), font, 1, (0, 0, 255), 2)
            cv2.imshow("Frame", detections)

        try:
            fps_inference = 1 / (sum(time_elap) / len(time_elap))
        except:
            print("Division by zero")

        print("[INFO] FPS INFERENCE: {:.2f}".format(fps_inference))
        fps_output = mean(fps_output_list)
        print("[INFO] FPS OUTPUT: {:.2f}".format(fps_output))
        print("[INFO] cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Input")] = fps_input
        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Output")] = fps_output
        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Video")] = fps_inference

        #print(dataframe)
    return dataframe

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

lst_data_dist = []
lst_data_org = []
for i in range(0, 50):
    dataframe_dist = get_fps('SSD Mobilenet-V1',
                           'SSD_distill_Freeze_1000.onnx',
                           lst_input)
    lst_data_dist.append(dataframe_dist)
    dataframe_org = get_fps('SSD Mobilenet-V1',
                          'teacher-ssd-1000.onnx',
                          lst_input)
    lst_data_org.append(dataframe_org)

#get_fps('SSD Mobilenet-V1', './teacher-ssd-1000.onnx', lst_input, 'fps_object_detection_distill')

def dataframe_bench(path, lst):
    dataframe = pd.DataFrame(columns=["Input", "Output", "Video"], index=lst_input)
    for video in lst_input:
        input = 0
        output = 0
        network = 0
        for i in range(0, len(lst)):
            input += lst[i].iloc[lst[i].index.get_loc(video), lst[i].columns.get_loc("Input")]
            output += lst[i].iloc[lst[i].index.get_loc(video), lst[i].columns.get_loc("Output")]
            network += lst[i].iloc[lst[i].index.get_loc(video), lst[i].columns.get_loc("Video")]

        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Input")] = input / len(lst)
        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Output")] = output / len(lst)
        dataframe.iloc[dataframe.index.get_loc(video), dataframe.columns.get_loc("Video")] = network / len(lst)

    print(dataframe)
    dataframe.to_csv(path)

dataframe_bench('./bench_original.csv', lst_data_org)
dataframe_bench('./bench_distill.csv', lst_data_dist)
