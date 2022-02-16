#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import argparse
import sys
import time
import pandas as pd
import os

from statistics import mean

input_list = [
	"video/240p_60fps.mp4",
	"video/360p_30fps.mp4",
	"video/480p_30fps.mp4",
	"video/720p_30fps.mp4",
	"video/1080p_30fps.mp4",
	"video/1080p_60fps.mp4",
	"csi://0", 
	"/dev/video1"
			]

def get_fps(input_list, single_output, operation):
    path = 'SSD_Distill_obj_detection/'+operation+'.csv'
    if os.path.isfile(path):
        dataframe = pd.read_csv(path, index_col=[0])
        print(dataframe)
    else:
        dataframe = pd.DataFrame(columns=["Input", "Output", "Video"], index=input_list)

    network='mb1-ssd-distill'
    for x in range(0, len(input_list)):	
        os.environ["DISPLAY"] = ':0'

        # parse the command line
        parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                        formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                        jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

        parser.add_argument("input_URI", type=str, default=input_list[x], nargs='?', help="URI of the input stream")
        parser.add_argument("output_URI", type=str, default=single_output, nargs='?', help="URI of the output stream")
        parser.add_argument("--network", type=str, default="", help="pre-trained model to load (see below for options)")
        parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
        #parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
        parser.add_argument("--input-codec", type=str, default="h264", help="type of output-codec")

        is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

        try:
            opt = parser.parse_known_args()[0]
        except:
            print("")
            parser.print_help()
            sys.exit(0)

        # load the object detection network
        net = jetson.inference.detectNet(argv=['--model=/home/flavio/thesis/jetson_nano/KD/model/SSD_distill_Freeze_1000.onnx', '--labels=/home/flavio/thesis/jetson_nano/KD/checkpoints_ssd_distill/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes', '--threshold=0.5'])
        # create video output object 
        output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
        # create video sources
        input = jetson.utils.videoSource(opt.input_URI)
        lst_frames_input = []
        lst_frames_output = []
        lst_frames_network = []
        #set a timeout usefull to skip before networks
        if input_list[x] != 'csi://0' and input_list[x] != "/dev/video1":
            while True:
                # capture the next image
                img = input.Capture()

                # detect objects in the image (with overlay)
                detections = net.Detect(img, overlay=opt.overlay)

                # render the image
                output.Render(img)
                
                print()
                print("***********************************")
                print("Actual Network: ", network)
                #print("Actual Input: ", single_input)
                print("{:s} | Input Source {:.0f} FPS".format("Video", input.GetFrameRate()))
                print("{:s} | Output Source {:.0f} FPS".format("Display", output.GetFrameRate()))
                print("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
                print("***********************************")
                print()

                lst_frames_input.append(input.GetFrameRate())
                lst_frames_output.append(output.GetFrameRate())
                lst_frames_network.append(net.GetNetworkFPS())

                # exit on input/output EOS
                if not input.IsStreaming() or not output.IsStreaming():
                    break

                # print out performance info
                net.PrintProfilerTimes()
        else:
            timeout= time.time()+10
            while time.time()<timeout:
                # capture the next image
                img = input.Capture()

                # detect objects in the image (with overlay)
                detections = net.Detect(img, overlay=opt.overlay)

                # render the image
                output.Render(img)
                
                print()
                print("***********************************")
                print("Actual Network: ", network)
                #print("Actual Input: ", single_input)
                print("{:s} | Input Source {:.0f} FPS".format("Video", input.GetFrameRate()))
                print("{:s} | Output Source {:.0f} FPS".format("Display", output.GetFrameRate()))
                print("{:s} | Network {:.0f} FPS".format(network, net.GetNetworkFPS()))
                print("***********************************")
                print()

                lst_frames_input.append(input.GetFrameRate())
                lst_frames_output.append(output.GetFrameRate())
                lst_frames_network.append(net.GetNetworkFPS())
                
                # print out performance info
                net.PrintProfilerTimes()
                
        dataframe.iloc[dataframe.index.get_loc(input_list[x]), dataframe.columns.get_loc("Input")] = mean(lst_frames_input)
        dataframe.iloc[dataframe.index.get_loc(input_list[x]), dataframe.columns.get_loc("Output")] = mean(lst_frames_output)
        dataframe.iloc[dataframe.index.get_loc(input_list[x]), dataframe.columns.get_loc("Video")] = mean(lst_frames_network)

        print(dataframe)
        dataframe.to_csv(path)


#output_list = ["display://0", "rtp://192.168.1.52:5005"]
#'--model=/home/flavio/thesis/jetson_nano/KD/model/SSD_distill_Freeze_1000.onnx', '--labels=/home/flavio/thesis/jetson_nano/KD/checkpoints_ssd_distill/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes', '--threshold=0.5'
get_fps(input_list, "display://0", "MAXP_ssd_studet_Distilled_object_detection_mean.csv")


