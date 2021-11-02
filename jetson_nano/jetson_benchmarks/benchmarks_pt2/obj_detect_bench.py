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

def save_Dataframe(method):
	switcher ={
		"video/240p_60fps.mp4": "video 240p_60fps",
		"video/360p_30fps.mp4": "video 360p_30fps",
		"video/480p_30fps.mp4": "video 480p_30fps",
		"video/720p_30fps.mp4": "video 720p_30fps",
		"video/1080p_30fps.mp4": "video 1080p_30fps",
		"video/1080p_60fps.mp4": "video 1080p_60fps",
		"rtp://192.168.1.52:5005": "SSH streaming",
		"display://0": "display",
		"/dev/video1": "video1 streaming",
		"csi://0": "video0 streaming"
	}
	return switcher.get(method, lambda: 'Invalid source')


def get_fps(input_list, output_list, networks, operation):
	for single_input in input_list:
		for single_output in output_list:
			os.environ["DISPLAY"] = ':0'

			# parse the command line
			parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
											formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
											jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

			parser.add_argument("input_URI", type=str, default=single_input, nargs='?', help="URI of the input stream")
			parser.add_argument("output_URI", type=str, default=single_output, nargs='?', help="URI of the output stream")
			parser.add_argument("--network", type=str, default="", help="pre-trained model to load (see below for options)")
			parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
			parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
			parser.add_argument("--input-codec", type=str, default="h264", help="type of output-codec")

			is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

			try:
				opt = parser.parse_known_args()[0]
			except:
				print("")
				parser.print_help()
				sys.exit(0)

			dataframe = pd.DataFrame(columns=["Input", "Output", "Network"], index=networks)

			for network in networks:
				# load the object detection network
				net = jetson.inference.detectNet(network, sys.argv, opt.threshold)
				# create video output object 
				output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
				# create video sources
				input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
				timeuout = time.time() + 10
				lst_frames_input = []
				lst_frames_output = []
				lst_frames_network = []
				#set a timeout usefull to skip before networks
				while time.time()<=timeuout:
					# capture the next image
					img = input.Capture()

					# detect objects in the image (with overlay)
					detections = net.Detect(img, overlay=opt.overlay)

					# render the image
					output.Render(img)
					
					print()
					print("***********************************")
					print("Actual Network: ", network)
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
					
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Input")] = max(lst_frames_input)
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Output")] = max(lst_frames_output)
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Network")] = max(lst_frames_network)

			print(dataframe)
			input_name = save_Dataframe(single_input)
			output_name = save_Dataframe(single_output)
			filename = "benchmarks jetson input: " + input_name + " output: " + output_name +  ".csv"
			dataframe.to_csv(operation+'/'+filename)

networks_detectNet = [
	"ssd-mobilenet-v1",
	"ssd-mobilenet-v2",
	"ssd-inception-v2",
	"pednet",
	"multiped"
]

input_list = [
	# "video/240p_60fps.mp4",
	# "video/360p_30fps.mp4",
	# "video/480p_30fps.mp4",
	# "video/720p_30fps.mp4",
	# "video/1080p_30fps.mp4",
	# "video/1080p_60fps.mp4",
	# "csi://0", 
	"/dev/video1"
			]

output_list = ["display://0", "rtp://192.168.1.52:5005"]

get_fps(input_list, output_list, networks_detectNet, "object_detection")


