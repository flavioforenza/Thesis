#!/usr/bin/python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from segnet_utils import *

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
		"csi://0": "video0 streaming",
		"images/*.jpg":"images"
	}
	return switcher.get(method, lambda: 'Invalid source')

def get_fps(input_list, output_list, networks, operation):
	for single_input in input_list:
		for single_output in output_list:
			os.environ["DISPLAY"] = ':0'

			# parse the command line
			# parse the command line
			parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
											formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
											jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

			parser.add_argument("input_URI", type=str, default=single_input, nargs='?', help="URI of the input stream")
			parser.add_argument("output_URI", type=str, default=single_output, nargs='?', help="URI of the output stream")
			parser.add_argument("--network", type=str, default="", help="pre-trained model to load, see below for options")
			parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
			parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
			parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
			parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
			parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

			is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

			try:
				opt = parser.parse_known_args()[0]
			except:
				print("")
				parser.print_help()
				sys.exit(0)

			dataframe = pd.DataFrame(columns=["Input", "Output", "Network"], index=networks)

			for network in networks:
				# load the segmentation network
				net = jetson.inference.segNet(opt.network, sys.argv)

				# set the alpha blending value
				net.SetOverlayAlpha(opt.alpha)

				# create video output
				output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

				# create buffer manager
				buffers = segmentationBuffers(net, opt)

				# create video source
				input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

				timeuout = time.time() + 10
				lst_frames_input = []
				lst_frames_output = []
				lst_frames_network = []
				#set a timeout usefull to skip before networks
				while time.time()<=timeuout:
					# capture the next image
					img_input = input.Capture()

					# allocate buffers for this size image
					buffers.Alloc(img_input.shape, img_input.format)

					# process the segmentation network
					net.Process(img_input, ignore_class=opt.ignore_class)

					# generate the overlay
					if buffers.overlay:
						net.Overlay(buffers.overlay, filter_mode=opt.filter_mode)

					# generate the mask
					if buffers.mask:
						net.Mask(buffers.mask, filter_mode=opt.filter_mode)

					# composite the images
					if buffers.composite:
						jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
						jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

					# render the output image
					output.Render(buffers.output)

					# update the title bar
					output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

					# print out performance info
					jetson.utils.cudaDeviceSynchronize()
					net.PrintProfilerTimes()

					# compute segmentation class stats
					if opt.stats:
						buffers.ComputeStats()
					
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
					
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Input")] = max(lst_frames_input)
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Output")] = max(lst_frames_output)
				dataframe.iloc[dataframe.index.get_loc(network), dataframe.columns.get_loc("Network")] = max(lst_frames_network)

			print(dataframe)
			input_name = save_Dataframe(single_input)
			output_name = save_Dataframe(single_output)
			filename = "benchmarks jetson input: " + input_name + " output: " + output_name +  ".csv"
			dataframe.to_csv(operation+'/'+filename)

networks_segNet = [
	"fcn-resnet18-cityscapes-512x256",
	"fcn-resnet18-cityscapes-1024x512",
	"fcn-resnet18-cityscapes-2048x1024",
	"fcn-resnet18-voc-320x320",
	"fcn-resnet18-voc-512x320"
]

input_list = [
	# "video/240p_60fps.mp4",
	# "video/360p_30fps.mp4",
	# "video/480p_30fps.mp4",
	# "video/720p_30fps.mp4",
	# "video/1080p_30fps.mp4",
	"video/1080p_60fps.mp4",
	"csi://0", 
	"/dev/video1", 
			]

output_list = ["display://0", "rtp://192.168.1.52:5005"]

get_fps(input_list, output_list, networks_segNet, "semantic_segmentation")




