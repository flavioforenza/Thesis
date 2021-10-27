import jetson.inference
import jetson.utils
import time

net = jetson.inference.detectNet("resnet-50", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video1")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("rtp://192.168.1.53:5005") # 'my_video.mp4' for file

method = 0

if method:
	while display.IsStreaming():
		print("HEIGHT: ", camera.GetHeight(), "WIDTH: ", camera.GetWidth())

		img = camera.Capture()

		detections = net.Detect(img)
		display.Render(img)
		#print("Width: ", camera.GetWidth(), " Height: ", camera.GetHeight())
		print("Object Detection | Network {:.0f} FPS".format(display.GetFrameRate()))
		#display.SetStatus("Object Detection | Network {:.0f} FPS".format(display.GetFrameRate()))
else:
	frames = 0
	start =  time.time()
	while display.IsStreaming():
		img = camera.Capture()
		detections = net.Detect(img)
		frames = frames + 1
		now = time.time()
		delta = now - start
		if delta >= 1:
			fps = frames / delta
			start = now
			frames = 0
			print("FPS {0:2f}".format(fps))
