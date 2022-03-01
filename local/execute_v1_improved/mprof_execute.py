from mprof_funcs import *

def server():
	npimg = MakeNPImage("../samples/eye0.jpg")
	eyetracker = MakeDeepEye("/../models/default.ckpt")
	network_model = MakeNetwork(eyetracker)
	SetNetwork(eyetracker, network_model)
	frame, frame_gray = MakeFrameAndGrayFrame(npimg)
	probability_mask = MakeProbabilityMask(eyetracker, frame_gray)
	blob_locs = MakeBlobLocations(probability_mask)
	ret_json = ProcessSingleImage(frame, frame_gray, blob_locs)
	return ret_json

if __name__ == "__main__":
	print(server())