from normal_funcs import *
from pympler import asizeof

def checkSize(label, my_object):
	print(label, asizeof.asizeof(my_object))

def server():
	npimg = MakeNPImage("../samples/eye0.jpg")
	eyetracker = MakeDeepEye("/../models/default.ckpt")
	network_model = MakeNetwork(eyetracker)
	SetNetwork(eyetracker, network_model)
	frame, frame_gray = MakeFrameAndGrayFrame(npimg)
	probability_mask = MakeProbabilityMask(eyetracker, frame_gray)
	blob_locs = MakeBlobLocations(probability_mask)
	ret_json = ProcessSingleImage(frame, frame_gray, blob_locs)

	print("============================")
	print("Memory Object Usage (bytes)")
	checkSize("image", npimg)
	checkSize("eyetracker", eyetracker)
	checkSize("network_model", network_model)
	checkSize("frame", frame)
	checkSize("frame_gray", frame_gray)
	checkSize("probability_mask", probability_mask)
	checkSize("blob_locs", blob_locs)
	print("============================")
	return ret_json

if __name__ == "__main__":
	print(server())