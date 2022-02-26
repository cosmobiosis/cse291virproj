import deepeye
import numpy as np


imgFile = open("../samples/eye0.jpg", "rb")
img_data = imgFile.read()
npimg = np.frombuffer(img_data, dtype=np.uint8)
print(npimg)

eyetracker = deepeye.DeepEye("/../models/default.ckpt")
ret_json = eyetracker.processSingleImage(npimg)
print(ret_json)