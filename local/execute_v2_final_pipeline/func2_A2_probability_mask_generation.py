import deepeye
import numpy as np

@profile
def MakeDeepEye(model_path):
    return deepeye.DeepEye(model_path)

@profile
def MakeNetwork(eyetracker):
    return deepeye.Network('Deep_eye', eyetracker.input_reshaped_casted, 1, is_training=False, reuse=False, deep=2, layers=16)

@profile
def SetNetwork(eyetracker, network_model):
    eyetracker.setNetwork(network_model)

@profile
def MakeProbabilityMask(eyetracker, frame_gray):
    return eyetracker.run(frame_gray)

if __name__ == '__main__':
    eyetracker = MakeDeepEye("/../models/default.ckpt")
    network_model = MakeNetwork(eyetracker)
    SetNetwork(eyetracker, network_model)
    frame_gray = np.load("func2_A1_framegray.npy")
    probability_mask = MakeProbabilityMask(eyetracker, frame_gray)
    np.save("func2_A2_probability_mask", probability_mask)