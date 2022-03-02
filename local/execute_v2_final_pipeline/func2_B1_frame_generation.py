import numpy as np
import cv2

@profile
def MakeFrame(npimg):
    frame = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return frame

if __name__ == '__main__':
    npimg = np.load("func1_output.npy")
    frame = MakeFrame(npimg)
    np.save("func2_B1_frame", frame)