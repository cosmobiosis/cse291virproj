import numpy as np
import cv2

@profile
def MakeGrayFrame(npimg):
    frame_gray = cv2.cvtColor(cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY)
    return frame_gray

if __name__ == '__main__':
    npimg = np.load("func1_output.npy")
    grayframe = MakeGrayFrame(npimg)
    np.save("func2_A1_framegray", grayframe)