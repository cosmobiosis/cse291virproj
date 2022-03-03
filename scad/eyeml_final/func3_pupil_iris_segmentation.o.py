#@ type: compute
#@ parents:
#@   - func2_A1_frame_gray_generation
#@   - func2_A3_blob_locations_generation
#@   - func2_B1_frame_generation
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma

import pickle
import random
import numpy as np
import json
import base64
from random import randrange
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array

import cv2

def main(params, action):
    # read metadata to setup
    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)

    context_dict_A3 = pickle.loads(base64.b64decode(params["func2_A3_blob_locations_generation"][0]['meta']))
    buffer_pool_A3 = buffer_pool_lib.buffer_pool({'mem1':trans}, context_dict_A3["buffer_pool_metadata"])
    coords = remote_array(buffer_pool_A3, metadata=context_dict_A3["remote_coord"]).materialize()
    coords = coords.tolist()
    bw_img = remote_array(buffer_pool_A3, metadata=context_dict_A3["remote_bw_image"]).materialize()

    context_dict_A1 = pickle.loads(base64.b64decode(params["func2_A1_frame_gray_generation"][0]['meta']))
    buffer_pool_A1 = buffer_pool_lib.buffer_pool({'mem1':trans}, context_dict_A1["buffer_pool_metadata"])
    frame_gray = remote_array(buffer_pool_A1, metadata=context_dict_A1["remote_frame_gray"]).materialize()

    context_dict_B1 = pickle.loads(base64.b64decode(params["func2_B1_frame_generation"][0]['meta']))
    buffer_pool_B1 = buffer_pool_lib.buffer_pool({'mem1':trans}, context_dict_B1["buffer_pool_metadata"])
    frame = remote_array(buffer_pool_B1, metadata=context_dict_B1["remote_frame"]).materialize()

    contours, _ = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [[]] * len(contours)
    pupilRadiuses = [0] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], pupilRadiuses[i] = cv2.minEnclosingCircle(contours_poly[i])

    pupilX = pupilY = pupilR = float('-inf')
    for i in range(len(contours)):
        if 2 * pupilRadiuses[i] > bw_img.shape[1] / 2:
            # filter big diameter
            continue
        if pupilRadiuses[i] > pupilR:
            pupilX = int(centers[i][0])
            pupilY = int(centers[i][1])
            pupilR = int(pupilRadiuses[i])
    pupilX = -1 if pupilX == float('-inf') else pupilX
    pupilY = -1 if pupilY == float('-inf') else pupilY
    pupilR = -1 if pupilR == float('-inf') else pupilR

    # IRIS!!!!!
    irisCannyThreshold = 60
    irisAccumulatorThreshold = 40
    irisRadMin = 3 * int(frame_gray.shape[1] / 40)
    irisRadMax = 10 * int(frame_gray.shape[1] / 40)
    irisCircles = cv2.HoughCircles(cv2.cvtColor(cv2.medianBlur(frame,5), cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 1,
                                param1=irisCannyThreshold, param2=irisAccumulatorThreshold, minRadius=irisRadMin, maxRadius=irisRadMax)
    irisCircles = np.uint16(np.around(irisCircles))

    irisX = irisY = irisR = -1
    for candidate in irisCircles[0, :]:
        tempIrisX, tempIrisY, tempIrisR = candidate[0], candidate[1],candidate[2]
        curDist = np.sqrt(np.sum(np.square(np.array([tempIrisX, tempIrisY]) - np.array([pupilX, pupilY]))))
        if curDist + pupilR >= tempIrisR:
            # wrong location for iris
            continue
        irisX, irisY, irisR = tempIrisX, tempIrisY, tempIrisR
        break

    height = frame_gray.shape[0]
    width = frame_gray.shape[1]
    return {
        "pupilIrisRatio" : float(pupilR) / float(irisR),
        "imgRatio": float(height) / float(width),
        "pupil": {
            "r" : float(pupilR) / float(width),
            "x" : float(pupilX) / float(width),
            "y": float(pupilY) / float(height),
        },
        "iris": {
            "r": float(irisR) / float(width),
            "x": float(irisX) / float(width),
            "y": float(irisY) / float(height),
        }
    }