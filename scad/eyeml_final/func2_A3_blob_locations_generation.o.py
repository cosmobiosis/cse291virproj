#@ type: compute
#@ parents:
#@   - func2_A2_probability_mask_generation
#@ dependents:
#@   - func3_pupil_iris_segmentation
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
    context_dict_in_b64 = params["func2_A2_probability_mask_generation"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    buffer_pool = buffer_pool_lib.buffer_pool({'mem1':trans}, context_dict["buffer_pool_metadata"])
    load_np_probability_mask = remote_array(buffer_pool, metadata=context_dict["remote_probability_mask"])
    prob_mask = load_np_probability_mask.materialize()

    factor = prob_mask.size / (288.0 * 384.0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100 * factor
    params.maxArea = 25000 * factor
    params.filterByConvexity = True
    params.minConvexity = 0.1

    detector = cv2.SimpleBlobDetector_create(params)

    found_blob = False
    prob = 0.3 # 0.5
    while found_blob == False:
        image = prob_mask.copy()
        image[image < prob] = 0
        image[image > prob] = 1
        image = (image * 255).astype('uint8')
        image = 255 - image
        keypoints = detector.detect(image)

        if len(keypoints) > 0:

            blob_sizes = []
            for k in keypoints:
                blob_sizes.append(k.size)
            detection = np.argmax(np.asarray(blob_sizes))
            out_coordinate = [int(keypoints[detection].pt[0]), int(keypoints[detection].pt[1])]
            found_blob = True
        else:
            out_coordinate = [0, 0]
            found_blob = False
            prob += -0.05
            if prob < 0.05:
                found_blob = True

    # update context
    remote_input_coord = remote_array(buffer_pool, input_ndarray=np.array(out_coordinate), transport_name = 'mem1')
    remote_input_bw_image = remote_array(buffer_pool, input_ndarray=image, transport_name = 'mem1')
    context_dict = {}
    context_dict["remote_coord"] = remote_input_coord.get_array_metadata()
    context_dict["remote_bw_image"] = remote_input_bw_image.get_array_metadata()
    context_dict["buffer_pool_metadata"] = buffer_pool.get_buffer_metadata()

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}