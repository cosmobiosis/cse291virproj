#@ type: compute
#@ parents:
#@   - func1_event_entry
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
    context_dict_in_b64 = params["func1_event_entry"][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = pickle.loads(context_dict_in_byte)

    buffer_pool = buffer_pool_lib.buffer_pool({'mem1':trans}, context_dict["buffer_pool_metadata"])
    load_np_img = remote_array(buffer_pool, metadata=context_dict["remote_input"])
    npimg = load_np_img.materialize()

    np_frame = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    # update context
    remote_input = remote_array(buffer_pool, input_ndarray=np_frame, transport_name = 'mem1')
    remote_input_metadata = remote_input.get_array_metadata()
    context_dict = {}
    context_dict["remote_frame"] = remote_input_metadata
    context_dict["buffer_pool_metadata"] = buffer_pool.get_buffer_metadata()

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}