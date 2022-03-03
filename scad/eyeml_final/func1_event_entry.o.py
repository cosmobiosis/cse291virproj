#@ type: compute
#@ dependents:
#@   - func2_A1_frame_gray_generation
#@   - func2_B1_frame_generation
#@ corunning:
#@   mem1:
#@     trans: mem1
#@     type: rdma

import pickle
import numpy as np
import json
import base64
import disaggrt.buffer_pool_lib as buffer_pool_lib
from disaggrt.rdma_array import remote_array
import urllib.request
#import tensorflow as tf

def main(params, action):
    # setup
    trans = action.get_transport('mem1', 'rdma')
    trans.reg(buffer_pool_lib.buffer_size)
    buffer_pool = buffer_pool_lib.buffer_pool({'mem1':trans})

    # loading data
    response = urllib.request.urlopen("http://localhost:8123/eye0.jpg")
    img_data = response.read()
    npimg = np.frombuffer(img_data, dtype=np.uint8)

    remote_input = remote_array(buffer_pool, input_ndarray=npimg, transport_name = 'mem1')
    # update context
    remote_input_metadata = remote_input.get_array_metadata()
    context_dict = {}
    context_dict["remote_input"] = remote_input_metadata
    context_dict["buffer_pool_metadata"] = buffer_pool.get_buffer_metadata()

    context_dict_in_byte = pickle.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}