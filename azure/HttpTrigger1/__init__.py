from email import header
import logging
import json
from . import deepeye
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    eyetracker = deepeye.DeepEye()
    req_body = req.get_json()
    ret_json = {}
    ret_code = 200
    try:
        ret_json = eyetracker.processSingleImage(req_body["image_base64"])
    except Exception as e:
        ret_json = { "ERR" : str(e) }
        ret_code = 500
    return func.HttpResponse(
        body=json.dumps(ret_json),
        status_code=ret_code,
        headers={'content-type':'application/json'}
    )