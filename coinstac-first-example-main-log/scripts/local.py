import os
from ancillary import list_recursive
import logging
from distributed import GenericLogger

log_path = '/output/local.log'
logger = GenericLogger(log_path)

def local_1(args):

    input_list = args["input"]
    myFile = input_list["covariates"]

    with open(os.path.join(args["state"]["baseDirectory"], myFile)) as fh:
        myval = fh.readlines()

    myval = list(map(int, myval))
    computation_output = {
        "output": {
            "output_val": myval,
            "computation_phase": 'local_1'
        }
    }

    logger.log_message('PARAM_DICT: {}'.format(args), level=logging.INFO)
    logger.log_message('computation_output: {}'.format(computation_output), level=logging.INFO)
    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if not PHASE_KEY:
        logger.log_file_path = PARAM_DICT['state']['outputDirectory']+'/local.log'
        logger.logger = logger._configure_logger()
        return local_1(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Local")