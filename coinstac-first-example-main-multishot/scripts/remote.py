import numpy as np
from ancillary import list_recursive
import logging
from distributed import GenericLogger

log_path = '/output/local.log'
logger = GenericLogger(log_path)


def remote_1(args):

    input_list = args["input"]
    myval = np.mean([input_list[site]["output_val"] for site in input_list])
    
    total_step = np.inf
    for site in input_list:
        total_step = min(total_step, input_list[site]["total_step"])
        
    for site in input_list:
        if input_list[site]["current_step"] == total_step:
            computation_output = {"output": {"output_list": myval}, "success": True}
        else:
            computation_output = {
                "output": {
                    "output_list": myval,
                    "current_step": input_list[site]["current_step"]+1, 
                    "total_step": total_step,
                    "covariates": input_list[site]["covariates"],
                    "computation_phase": 'remote_1'
                }
            }
        
    logger.log_message('PARAM_DICT: {}'.format(args), level=logging.INFO)
    logger.log_message('computation_output: {}'.format(computation_output), level=logging.INFO)
    
    return computation_output


def start(PARAM_DICT):
    PHASE_KEY = list(list_recursive(PARAM_DICT, "computation_phase"))

    if "local_1" in PHASE_KEY:
        logger.log_file_path = PARAM_DICT['state']['outputDirectory']+'/local.log'
        logger.logger = logger._configure_logger()
        
        return remote_1(PARAM_DICT)
    else:
        raise ValueError("Error occurred at Remote")