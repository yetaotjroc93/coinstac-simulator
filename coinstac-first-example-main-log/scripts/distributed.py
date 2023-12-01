import inspect
import logging


class GenericLogger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.logger = self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_message(self, message, level=logging.INFO):
        if level == logging.DEBUG:
            frame = inspect.currentframe().f_back
            filename = inspect.getframeinfo(frame).filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            caller_info = f"{filename} - {function_name} - Line {line_number}"
            message = f"{caller_info} - {message}"

        self.logger.log(level, message)