import logging
import os


class GeneralLogger:
    def __init__(self, level=logging.INFO, name='default', path='../log/default.log',
                 log_format='%(asctime)s [%(threadName)s] [%(name)s] [%(levelname)s] %(filename)s[line:%(lineno)d] %('
                            'message)s',
                 date_format='%Y-%m-%d %H:%M:%S'):
        """
        settings
        """
        self.log_format = log_format
        self.date_format = date_format
        self.log_level = level
        self.log_name = name
        self.log_path = path
        self.log = logging.getLogger(self.log_name)
        self.log.setLevel(self.log_level)
        # # create file handler
        if not os.path.exists(self.log_path):
            # # log_path detected to avoid the logger chaos
            logging.warning("Warning: Path to record logging doesn't exist!")
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setLevel(self.log_level)
        self.file_handler.setFormatter(logging.Formatter(self.log_format, self.date_format))
        self.log.addHandler(self.file_handler)
        self.info = self.log.info
        self.debug = self.log.debug
        self.warning = self.log.warning
        self.error = self.log.error
        self.critical = self.log.critical


if __name__ == "__main__":
    SpiderLogger = GeneralLogger(name="SpiderLogging")
    SpiderLogger.error("Error: No this Image!")
