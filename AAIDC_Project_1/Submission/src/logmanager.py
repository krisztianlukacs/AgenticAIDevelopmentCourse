import logging
import datetime

"""
LogManager handles logging functionalities for the AI assistant.

Usage:
    log_manager = LogManager()
    log_manager.add_logfile("assistant")
    log_manager.write_log("assistant", logging.INFO, "Assistant started.")

"""

class LogManager:
    def __init__(self):
        """Initialize the LogManager."""
        self.logfiles = {}
        self.today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.log_files_path = "./logs/"

    def add_logfile(self, name: str):
        """Adds a logfile for the given name."""
        logfile_path = f"{self.log_files_path}{name}_{self.today_str}.log"
        logger = self.setup_logger(name)
        file_handler = logging.FileHandler(logfile_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.logfiles[name] = logger

    def write_log(self, name: str, level: int, message: str):
        """Writes a log message to the specified logfile."""
        if name not in self.logfiles:
            self.add_logfile(name)
        logger = self.logfiles[name]
        logger.log(level, message)

    def setup_logger(self, name: str) -> logging.Logger:
        """Sets up a logger with the specified name."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        return logger