import logging
import os

def setup_logger(log_file=None, level=logging.INFO, logger_name='NeRFLogger'):
    """
    Set up a logger that outputs messages to both the console and (optionally) a file.

    Args:
        log_file (str, optional): Path to the log file. If provided, logs will also be written to this file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        logger_name (str): Name of the logger.
    
    Returns:
        logger (logging.Logger): Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Example usage:
if __name__ == "__main__":
    logger = setup_logger(log_file="logs/training.log", level=logging.DEBUG)
    logger.info("Logger is set up and working.")
