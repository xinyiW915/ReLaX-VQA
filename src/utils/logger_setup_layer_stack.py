import logging

def setup_logger(log_file_path):
    # Get the logger for the current module
    logger = logging.getLogger(__name__)

    # Check if the logger has handlers and clear them if it does
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler and add it to the logger (use 'w' to overwrite existing file)
    file_handler = logging.FileHandler(log_file_path, 'w')
    logger.addHandler(file_handler)

    # Set the logging level to DEBUG
    logger.setLevel(logging.DEBUG)

    # Set the formatter for the file handler
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    return logger

log_file_path = './utils/log_debug_layer_stack.log'
logger = setup_logger(log_file_path)



# # use the logger for logging messages
# logger.debug('This is a debug message.')
# logger.info('This is an info message.')
# logger.warning('This is a warning message.')
# logger.error('This is an error message.')