import os
import logging

__LOG_DIR_PATH = "logs"
__LOG_FILE_PATH = os.path.join(__LOG_DIR_PATH, "flexml_logs.log")

def _logger_configuration(logging_to_file: bool = True):
    """
    Configures the logger to save logs to a file or not

    Parameters
    ----------
    logging_to_file : bool, (default=True)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.
    """
    if logging_to_file:
        # Create the log directory if it doesn't exist
        try:
            if not os.path.exists(__LOG_DIR_PATH):
                os.makedirs(__LOG_DIR_PATH)
        except Exception as e:
            print(f"Error creating log directory, no logs will be saved into a .log file, Error: {e}")

        logging.basicConfig(level="INFO",
                            format='%(levelname)s | %(asctime)-3s | %(name)s.%(funcName)s | %(message)-3s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.FileHandler(__LOG_FILE_PATH),
                                logging.StreamHandler()
                            ])
        
    else:
        logging.basicConfig(level="INFO",
                            format='%(levelname)s | %(asctime)-3s | %(name)s.%(funcName)s | %(message)-3s',
                            datefmt="%Y-%m-%d %H:%M:%S",
                            handlers=[
                                logging.StreamHandler()
                            ])

def get_logger(name: str, logging_to_file: bool = True) -> logging.Logger:
    """
    Returns a logger object with the given name

    Parameters
    ----------
    name : str
        The name of the logger (It's always the name of the class or the module)

    Returns
    -------
    logger : logging.Logger
        The logger object with the given name
    """
    _logger_configuration(logging_to_file)
    return logging.getLogger(name)