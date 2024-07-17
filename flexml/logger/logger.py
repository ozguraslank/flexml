import os
import logging

__LOG_DIR_PATH = "logs"
__LOG_FILE_PATH = os.path.join(__LOG_DIR_PATH, "flexml_logs.log")

def _logger_configuration(logging_to_file: bool = True):
    """
    Configures the logger to save logs to a file or not.
    
    Parameters
    ----------
    logging_to_file : bool, (default=True)
        If True, logs are saved to /logs/flexml_logs.log. Otherwise, logs are not saved to a file.
    """
    handlers = [logging.StreamHandler()]
    if logging_to_file:
        os.makedirs(__LOG_DIR_PATH, exist_ok=True)
        handlers.append(logging.FileHandler(__LOG_FILE_PATH))

    logging.basicConfig(
        level="INFO",
        format='%(levelname)s | %(asctime)-3s | %(name)s.%(funcName)s | %(message)-3s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

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