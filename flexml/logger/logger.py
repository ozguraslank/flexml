import os
import logging

__LOG_DIR_PATH = "logs"
__LOG_FILE_PATH = os.path.join(__LOG_DIR_PATH, "flexml_logs.log")

def _logger_configuration(log_level: str, logging_to_file: bool = False):
    """
    Configures the logger to save logs to a file or not.
    
    Parameters
    ----------
    log_level: str,
        The log level to set for the logger. It can be either "TEST" or "PROD"
    
    logging_to_file : bool, (default=False)
        If True, logs are saved to /logs/flexml_logs.log. Otherwise, logs are not saved to a file.
    """
    handlers = [logging.StreamHandler()]
    log_format = None
    
    if log_level == "TEST":
        log_format = '%(levelname)s | %(asctime)-3s | %(name)s.%(funcName)s | %(message)-3s'
    elif log_level == "PROD":
        log_format = '%(levelname)s | %(asctime)-3s | %(message)-3s'
    else:
        raise ValueError("Invalid log level. It should be either 'TEST' or 'PROD'.")
    
    if logging_to_file:
        os.makedirs(__LOG_DIR_PATH, exist_ok=True)
        handlers.append(logging.FileHandler(__LOG_FILE_PATH))

    logging.basicConfig(
        level="INFO",
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )
    
    # Set some of the libraries logging to ERROR level to reduce verbosity
    logging.getLogger('shap').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    logging.getLogger("numexpr").setLevel(logging.ERROR)

def get_logger(
    name: str,
    log_level: str,
    logging_to_file: bool = False
) -> logging.Logger:
    """
    Returns a logger object with the given name

    Parameters
    ----------
    name : str
        The name of the logger (It's always the name of the class or the module)

    log_level: str
        The log level to set for the logger. It can be either "TEST" or "PROD"
        
        Example output for TEST
        >>> logger = get_logger("test_logger", "TEST")
        >>> logger.info("This is a test message")
        >>> 2021-07-07 12:00:00 | test_logger.<module> | This is a test message

        Example output for PROD
        >>> logger = get_logger("test_logger", "PROD")
        >>> logger.info("This is a test message")
        >>> 2021-07-07 12:00:00 | This is a test message

    logging_to_file : bool, (default=False)
        If True, logs are saved to /logs/flexml_logs.log. Otherwise, logs are not saved to a file

    Returns
    -------
    logger : logging.Logger
        The logger object with the given name
    """
    _logger_configuration(log_level, logging_to_file)
    return logging.getLogger(name)