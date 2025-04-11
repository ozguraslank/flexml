from IPython import get_ipython
from flexml.logger import get_logger
import warnings
warnings.filterwarnings("ignore")


def is_interactive_notebook():
    """Detects interactive environments including Jupyter and Colab"""
    try:
        # Get the shell class name
        shell = get_ipython().__class__.__name__
        # Both Jupyter and Colab have specific shell names
        if shell in ['ZMQInteractiveShell', 'Shell']:  # ZMQ is for Jupyter, Shell is for Colab
            return True
        return False
    except:
        # get_ipython() will not be defined in non-interactive environments
        return False
    

def check_numpy_dtype_error():
    """
    Checks if the numpy version is compatible with the pandas version in Colab
    """
    logger = get_logger(__name__, "PROD")
    try:
        shell = get_ipython().__class__.__name__
        if shell != "Shell": # If environment is not Colab, no need for this check since It only happens in Colab
            return

        import pandas
    except ValueError as e: # Catch ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
        if 'numpy.dtype size changed' in str(e):
            logger.warning("Colab has cronic version issue, restarting the kernel... (details: https://shorturl.at/ZMJBh)")
            try:
                import os
                os.kill(os.getpid(), 9)
            except: # If it fails, try to exit the program. 
                exit()
        else:
            raise e