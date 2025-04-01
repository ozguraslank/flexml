from IPython import get_ipython


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