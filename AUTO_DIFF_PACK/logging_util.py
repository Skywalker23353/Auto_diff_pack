import logging
import os

# Configure logger
logger = logging.getLogger('AUTO_DIFF_PACK')

def setup_logging(level=logging.INFO, script_dir=None):
    """
    Setup logging to both console and file.
    
    Args:
        level: logging.INFO (minimal) or logging.DEBUG (detailed)
        script_dir: Directory where the calling script is located. If None, uses current working directory.
    
    Logs are stored in logs/auto_diff_pack.log (in script directory) with rotation (max 5 files, 5MB each).
    
    Example:
        >>> import os
        >>> setup_logging(level=logging.INFO, script_dir=os.path.dirname(os.path.abspath(__file__)))
        >>> setup_logging(level=logging.DEBUG, script_dir=os.path.dirname(os.path.abspath(__file__)))
    """
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create logs directory in script directory
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # File handler - mode='w' overwrites log file each time
    log_file = os.path.join(logs_dir, 'auto_diff_pack.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.setLevel(level)


def get_logger():
    """Get the AUTO_DIFF_PACK logger instance."""
    return logger
