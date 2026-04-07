import logging
import sys
import atexit
from pathlib import Path


def _cleanup():
    """防止程序结束时 multiprocessing 资源追踪器报错"""
    pass


atexit.register(_cleanup)


def setup_logger(name: str = "find-skills", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()