import logging


def logging_exception(func):
    logger = logging.getLogger(__name__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(e)
            raise e

    return wrapper
