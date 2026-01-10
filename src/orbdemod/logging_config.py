import logging

ORBDEMOD_ROOT_NAME = 'orbdemod'
ORBDEMOD_ROOT_LOGGER = logging.getLogger(ORBDEMOD_ROOT_NAME)

ORBDEMOD_ROOT_LOGGER.addHandler(logging.NullHandler())


def get_module_logger(name: str) -> logging.Logger:
    if not name.startswith(ORBDEMOD_ROOT_NAME):
        name = f"{ORBDEMOD_ROOT_NAME}.{name}"
    return logging.getLogger(name)

def enable_logging(level=logging.INFO):

    logger = logging.getLogger(ORBDEMOD_ROOT_NAME)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(name)s: %(levelname)s] %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    logger.setLevel(level)
