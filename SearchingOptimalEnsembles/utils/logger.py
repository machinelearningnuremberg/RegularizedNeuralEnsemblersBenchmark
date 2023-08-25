import logging

log_level_mapping = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(name: str = "SEO", logging_level: str = "debug") -> logging.Logger:
    # Create a logger
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    logger = logging.getLogger(name)

    # Set the log level
    logger.setLevel(log_level_mapping[logging_level.lower()])

    # # Remove all handlers
    # for handler in logger.handlers[:]:
    #     logger.removeHandler(handler)

    # Create a new handler for outputting log messages to the console
    # console_handler = logging.StreamHandler()

    # # Create a formatter with a custom log format
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # # Set the formatter for the handler
    # console_handler.setFormatter(formatter)

    # Add the handler to the logger
    # logger.addHandler(console_handler)

    return logger
