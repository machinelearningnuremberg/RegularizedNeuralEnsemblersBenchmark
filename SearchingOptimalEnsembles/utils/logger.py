import logging

log_level_mapping = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(name: str = "SEO", logging_level: str = "debug") -> logging.Logger:
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    # Create or retrieve a logger
    logger = logging.getLogger(name)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)-15s - %(levelname)-7s - %(message)s", datefmt="%H:%M"
    )

    if not logger.handlers:
        # Logger does not have handlers, so we assume it hasn't been configured yet.

        # Suppress "No handlers could be found for logger X" warnings
        logger.propagate = False

        # Set the log level
        logger.setLevel(log_level_mapping[logging_level.lower()])

        # Create a handler that writes log messages to the console
        handler = logging.StreamHandler()

        # Attach formatter to handler
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    else:
        # Logger already has handlers, update their log levels and formatters
        for handler in logger.handlers:
            handler.setLevel(log_level_mapping[logging_level.lower()])
            handler.setFormatter(formatter)

    return logger
