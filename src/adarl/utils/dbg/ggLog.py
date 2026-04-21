#!/usr/bin/env python3

import os
import sys
import logging
import datetime


import logging

class ColoredLevelsFormatter(logging.Formatter):

    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self, fmt : str, datefmt):
        super().__init__(fmt, datefmt=datefmt)
        self._sub_format = fmt
        self._date_format = datefmt


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey) + self._sub_format + self.reset
        formatter = logging.Formatter(log_fmt, datefmt=self._date_format)
        return formatter.format(record)

logger = logging.getLogger('GGLog')
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler(sys.__stderr__)
levels = {"DEBUG":logging.DEBUG,
          "INFO":logging.INFO,
          "WARNING":logging.WARNING,
          "ERROR":logging.ERROR,
          "CRITICAL":logging.CRITICAL}
level = levels.get(os.environ.get("GGLOG_LEVEL","INFO").upper(), logging.INFO)
ch.setLevel(level)
# create formatter and add it to the handlers
console_formatter = ColoredLevelsFormatter('[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y%m%d%H:%M:%S')
file_formatter = logging.Formatter('[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%Y%m%d%H:%M:%S')
# formatter = logging.Formatter('[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s', datefmt='%s')
ch.setFormatter(console_formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.propagate = False

runid = ""

def setId(id : str):
    global runid
    runid = id

def getId():
    return runid


def _addId(msg):
    # ros_master_uri = os.environ['ROS_MASTER_URI'].split(":")[-1]
    # if ros_master_uri is None:
    #     return "[] "+msg
    # else:
    #     return "["+str(ros_master_uri)+"] "+msg
    return f"[{runid}p{os.getpid()}] "+msg


class _StreamToLogger:
    def __init__(self, level: int, passthrough_stream):
        self._level = level
        self._passthrough_stream = passthrough_stream
        self._buffer = ""
        self.encoding = getattr(passthrough_stream, "encoding", "utf-8")
        self.errors = getattr(passthrough_stream, "errors", None)

    def __getattr__(self, name):
        return getattr(self._passthrough_stream, name)

    def _save_line(self, line: str):
        record = logger.makeRecord(logger.name, self._level, __file__, 0, _addId(line), (), None)
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            if self._level < handler.level:
                continue
            handler.handle(record)

    def write(self, data):
        if not data:
            return 0
        self._passthrough_stream.write(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._save_line(line.rstrip("\r"))
        return len(data)

    def flush(self):
        self._passthrough_stream.flush()
        if self._buffer:
            self._save_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def isatty(self):
        return self._passthrough_stream.isatty()

    def writable(self):
        return self._passthrough_stream.writable()


def captureStdStreams():
    if not isinstance(sys.stdout, _StreamToLogger):
        sys.stdout = _StreamToLogger(logging.INFO, sys.__stdout__)
    if not isinstance(sys.stderr, _StreamToLogger):
        sys.stderr = _StreamToLogger(logging.ERROR, sys.__stderr__)

def _log_error_fallback(e,msg):
    sys.__stderr__.write(f"logging failed with exception {e}. Msg:\n")
    sys.__stderr__.write(f"{msg}\n")
    sys.__stderr__.flush()

def debug(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.debug(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def info(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.info(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def warn(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.warning(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def error(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.error(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def critical(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.critical(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def exception(msg, *args, **kwargs):
    msg = _addId(msg)
    try:
        logger.exception(msg, *args, **kwargs)
    except Exception as e:
        _log_error_fallback(e, msg)

def addLogFile(path :str, level = logging.DEBUG, capture_std: bool = False):
    fh = logging.FileHandler(path)
    fh.setLevel(level)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    if capture_std:
        captureStdStreams()
