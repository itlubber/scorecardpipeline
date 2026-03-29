import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, ERROR


def init_logger(filename=None, stream=True, fmt="[ %(asctime)s ][ %(levelname)s ][ %(filename)s:%(funcName)s:%(lineno)d ] %(message)s", datefmt=None):
    """
    初始化日志

    :param filename: 日志文件存储地址，如果不传不记录日志到文件中，默认为 None
    :param stream: 是否显示在终端中，默认 True
    :param fmt: 日志格式，参考：https://docs.python.org/3/library/logging.html#formatter-objects
    :param datefmt: 日期格式
    :return: logging.Logger
    """
    logger = getLogger("scorecardpipeline")
    logger.setLevel(DEBUG)
    formatter = Formatter(fmt, datefmt=datefmt)

    if filename:
        if os.path.dirname(filename) != "" and not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except Exception as error:
                print(f'错误 >> 创建日志目录失败,清手动创建目录文件位置,运行 sudo mkdir -p {os.path.dirname(filename)}')
                print('错误 >> 报错信息 : {}'.format(error))

        # fh = TimedRotatingFileHandler(filename=filename, when='D', backupCount=30, encoding="utf-8")
        fh = RotatingFileHandler(filename=filename, mode='a', maxBytes=10*1024**2, backupCount=0, encoding="utf-8")
        fh.setLevel(INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        fh.close()
    
    if stream:
        ch = StreamHandler(sys.stdout)
        ch.setLevel(INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        ch.close()

    return logger


if __name__ == '__main__':
    logger = init_logger()
    logger.info(type(logger))
    logger.info("import scorecardpipeline as sp")
