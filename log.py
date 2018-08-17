#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017-2087 TIC-Recruit Project
# Author: Chao Ma <machao13@baidu.com>
"""
This module contains log module
"""

import os
import logging
import logging.handlers
import datetime




def init_log(log_fold=None, log_name=None, level=logging.INFO, when="D", backup=7,
             format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
             datefmt="%m-%d %H:%M:%S"):
    """
    init_log - initialize log module

    Args:
      log_path      - Log file path prefix.
                      Log data will go to two files: log_path.log and log_path.log.wf
                      Any non-exist parent directories will be created automatically
      level         - msg above the level will be displayed
                      DEBUG < INFO < WARNING < ERROR < CRITICAL
                      the default value is logging.INFO
      when          - how to split the log file by time interval
                      'S' : Seconds
                      'M' : Minutes
                      'H' : Hours
                      'D' : Days
                      'W' : Week day
                      default value: 'D'
      format        - format of the log
                      default format:
                      %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
                      INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
      backup        - how many backup file to keep
                      default value: 7

    Raises:
        OSError: fail to create log directories
        IOError: fail to open log file
    """
    log_path = log_fold + log_name
    formatter = logging.Formatter(format, datefmt)
    # 获得对应log_name 的log实例， 如果没有log_name的话， 会导致多个模块的log写入同样的内容
    logger = logging.getLogger(log_name)
    # 如果 logger.handlers列表为空， 则添加， 否则，直接去写日志； 如果不添加这句， 反复运行server， 就会有多个logger实例， 会导致同一个日志中写入了多条
    """ 之前同样的操作却没有产生好的结果是因为， 我在后面纪录日志的时候使用的是 logging.info ， 而不是log函数中的 logger 实例，
    我自己理解的是，直接使用logging并没有使用这里的实例， 所以写入不了数据， 除非将log_name设置为空， 这样的话， 在log函数中实例出来的logger名字
    和直接使用logging模块应该是对应这一个logger实例。     
    """
    if not logger.handlers:
        logger.setLevel(level)

        dir = os.path.dirname(log_path)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        ####   加入一个 console端的输出  #########
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    #         logging.getLogger().addHandler(console)
        ########################################

        now_ = '_'+datetime.datetime.now().strftime('%Y-%m-%d')
        handler = logging.handlers.TimedRotatingFileHandler(log_path+ now_ + ".log",
                                                            when=when,
                                                            backupCount=backup)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.handlers.TimedRotatingFileHandler(log_path + now_ +  ".log.wf",
                                                            when=when,
                                                            backupCount=backup)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


if __name__=='__main__':
    init_log()
    logging.info('hhh')
    logging.warning('nicai')
