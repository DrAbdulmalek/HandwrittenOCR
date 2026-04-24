"""
HandwrittenOCR - إعداد نظام التسجيل
======================================
وحدة مركزية لإعداد logging بدلاً من التكرار في كل ملف.
"""

import logging
import os
from config import Config


def setup_logging(config: Config) -> logging.Logger:
    """
    إعداد نظام التسجيل مع ملف وسجل الشاشة.

    Parameters:
        config: كائن الإعدادات

    Returns:
        كائن Logger جاهز للاستخدام
    """
    config.ensure_dirs()

    logger = logging.getLogger("HandwrittenOCR")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # تنظيف المعالجات السابقة إن وجدت
    logger.handlers.clear()

    # تنسيق الرسائل
    formatter = logging.Formatter(config.log_format)

    # معالج ملف السجل
    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # معالج الشاشة
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
