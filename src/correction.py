"""
HandwrittenOCR - محرك التصحيح الإملائي
=========================================
تصحيح النصوص حسب اللغة (عربي أو إنجليزي).

تم تصحيح خطأ مهم: النسخة الأصلية تستخدم english_spell.correction(text)
التي تعمل فقط على كلمة واحدة. النسخة المصححة تعالج الجمل كاملة.
"""

import logging
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory

logger = logging.getLogger("HandwrittenOCR")

# تثبيت البذرة لتوحيد نتائج كشف اللغة
DetectorFactory.seed = 0

# متغيرات على مستوى الوحدة (يتم تهيئتها في init)
_ar_corrector = None
_en_spellchecker = None


def init_correctors() -> None:
    """
    تهيئة المدققات الإملائية.

    يتم استدعاؤها مرة واحدة عند بدء التطبيق.
    """
    global _ar_corrector, _en_spellchecker

    # المدقق العربي
    try:
        from ar_corrector.corrector import Corrector
        _ar_corrector = Corrector()
        logger.info("تم تحميل المدقق الإملائي العربي")
    except ImportError:
        logger.warning(
            "ar-corrector غير مثبت. "
            "التصحيح العربي لن يكون متاحاً. "
            "ثبّته بـ: pip install ar-corrector"
        )

    # المدقق الإنجليزي
    _en_spellchecker = SpellChecker(language="en")
    logger.info("تم تحميل المدقق الإملائي الإنجليزي")


def correct_text(text: str) -> str:
    """
    تصحيح إملائي حسب اللغة المكتشفة.

    Parameters:
        text: النص المراد تصحيحه

    Returns:
        النص المصحح
    """
    if not text or not text.strip():
        return text

    try:
        lang = detect(text)
    except Exception:
        return text

    if lang == "ar":
        return _correct_arabic(text)
    elif lang == "en":
        return _correct_english(text)

    return text


def _correct_arabic(text: str) -> str:
    """تصحيح النص العربي"""
    if _ar_corrector is None:
        logger.debug("المدقق العربي غير متاح")
        return text

    try:
        return _ar_corrector.contextual_correct(text)
    except Exception as e:
        logger.debug(f"خطأ في التصحيح العربي: {e}")
        return text


def _correct_english(text: str) -> str:
    """
    تصحيح النص الإنجليزي - كلمة بكلمة.

    تم تصحيح الخطأ من النسخة الأصلية: كانت تستخدم
    correction() على الجملة كاملة بينما تعمل فقط
    على كلمة واحدة.
    """
    if _en_spellchecker is None:
        return text

    try:
        words = text.split()
        corrected = []
        for word in words:
            # الاحتفاظ بالعلامات الترقيمية
            clean = word.strip(".,;:!?\"'()-")
            if clean:
                fixed = _en_spellchecker.correction(clean)
                if fixed and fixed != clean:
                    logger.debug(f"تصحيح: '{clean}' -> '{fixed}'")
                corrected_word = word.replace(clean, fixed) if fixed else word
                corrected.append(corrected_word)
            else:
                corrected.append(word)
        return " ".join(corrected)
    except Exception as e:
        logger.debug(f"خطأ في التصحيح الإنجليزي: {e}")
        return text
