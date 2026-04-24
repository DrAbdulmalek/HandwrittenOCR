"""
HandwrittenOCR - محرك التصحيح الإملائي وقاموس التصحيح المستمر
================================================================
تصحيح النصوص حسب اللغة + تعلم من تصحيحات المستخدم.

المميزات:
- تصحيح إملائي عربي (ar-corrector) وإنجليزي (pyspellchecker)
- قاموس تصحيح يتعلم من مراجعات المستخدم (build_correction_dict)
- تطبيق القاموس تلقائياً على الاستخراجات الجديدة
"""

import json
import os
import logging
import pandas as pd
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory

logger = logging.getLogger("HandwrittenOCR")

DetectorFactory.seed = 0

_ar_corrector = None
_en_spellchecker = None


def init_correctors() -> None:
    """تهيئة المدققات الإملائية"""
    global _ar_corrector, _en_spellchecker

    try:
        from ar_corrector.corrector import Corrector
        _ar_corrector = Corrector()
        logger.info("تم تحميل المدقق الإملائي العربي")
    except ImportError:
        logger.warning("ar-corrector غير مثبت. التصحيح العربي لن يكون متاحاً.")

    _en_spellchecker = SpellChecker(language="en")
    logger.info("تم تحميل المدقق الإملائي الإنجليزي")


def correct_text(text: str) -> str:
    """تصحيح إملائي حسب اللغة المكتشفة"""
    if not text or not text.strip():
        return text
    try:
        lang = detect(text)
        if lang == "ar":
            return _correct_arabic(text)
        elif lang == "en":
            return _correct_english(text)
    except Exception:
        pass
    return text


def _correct_arabic(text: str) -> str:
    if _ar_corrector is None:
        return text
    try:
        return _ar_corrector.contextual_correct(text)
    except Exception as e:
        logger.debug(f"خطأ في التصحيح العربي: {e}")
        return text


def _correct_english(text: str) -> str:
    """تصحيح الجمل الإنجليزية كلمة بكلمة مع حفظ الترقيم"""
    if _en_spellchecker is None:
        return text
    try:
        words = text.split()
        corrected = []
        for word in words:
            clean = word.strip(".,;:!?\"'()-")
            if clean:
                fixed = _en_spellchecker.correction(clean)
                corrected_word = word.replace(clean, fixed) if fixed else word
                corrected.append(corrected_word)
            else:
                corrected.append(word)
        return " ".join(corrected)
    except Exception as e:
        logger.debug(f"خطأ في التصحيح الإنجليزي: {e}")
        return text


# ===================== قاموس التصحيح المستمر =====================

def build_correction_dict(
    feedback_csv: str,
    correction_dict_path: str,
    min_votes: int = 2
) -> dict:
    """
    بناء قاموس تصحيح من تصحيحات المستخدم السابقة.

    يقرأ ملف feedback CSV ويبني قاموساً يربط بين
    النص الأصلي والتصحيح الأكثر شيوعاً (مع حد أدنى
    من الأصوات لضمان الموثوقية).

    Parameters:
        feedback_csv: مسار ملف تصحيحات المستخدم
        correction_dict_path: مسار حفظ القاموس
        min_votes: الحد الأدنى من التصحيحات المتطابقة

    Returns:
        قاموس {original_text: corrected_text}
    """
    if not os.path.exists(feedback_csv):
        return {}

    try:
        df_fb = pd.read_csv(feedback_csv, encoding="utf-8")

        # عد التصحيحات لكل كلمة
        corrections = {}
        for _, row in df_fb.iterrows():
            orig = str(row["original_text"]).strip()
            corr = str(row["corrected_text"]).strip()
            if orig and corr and orig != corr:
                if orig not in corrections:
                    corrections[orig] = {}
                corrections[orig][corr] = corrections[orig].get(corr, 0) + 1

        # اختيار التصحيح الأكثر شيوعاً (مع حد أدنى)
        final_dict = {}
        for orig, candidates in corrections.items():
            best = max(candidates, key=candidates.get)
            if candidates[best] >= min_votes:
                final_dict[orig] = best

        # حفظ القاموس
        os.makedirs(os.path.dirname(correction_dict_path), exist_ok=True)
        with open(correction_dict_path, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"تم تحديث قاموس التصحيح: {len(final_dict)} كلمة")
        return final_dict

    except Exception as e:
        logger.error(f"خطأ في بناء القاموس: {e}")
        return {}


def load_correction_dict(correction_dict_path: str) -> dict:
    """
    تحميل قاموس التصحيح من الملف.

    Parameters:
        correction_dict_path: مسار ملف القاموس

    Returns:
        قاموس {original_text: corrected_text}
    """
    if not os.path.exists(correction_dict_path):
        return {}
    try:
        with open(correction_dict_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"خطأ في تحميل القاموس: {e}")
        return {}


def apply_correction_dict(text: str, correction_dict: dict) -> str:
    """
    تطبيق قاموس التصحيح على نص.

    Parameters:
        text: النص المراد تصحيحه
        correction_dict: قاموس التصحيح

    Returns:
        النص المصحح
    """
    if not correction_dict or not text:
        return text
    words = text.split()
    corrected = [correction_dict.get(w, w) for w in words]
    return " ".join(corrected)
