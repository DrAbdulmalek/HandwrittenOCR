"""
HandwrittenOCR - إعادة تجميع الجمل
====================================
إعادة تجميع الكلمات المراجعة إلى جمل مع دعم RTL للعربية.
يدعم حالتي verified و sentence_corrected.
"""

import logging
import pandas as pd
from langdetect import detect

logger = logging.getLogger("HandwrittenOCR")


def reconstruct_sentences(
    db,
    y_tolerance: int = 25,
    include_sentence_corrected: bool = True
) -> list[dict] | None:
    """
    إعادة تجميع الكلمات المراجعة إلى جمل.

    يعمل على الكلمات الموثقة (verified و sentence_corrected) ويجمعها حسب:
    1. رقم الصفحة
    2. موقع Y (الأسطر)
    3. موقع X (مع دعم RTL للعربية)

    Parameters:
        db: كائن قاعدة البيانات
        y_tolerance: الحد الأدنى لتباعد Y للنظر في الكلمات على نفس السطر
        include_sentence_corrected: تضمين الكلمات المصححة على مستوى الجمل

    Returns:
        قائمة من القواميس [{page, text, lang, word_ids}, ...] أو None
    """
    words = db.get_verified()

    # فلترة الكلمات المصححة فقط
    if include_sentence_corrected:
        words = [w for w in words if w.get('status') in ('verified', 'sentence_corrected')]
    else:
        words = [w for w in words if w.get('status') == 'verified']

    if not words:
        logger.info("لا توجد كلمات موثقة لإعادة التجمع")
        return None

    all_sentences = []

    for page in set(w["page_num"] for w in words):
        p_words = [w for w in words if w["page_num"] == page]
        p_words.sort(key=lambda k: (k["y"], k["x"]))

        if not p_words:
            continue

        # تقسيم إلى أسطر
        lines = []
        curr_line = [p_words[0]]
        for i in range(1, len(p_words)):
            row = p_words[i]
            if abs(row["y"] - curr_line[-1]["y"]) <= y_tolerance:
                curr_line.append(row)
            else:
                lines.append(curr_line)
                curr_line = [row]
        lines.append(curr_line)

        # تجميع كل سطر إلى جملة
        for line in lines:
            text_preview = " ".join(str(w["predicted_text"]) for w in line)
            try:
                lang = detect(text_preview)
            except Exception:
                lang = "en"

            # ترتيب RTL للعربية (x تنازلي)
            sorted_line = sorted(
                line,
                key=lambda k: k["x"],
                reverse=(lang == "ar")
            )
            sentence = " ".join(str(w["predicted_text"]) for w in sorted_line)
            word_ids = [w["image_id"] for w in sorted_line]

            all_sentences.append({
                "page": page,
                "text": sentence,
                "lang": lang,
                "word_ids": word_ids,
            })

    if not all_sentences:
        return None

    logger.info(f"تم تجميع {len(all_sentences)} جملة")
    return all_sentences


def derive_word_corrections(original: str, corrected: str) -> list[dict]:
    """
    استخراج تصحيحات على مستوى الكلمات من تصحيح جملة.
    يعمل فقط عندما يتطابق عدد الكلمات.

    Parameters:
        original: الجملة الأصلية
        corrected: الجملة المصححة

    Returns:
        قائمة التصحيحات المشتقة [{original, corrected}, ...]
    """
    orig_words = original.split()
    corr_words = corrected.split()

    if len(orig_words) != len(corr_words):
        return []

    derived = []
    for o, c in zip(orig_words, corr_words):
        if o != c:
            derived.append({"original": o, "corrected": c})

    return derived
