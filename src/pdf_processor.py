"""
HandwrittenOCR - معالجة PDF
=============================
استخراج الكلمات من ملفات PDF وحفظ النتائج في قاعدة البيانات.
تم فصل منطق المعالجة عن الواجهة لتسهيل الاختبار والصيانة.
"""

import cv2
import json
import time
import logging
from datetime import datetime
from pdf2image import convert_from_path
import numpy as np
import pandas as pd

from config import Config
from src.preprocessing import preprocess_image, extract_word_bounding_boxes
from src.recognition import OCREngine
from src.correction import correct_text
from src.database import HandwritingDB

logger = logging.getLogger("HandwrittenOCR")


class PDFProcessor:
    """
    معالج ملفات PDF يستخرج الكلمات ويحفظها في قاعدة البيانات.
    """

    def __init__(
        self,
        config: Config,
        ocr_engine: OCREngine,
        db: HandwritingDB
    ):
        """
        تهيئة معالج PDF.

        Parameters:
            config: إعدادات المشروع
            ocr_engine: محرك التعرف
            db: قاعدة البيانات
        """
        self.config = config
        self.ocr = ocr_engine
        self.db = db

    def process(self) -> dict:
        """
        معالجة ملف PDF كاملاً.

        Returns:
            قاموس بالإحصائيات
        """
        start_time = time.time()
        pages_start = self.config.pages_start
        pages_end = self.config.pages_end

        logger.info(f"بدء معالجة: {self.config.pdf_path}")
        logger.info(f"نطاق الصفحات: {pages_start} إلى {pages_end}")

        # تحويل PDF إلى صور
        try:
            images = convert_from_path(
                self.config.pdf_path,
                dpi=self.config.dpi,
                first_page=pages_start,
                last_page=pages_end
            )
            logger.info(f"تم تحويل {len(images)} صفحة")
        except FileNotFoundError:
            logger.error(f"الملف غير موجود: {self.config.pdf_path}")
            return self._empty_stats()
        except Exception as e:
            logger.error(f"فشل تحويل PDF: {e}")
            return self._empty_stats()

        # إحصائيات
        total_words = 0
        failed_ocr = 0
        corrected_by_spell = 0

        for page_idx, pil_img in enumerate(images):
            page_num = pages_start + page_idx
            logger.info(f"معالجة صفحة {page_num}/{pages_end}")

            # تحويل PIL إلى NumPy (BGR)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # معالجة مسبقة
            binary, enhanced_gray = preprocess_image(img_bgr, self.config)

            # استخراج المستطيلات المحيطة بالكلمات
            boxes = extract_word_bounding_boxes(binary, self.config)
            logger.info(f"تم العثور على {len(boxes)} كلمة في صفحة {page_num}")

            # إنشاء صورة المعاينة
            preview = img_bgr.copy()

            for x, y, w, h in boxes:
                # قص الكلمة من الصورة الأصلية (BGR) - إصلاح مهم
                crop_bgr = img_bgr[y:y + h, x:x + w]

                # التعرف على الكلمة
                raw_text = self.ocr.recognize_word(crop_bgr)

                if not raw_text:
                    failed_ocr += 1

                # التصحيح الإملائي
                final_text = correct_text(raw_text)
                if raw_text and raw_text != final_text:
                    corrected_by_spell += 1
                    logger.info(f"تصحيح: '{raw_text}' -> '{final_text}'")

                # حفظ في قاعدة البيانات
                _, buf = cv2.imencode(".png", crop_bgr)
                self.db.insert_word(
                    image_data=buf.tobytes(),
                    predicted_text=final_text
                )

                # رسم المستطيل على المعاينة
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                total_words += 1

            # حفظ صورة المعاينة
            preview_path = f"{self.config.output_dir}/preview_page_{page_num}.png"
            cv2.imwrite(preview_path, preview)
            logger.info(f"تم حفظ معاينة الصفحة {page_num}: {preview_path}")

        elapsed = time.time() - start_time
        stats = {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": self.config.pdf_path,
            "pages_processed": len(images),
            "total_words": total_words,
            "ocr_failures": failed_ocr,
            "spell_corrections": corrected_by_spell,
            "time_seconds": round(elapsed, 2),
            "avg_time_per_word": round(elapsed / total_words, 4) if total_words else 0,
        }

        # حفظ الإحصائيات
        with open(self.config.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        logger.info(f"اكتملت المعالجة: {total_words} كلمة في {elapsed:.2f} ثانية")
        logger.info(f"إحصائيات: {stats}")

        # إنشاء ملف feedback إذا لم يكن موجوداً
        self._init_feedback_csv()

        return stats

    def _init_feedback_csv(self) -> None:
        """إنشاء ملف CSV للتصحيحات إذا لم يكن موجوداً"""
        if not pd.io.common.file_exists(self.config.feedback_csv):
            pd.DataFrame(columns=[
                "timestamp", "image_id", "original_text",
                "corrected_text", "status"
            ]).to_csv(
                self.config.feedback_csv,
                index=False,
                encoding="utf-8"
            )
            logger.info(f"تم إنشاء ملف التصحيحات: {self.config.feedback_csv}")

    def _empty_stats(self) -> dict:
        """إرجاع إحصائيات فارغة عند الفشل"""
        return {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": self.config.pdf_path,
            "pages_processed": 0,
            "total_words": 0,
            "ocr_failures": 0,
            "spell_corrections": 0,
            "time_seconds": 0,
            "avg_time_per_word": 0,
            "error": True,
        }
