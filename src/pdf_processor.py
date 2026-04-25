"""
HandwrittenOCR - معالجة PDF
=============================
استخراج الكلمات من ملفات PDF مع:
- تجزئة ذكية (EasyOCR أولاً، الكنتورات كبديل)
- التعرف بالـ Ensemble (TrOCR + EasyOCR)
- قاموس تصحيح مستمر
- إحداثيات الموقع ورقم الصفحة
- نظام Checkpoint لاستئناف المعالجة
"""

import cv2
import json
import time
import logging
import os
from datetime import datetime
from pdf2image import convert_from_path
import numpy as np
import pandas as pd

from config import Config
from src.preprocessing import preprocess_image, smart_word_segmentation
from src.recognition import OCREngine
from src.correction import (
    correct_text, build_correction_dict, apply_correction_dict
)
from src.database import HandwritingDB

logger = logging.getLogger("HandwrittenOCR")


class PDFProcessor:
    """معالج ملفات PDF مع Ensemble التعرف والتصحيح المستمر."""

    def __init__(self, config: Config, ocr_engine: OCREngine, db: HandwritingDB):
        self.config = config
        self.ocr = ocr_engine
        self.db = db
        self.checkpoint_file = os.path.join(config.output_dir, 'ocr_checkpoint.json')

    def process(self, resume: bool = True) -> dict:
        """معالجة ملف PDF كاملاً مع Ensemble وقاموس التصحيح ونظام Checkpoint."""
        start_time = time.time()
        pages_start = self.config.pages_start
        pages_end = self.config.pages_end

        # استئناف من checkpoint
        checkpoint = self._load_checkpoint() if resume else None
        if checkpoint:
            logger.info(f"استئناف من الصفحة {checkpoint['last_page_processed']}")
            pages_start = checkpoint['last_page_processed']

        logger.info(f"بدء معالجة: {self.config.pdf_path}")
        logger.info(f"نطاق الصفحات: {pages_start} إلى {pages_end}")

        # بناء قاموس التصحيح من تصحيحات سابقة
        correction_dict = build_correction_dict(
            self.config.feedback_csv,
            self.config.correction_dict_path,
            self.config.correction_dict_min_votes
        )
        if correction_dict:
            logger.info(f"قاموس التصحيح: {len(correction_dict)} كلمة")

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

        total_words = checkpoint.get('processed_words', 0) if checkpoint else 0
        failed_ocr = 0
        corrected_by_spell = 0
        corrected_by_dict = 0

        for page_idx, pil_img in enumerate(images):
            page_num = pages_start + page_idx
            logger.info(f"معالجة صفحة {page_num}/{pages_end}")

            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # كشف الكلمات باستخدام EasyOCR أولاً
            try:
                easyocr_detections = self.ocr.detect_words_full(img_bgr)
            except Exception:
                easyocr_detections = []

            # معالجة مسبقة
            binary, enhanced = preprocess_image(img_bgr, self.config)

            # تجزئة ذكية: EasyOCR أولاً، الكنتورات كبديل
            boxes = smart_word_segmentation(
                img_bgr, binary,
                easyocr_detections=easyocr_detections,
                config=self.config
            )
            logger.info(f"تم العثور على {len(boxes)} كلمة في صفحة {page_num}")

            # ربط الكنتورات مع كشف EasyOCR
            boxes_info = self._match_boxes_with_detections(
                boxes, easyocr_detections
            )

            for (x, y, w, h), easyocr_raw in boxes_info:
                crop = img_bgr[y:y + h, x:x + w]

                # التعرف بالـ Ensemble
                raw_text, conf, source, is_low = self.ocr.recognize_word_ensemble(
                    crop, easyocr_raw=easyocr_raw
                )

                if not raw_text:
                    failed_ocr += 1

                # التصحيح الإملائي
                final_text = correct_text(raw_text)

                # تطبيق قاموس التصحيح
                before_dict = final_text
                final_text = apply_correction_dict(final_text, correction_dict)
                if before_dict != final_text:
                    corrected_by_dict += 1

                if raw_text and raw_text != final_text:
                    corrected_by_spell += 1
                    logger.debug(f"تصحيح: '{raw_text}' -> '{final_text}'")

                # حفظ في قاعدة البيانات (مخطط v2)
                _, buf = cv2.imencode(".png", crop)
                self.db.insert_word(
                    image_data=buf.tobytes(),
                    predicted_text=final_text,
                    status="unverified",
                    confidence=conf,
                    model_source=source,
                    x=x, y=y, w=w, h=h,
                    page_num=page_num,
                )

                total_words += 1

            # حفظ checkpoint بعد كل صفحة
            self._save_checkpoint(
                page_num + 1, pages_end, total_words
            )

        # مسح checkpoint عند الاكتمال
        self._clear_checkpoint()

        elapsed = time.time() - start_time
        stats = {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": self.config.pdf_path,
            "pages_processed": len(images),
            "total_words": total_words,
            "ocr_failures": failed_ocr,
            "spell_corrections": corrected_by_spell,
            "dict_corrections": corrected_by_dict,
            "correction_dict_size": len(correction_dict),
            "time_seconds": round(elapsed, 2),
            "avg_time_per_word": round(elapsed / total_words, 4) if total_words else 0,
        }

        with open(self.config.stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        logger.info(f"اكتملت المعالجة: {total_words} كلمة في {elapsed:.2f} ثانية")

        # إنشاء ملف feedback إذا لم يكن موجوداً
        self._init_feedback_csv()

        return stats

    def _save_checkpoint(self, page_num: int, total_pages: int, processed_words: int) -> None:
        """حفظ checkpoint لاستئناف المعالجة"""
        checkpoint = {
            'last_page_processed': page_num,
            'total_pages': total_pages,
            'processed_words': processed_words,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            logger.info(f"تم حفظ checkpoint: الصفحة {page_num}/{total_pages}")
        except Exception as e:
            logger.warning(f"فشل حفظ checkpoint: {e}")

    def _load_checkpoint(self) -> dict | None:
        """تحميل checkpoint"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _clear_checkpoint(self) -> None:
        """مسح checkpoint عند الاكتمال"""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                logger.info("تم مسح checkpoint")
            except Exception:
                pass

    def _match_boxes_with_detections(
        self,
        boxes: list,
        easyocr_detections: list
    ) -> list:
        """ربط المستطيلات المحيطة مع كشف EasyOCR."""
        if not easyocr_detections:
            return [(box, None) for box in boxes]

        det_boxes = []
        for det in easyocr_detections:
            pts = np.array(det[0], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            det_boxes.append(((x, y, w, h), det))

        boxes_info = []
        used_dets = set()

        for box in boxes:
            bx, by, bw, bh = box
            best_det = None
            best_overlap = 0

            for idx, (det_box, det_raw) in enumerate(det_boxes):
                if idx in used_dets:
                    continue
                dx, dy, dw, dh = det_box
                overlap = self._iou(box, det_box)
                if overlap > best_overlap and overlap > 0.3:
                    best_overlap = overlap
                    best_det = det_raw
                    used_dets.add(idx)

            boxes_info.append((box, best_det))

        return boxes_info

    @staticmethod
    def _iou(box1, box2) -> float:
        """حساب نسبة التداخل بين مستطيلين"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter

        return inter / union if union > 0 else 0

    def _init_feedback_csv(self) -> None:
        if not pd.io.common.file_exists(self.config.feedback_csv):
            pd.DataFrame(columns=[
                "timestamp", "image_id", "original_text",
                "corrected_text", "status"
            ]).to_csv(
                self.config.feedback_csv,
                index=False, encoding="utf-8"
            )

    def _empty_stats(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": self.config.pdf_path,
            "pages_processed": 0,
            "total_words": 0,
            "ocr_failures": 0,
            "spell_corrections": 0,
            "dict_corrections": 0,
            "correction_dict_size": 0,
            "time_seconds": 0,
            "avg_time_per_word": 0,
            "error": True,
        }
