"""
HandwrittenOCR - محرك التعرف على النصوص
===========================================
يدعم TrOCR كمحرك أساسي و EasyOCR كبديل.
تم تصحيح: الدالة تستقبل الآن الصورة BGR الأصلية بدلاً من الصورة الثنائية.
"""

import cv2
import numpy as np
import torch
import logging
from typing import Optional
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

logger = logging.getLogger("HandwrittenOCR")


class OCREngine:
    """
    محرك التعرف على النصوص يدعم TrOCR و EasyOCR.

    يتم تحميل النماذج عند إنشاء الكائن ويُستخدم TrOCR
    كمحرك أساسي مع EasyOCR كبديل عند الفشل.
    """

    def __init__(
        self,
        trocr_model_name: str = "David-Magdy/TR_OCR_LARGE",
        ocr_languages: list | None = None,
        max_text_length: int = 50,
        device: str | None = None
    ):
        """
        تهيئة محرك التعرف.

        Parameters:
            trocr_model_name: اسم نموذج TrOCR من HuggingFace
            ocr_languages: لغات EasyOCR
            max_text_length: الحد الأقصى لطول النص المعترف
            device: الجهاز (cuda/cpu) - يتم الكشف تلقائياً إذا لم يُحدد
        """
        self.max_text_length = max_text_length

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"جهاز التعرف: {self.device}")

        # تحميل TrOCR
        logger.info(f"جاري تحميل TrOCR: {trocr_model_name}")
        self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            trocr_model_name
        ).to(self.device)
        logger.info("تم تحميل TrOCR بنجاح")

        # تحميل EasyOCR كبديل
        if ocr_languages is None:
            ocr_languages = ["en", "ar"]
        logger.info(f"جاري تحميل EasyOCR بلغات: {ocr_languages}")
        self.easy_reader = easyocr.Reader(ocr_languages)
        logger.info("تم تحميل EasyOCR بنجاح")

    def recognize_word(self, img_bgr: np.ndarray) -> str:
        """
        التعرف على كلمة واحدة باستخدام TrOCR أولاً ثم EasyOCR.

        Parameters:
            img_bgr: صورة الكلمة بصيغة BGR (OpenCV)

        Returns:
            النص المعترف (سلسلة فارغة عند الفشل)
        """
        if img_bgr is None or img_bgr.size == 0:
            return ""

        # --- محاولة TrOCR ---
        text = self._recognize_trocr(img_bgr)
        if len(text.strip()) > 1:
            return text.strip()

        # --- محاولة EasyOCR كبديل ---
        text = self._recognize_easyocr(img_bgr)
        if text.strip():
            return text.strip()

        return ""

    def _recognize_trocr(self, img_bgr: np.ndarray) -> str:
        """التعرف باستخدام TrOCR"""
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pixel_values = self.trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(self.device)

            generated_ids = self.trocr_model.generate(
                pixel_values,
                max_length=self.max_text_length
            )
            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return text.strip()
        except Exception as e:
            logger.debug(f"TrOCR فشل: {e}")
            return ""

    def _recognize_easyocr(self, img_bgr: np.ndarray) -> str:
        """التعرف باستخدام EasyOCR كبديل"""
        try:
            results = self.easy_reader.readtext(img_bgr)
            if results:
                # إرجاع النص بأعلى ثقة
                best = max(results, key=lambda r: r[2])
                return best[1]
            return ""
        except Exception as e:
            logger.debug(f"EasyOCR فشل: {e}")
            return ""
