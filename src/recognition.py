"""
HandwrittenOCR - محرك التعرف على النصوص
===========================================
يدعم TrOCR كمحرك أساسي و EasyOCR كبديل.

المميزات:
- Ensemble التعرف: TrOCR + EasyOCR مع اختيار الأفضل
- دعم cache_dir لتحميل النماذج من مسار مخصص
- دعم HF_TOKEN للنماذج المحمية
- EasyOCR يختار النص بأعلى ثقة
"""

import cv2
import numpy as np
import torch
import logging
from typing import Tuple
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

logger = logging.getLogger("HandwrittenOCR")


class OCREngine:
    """
    محرك التعرف على النصوص يدعم TrOCR و EasyOCR.

    يدعم وضعين:
    1. recognize_word: TrOCR أولاً ثم EasyOCR كبديل
    2. recognize_word_ensemble: كلاهما مع اختيار الأفضل حسب الثقة
    """

    def __init__(
        self,
        trocr_model_name: str = "David-Magdy/TR_OCR_LARGE",
        ocr_languages: list | None = None,
        max_text_length: int = 50,
        device: str | None = None,
        cache_dir: str = "",
        hf_token: str = "",
        trocr_default_confidence: float = 0.7,
    ):
        self.max_text_length = max_text_length
        self.trocr_default_confidence = trocr_default_confidence

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        logger.info(f"جهاز التعرف: {self.device}")

        # خيارات تحميل HuggingFace
        hf_kwargs = {}
        if cache_dir:
            hf_kwargs["cache_dir"] = cache_dir
        if hf_token:
            hf_kwargs["token"] = hf_token

        # تحميل TrOCR
        logger.info(f"جاري تحميل TrOCR: {trocr_model_name}")
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                trocr_model_name, **hf_kwargs
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                trocr_model_name, **hf_kwargs
            ).to(self.device)
            logger.info("تم تحميل TrOCR بنجاح")
        except Exception as e:
            logger.error(f"فشل تحميل TrOCR: {e}")
            raise

        # تحميل EasyOCR
        if ocr_languages is None:
            ocr_languages = ["en", "ar"]
        logger.info(f"جاري تحميل EasyOCR بلغات: {ocr_languages}")
        self.easy_reader = easyocr.Reader(ocr_languages)
        logger.info("تم تحميل EasyOCR بنجاح")

    def recognize_word(self, img_bgr: np.ndarray) -> str:
        """
        التعرف على كلمة: TrOCR أولاً ثم EasyOCR كبديل.

        Returns:
            النص المعترف (سلسلة فارغة عند الفشل)
        """
        if img_bgr is None or img_bgr.size == 0:
            return ""
        text = self._recognize_trocr(img_bgr)
        if len(text.strip()) > 1:
            return text.strip()
        text = self._recognize_easyocr(img_bgr)
        return text.strip() if text.strip() else ""

    def recognize_word_ensemble(
        self,
        img_bgr: np.ndarray,
        easyocr_raw: list | None = None
    ) -> Tuple[str, float, str, bool]:
        """
        التعرف بالـ Ensemble: يجمع نتائج TrOCR و EasyOCR
        ويختار الأفضل حسب الثقة.

        Parameters:
            img_bgr: صورة الكلمة BGR
            easyocr_raw: نتيجة EasyOCR الخام [bbox, text, conf]
                         (اختياري - إذا توفرت تُستخدم مباشرة)

        Returns:
            tuple: (text, confidence, model_source, is_low_confidence)
        """
        results = []

        # TrOCR
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pix = self.trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(self.device)
            ids = self.trocr_model.generate(pix, max_length=self.max_text_length)
            txt = self.trocr_processor.batch_decode(
                ids, skip_special_tokens=True
            )[0].strip()
            if len(txt) > 1:
                results.append(("trocr", txt, self.trocr_default_confidence))
        except Exception as e:
            logger.debug(f"TrOCR فشل: {e}")

        # EasyOCR
        if easyocr_raw:
            results.append(("easyocr", easyocr_raw[1], easyocr_raw[2]))
        else:
            try:
                res = self.easy_reader.readtext(img_bgr)
                if res:
                    best = max(res, key=lambda r: r[2])
                    results.append(("easyocr", best[1], best[2]))
            except Exception as e:
                logger.debug(f"EasyOCR فشل: {e}")

        if not results:
            return "", 0.0, "none", True

        # اختيار النص بأعلى ثقة
        best = max(results, key=lambda x: x[2])
        text, confidence, source = best
        is_low = confidence < 0.5

        return text, confidence, source, is_low

    def detect_words_full(self, img_bgr: np.ndarray) -> list:
        """
        كشف الكلمات مع الإحداثيات والنص والثقة باستخدام EasyOCR.

        Returns:
            قائمة بـ [bbox, text, conf] لكل كلمة
        """
        try:
            return self.easy_reader.readtext(img_bgr, detail=1)
        except Exception as e:
            logger.debug(f"EasyOCR كشف فشل: {e}")
            return []

    def _recognize_trocr(self, img_bgr: np.ndarray) -> str:
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pixel_values = self.trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values.to(self.device)
            generated_ids = self.trocr_model.generate(
                pixel_values, max_length=self.max_text_length
            )
            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text.strip()
        except Exception as e:
            logger.debug(f"TrOCR فشل: {e}")
            return ""

    def _recognize_easyocr(self, img_bgr: np.ndarray) -> str:
        try:
            results = self.easy_reader.readtext(img_bgr)
            if results:
                best = max(results, key=lambda r: r[2])
                return best[1]
            return ""
        except Exception as e:
            logger.debug(f"EasyOCR فشل: {e}")
            return ""
