"""
HandwrittenOCR - معالجة الصور المسبقة
========================================
تحسين الصور قبل التعرف: تسوية الميل، CLAHE، إزالة الضوضاء، Thresholding.
يدعم تجزئة ذكية: EasyOCR أولاً ثم الكنتورات كبديل.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import Config


def preprocess_image(
    img_bgr: np.ndarray,
    config: Config | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    تحسين الصورة للتعرف على النصوص.

    Parameters:
        img_bgr: الصورة بصيغة BGR (OpenCV)
        config: إعدادات المشروع (اختياري)

    Returns:
        tuple: (binary_image, enhanced_gray)
    """
    if config is None:
        config = Config()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- تسوية الميل (Deskewing) - اختياري ---
    if config.enable_deskewing:
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle += 90
            if abs(angle) > 0.5:
                h, w = gray.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(
                    gray, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

    # --- تحسين التباين (CLAHE) ---
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_size
    )
    enhanced = clahe.apply(gray)

    # --- إزالة الضوضاء ---
    denoised = cv2.fastNlMeansDenoising(enhanced, h=config.denoise_strength)

    # --- تحويل إلى صورة ثنائية ---
    _, binary = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary, enhanced


def smart_word_segmentation(
    img_bgr: np.ndarray,
    binary_img: np.ndarray,
    easyocr_detections: Optional[list] = None,
    config: Config | None = None
) -> List[Tuple[int, int, int, int]]:
    """
    تجزئة ذكية للكلمات: EasyOCR أولاً، الكنتورات كبديل.

    Parameters:
        img_bgr: الصورة الأصلية BGR
        binary_img: الصورة الثنائية
        easyocr_detections: نتائج EasyOCR (list of [bbox, text, conf])
            bbox = list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        config: إعدادات المشروع

    Returns:
        قائمة بـ (x, y, w, h) لكل كلمة
    """
    if config is None:
        config = Config()

    # --- استخدام كشف EasyOCR إذا متوفر ---
    if easyocr_detections:
        boxes = []
        for detection in easyocr_detections:
            bbox = detection[0]  # 4 نقاط
            pts = np.array(bbox, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            if w > 15 and h > 10:
                boxes.append((x, y, w, h))
        if boxes:
            return boxes

    # --- بديل: التجزئة بالكنتورات ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.dilation_kernel)
    dilated = cv2.dilate(binary_img, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= config.min_word_width and h >= config.min_word_height:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    return boxes
