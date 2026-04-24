"""
HandwrittenOCR - معالجة الصور المسبقة
========================================
تحسين الصور قبل التعرف: تسوية الميل، CLAHE، إزالة الضوضاء، Thresholding.
تم تصحيح خط أنابيب الصور: الدالة الآن ترجع كلاً من الصورة الثنائية
والصورة الأصلية المحسنة لاستخدامها في مراحل مختلفة.
"""

import cv2
import numpy as np
from typing import Tuple
from config import Config


def preprocess_image(
    img_bgr: np.ndarray,
    config: Config | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    تحسين الصورة للتعرف على النصوص.

    تسوية الميل (deskewing)، تحسين التباين (CLAHE)،
    إزالة الضوضاء، وتحويل إلى صورة ثنائية.

    Parameters:
        img_bgr: الصورة بصيغة BGR (OpenCV)
        config: إعدادات المشروع (اختياري)

    Returns:
        tuple: (binary_image, enhanced_gray)
            - binary_image: صورة ثنائية لتجزئة الكلمات
            - enhanced_gray: صورة رمادية محسنة (للتعرف بالـ TrOCR)
    """
    if config is None:
        config = Config()

    # تحويل إلى رمادي
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- تسوية الميل (Deskewing) ---
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 100:  # الحد الأدنى من النقاط لتجنب الضوضاء
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        if abs(angle) > 0.5:  # فقط إذا كان الميل ملحوظاً
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
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        h=config.denoise_strength
    )

    # --- تحويل إلى صورة ثنائية (للتجزئة) ---
    _, binary = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary, enhanced


def extract_word_bounding_boxes(
    binary: np.ndarray,
    config: Config | None = None
) -> list:
    """
    استخراج المستطيلات المحيطة بالكلمات من الصورة الثنائية.

    Parameters:
        binary: الصورة الثنائية
        config: إعدادات المشروع

    Returns:
        قائمة بالنتائج (x, y, w, h) لكل كلمة
    """
    if config is None:
        config = Config()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        config.dilation_kernel
    )
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= config.min_word_width and h >= config.min_word_height:
            boxes.append((x, y, w, h))

    # ترتيب من اليسار لليمين ومن الأعلى للأسفل
    boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    return boxes
