# HandwrittenOCR
# استخراج وتصحيح نصوص الخط اليدوي

"""
HandwrittenOCR - مشروع استخراج وتصحيح النصوص من الخط اليدوي
================================================================

يدعم العربية والإنجليزية باستخدام:
- TrOCR (محرك أساسي)
- EasyOCR (بديل)
- Tesseract (إضافي)
- ar-corrector (تصحيح عربي)
- pyspellchecker (تصحيح إنجليزي)
"""

from src.preprocessing import preprocess_image, extract_word_bounding_boxes
from src.recognition import OCREngine
from src.correction import correct_text, init_correctors
from src.database import HandwritingDB
from src.pdf_processor import PDFProcessor
from src.review_ui import ReviewUI
from config import Config

__version__ = "1.0.0"
