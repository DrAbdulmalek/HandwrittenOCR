"""
HandwrittenOCR - إعدادات المشروع
===================================
ملف الإعدادات المركزي - يسمح بتخصيص جميع المسارات والمعاملات
بدلاً من المسارات المرمَّجة يدوياً في الكود.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """إعدادات المشروع الرئيسية"""

    # --- مسارات الملفات ---
    pdf_path: str = "input.pdf"
    output_dir: str = str(Path.home() / "Handwriting_Dataset")

    # --- إعدادات معالجة الصور ---
    dpi: int = 300
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    denoise_strength: int = 30
    min_word_width: int = 25
    min_word_height: int = 15
    dilation_kernel: tuple = (25, 5)

    # --- إعدادات التعرف ---
    max_text_length: int = 50
    trocr_model_name: str = "David-Magdy/TR_OCR_LARGE"
    ocr_languages: list = field(default_factory=lambda: ["en", "ar"])

    # --- نطاق الصفحات ---
    pages_start: int = 1
    pages_end: int = 2

    # --- إعدادات قاعدة البيانات ---
    db_name: str = "handwriting_data.db"

    # --- إعدادات التسجيل ---
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"

    @property
    def db_path(self) -> str:
        return os.path.join(self.output_dir, self.db_name)

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.output_dir, "Logs")

    @property
    def log_file(self) -> str:
        from datetime import datetime
        return os.path.join(
            self.logs_dir,
            f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

    @property
    def feedback_csv(self) -> str:
        return os.path.join(self.logs_dir, "user_corrections_feedback.csv")

    @property
    def stats_json(self) -> str:
        return os.path.join(self.logs_dir, "processing_stats.json")

    def ensure_dirs(self) -> None:
        """إنشاء المجلدات المطلوبة إذا لم تكن موجودة"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    @classmethod
    def from_colab_drive(cls, pdf_name: str = "python notes.pdf",
                         output_folder: str = "Handwriting_Dataset") -> "Config":
        """إنشاء إعدادات مخصصة لـ Google Colab مع Drive"""
        base = "/content/drive/MyDrive"
        return cls(
            pdf_path=os.path.join(base, pdf_name),
            output_dir=os.path.join(base, output_folder),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """إنشاء إعدادات من قاموس"""
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
