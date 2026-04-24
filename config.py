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

    # --- إعدادات النماذج والتخزين المؤقت ---
    hf_token: str = ""
    model_cache_dir: str = ""
    easyocr_persistent: bool = False

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

    @property
    def easyocr_drive_path(self) -> str:
        """مسار تخزين نماذج EasyOCR على Drive (لـ Colab)"""
        return os.path.join(self.output_dir, ".EasyOCR")

    @property
    def easyocr_local_path(self) -> str:
        """المسار المحلي لنماذج EasyOCR"""
        return str(Path.home() / ".EasyOCR")

    def ensure_dirs(self) -> None:
        """إنشاء المجلدات المطلوبة إذا لم تكن موجودة"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        if self.model_cache_dir:
            os.makedirs(self.model_cache_dir, exist_ok=True)

    def apply_hf_token(self) -> None:
        """
        تطبيق توكن Hugging Face على متغيرات البيئة.
        يُستخدم عند تحميل النماذج الخاصة أو المحمية.
        """
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

    def apply_cache_env(self) -> None:
        """
        تطبيق مسارات التخزين المؤقت على متغيرات البيئة
        لكي تستخدمها مكتبات transformers و torch.
        """
        if self.model_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.model_cache_dir
            os.environ["TORCH_HOME"] = self.model_cache_dir
            os.environ["HF_HOME"] = self.model_cache_dir

    def setup_easyocr_symlink(self) -> None:
        """
        نقل نماذج EasyOCR إلى Drive وإنشاء رابط رمزي.
        يعمل فقط في بيئة Colab مع easyocr_persistent=True.
        """
        if not self.easyocr_persistent:
            return

        drive_path = self.easyocr_drive_path
        local_path = self.easyocr_local_path

        # نقل النماذج إلى Drive إذا لم تكن موجودة هناك
        if os.path.exists(local_path) and not os.path.exists(drive_path):
            import shutil
            print("جاري نقل نماذج EasyOCR إلى Drive للمرة الأولى...")
            shutil.move(local_path, drive_path)

        # إنشاء الرابط الرمزي
        if not os.path.islink(local_path):
            if os.path.exists(local_path):
                import shutil
                shutil.rmtree(local_path)
            os.symlink(drive_path, local_path)
            print("تم ربط نماذج EasyOCR بـ Google Drive بنجاح")

    @classmethod
    def from_colab_drive(
        cls,
        pdf_name: str = "python notes.pdf",
        output_folder: str = "Handwriting_Dataset",
        hf_token: str = ""
    ) -> "Config":
        """
        إنشاء إعدادات مخصصة لـ Google Colab مع Drive.

        Parameters:
            pdf_name: اسم ملف PDF على Drive
            output_folder: اسم مجلد الإخراج على Drive
            hf_token: توكن Hugging Face (اختياري)
        """
        base = "/content/drive/MyDrive"
        output_dir = os.path.join(base, output_folder)
        model_cache = os.path.join(output_dir, "models_cache")
        return cls(
            pdf_path=os.path.join(base, pdf_name),
            output_dir=output_dir,
            model_cache_dir=model_cache,
            hf_token=hf_token,
            easyocr_persistent=True,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """إنشاء إعدادات من قاموس"""
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
