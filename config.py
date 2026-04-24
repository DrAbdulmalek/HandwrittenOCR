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
    enable_deskewing: bool = True

    # --- إعدادات التعرف ---
    max_text_length: int = 50
    trocr_model_name: str = "David-Magdy/TR_OCR_LARGE"
    ocr_languages: list = field(default_factory=lambda: ["en", "ar"])
    trocr_default_confidence: float = 0.7
    low_confidence_threshold: float = 0.5

    # --- إعدادات قاموس التصحيح ---
    correction_dict_min_votes: int = 2

    # --- إعدادات Fine-tuning ---
    finetune_min_samples: int = 100
    finetune_epochs: int = 3
    finetune_batch_size: int = 4
    finetune_lr: float = 5e-5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["query", "value"])

    # --- إعدادات التصدير ---
    export_val_ratio: float = 0.1
    hf_dataset_repo: str = ""

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
    def correction_dict_path(self) -> str:
        return os.path.join(self.output_dir, "correction_dict.json")

    @property
    def easyocr_drive_path(self) -> str:
        return os.path.join(self.output_dir, ".EasyOCR")

    @property
    def easyocr_local_path(self) -> str:
        return str(Path.home() / ".EasyOCR")

    @property
    def export_dir(self) -> str:
        return os.path.join(self.output_dir, "hf_training_dataset")

    @property
    def lora_save_path(self) -> str:
        return os.path.join(
            self.model_cache_dir or self.output_dir,
            "trocr_lora_finetuned"
        )

    def ensure_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        if self.model_cache_dir:
            os.makedirs(self.model_cache_dir, exist_ok=True)

    def apply_hf_token(self) -> None:
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

    def apply_cache_env(self) -> None:
        if self.model_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.model_cache_dir
            os.environ["TORCH_HOME"] = self.model_cache_dir
            os.environ["HF_HOME"] = self.model_cache_dir

    def setup_easyocr_symlink(self) -> None:
        if not self.easyocr_persistent:
            return
        import shutil
        drive_path = self.easyocr_drive_path
        local_path = self.easyocr_local_path
        if os.path.exists(local_path) and not os.path.exists(drive_path):
            print("جاري نقل نماذج EasyOCR إلى Drive للمرة الأولى...")
            shutil.move(local_path, drive_path)
        if not os.path.islink(local_path):
            if os.path.exists(local_path):
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
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
