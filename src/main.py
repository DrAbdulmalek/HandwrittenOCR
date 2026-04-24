"""
HandwrittenOCR - الوحدة الرئيسية
===================================
نقطة الدخول الرئيسية للتطبيق - تجمع بين جميع المكونات.
"""

import time
import logging
from config import Config
from src.logger import setup_logging
from src.recognition import OCREngine
from src.correction import init_correctors
from src.database import HandwritingDB
from src.pdf_processor import PDFProcessor
from src.review_ui import ReviewUI


def main(config: Config | None = None):
    if config is None:
        config = Config()

    config.apply_hf_token()
    config.apply_cache_env()
    config.ensure_dirs()
    config.setup_easyocr_symlink()

    logger = setup_logging(config)
    logger.info("بدء تشغيل HandwrittenOCR")
    logger.info(f"ملف PDF: {config.pdf_path}")
    logger.info(f"مجلد الإخراج: {config.output_dir}")
    if config.model_cache_dir:
        logger.info(f"تخزين مؤقت: {config.model_cache_dir}")

    # تحميل المدققات الإملائية
    init_correctors()

    # تحميل محرك التعرف
    start = time.time()
    ocr_engine = OCREngine(
        trocr_model_name=config.trocr_model_name,
        ocr_languages=config.ocr_languages,
        max_text_length=config.max_text_length,
        cache_dir=config.model_cache_dir,
        hf_token=config.hf_token,
        trocr_default_confidence=config.trocr_default_confidence,
    )
    logger.info(f"تم تحميل النماذج في {time.time() - start:.2f} ثانية")

    # تهيئة قاعدة البيانات
    db = HandwritingDB(config.db_path)

    # معالجة PDF
    processor = PDFProcessor(config, ocr_engine, db)
    stats = processor.process()

    if stats.get("error"):
        logger.error("فشلت المعالجة!")
        return

    # عرض الإحصائيات
    print("\n" + "=" * 50)
    print("  إحصائيات المعالجة")
    print("=" * 50)
    print(f"  الصفحات:       {stats['pages_processed']}")
    print(f"  الكلمات:       {stats['total_words']}")
    print(f"  إخفاقات OCR:   {stats['ocr_failures']}")
    print(f"  تصحيحات:       {stats['spell_corrections']}")
    print(f"  تصحيحات قاموس:{stats['dict_corrections']}")
    print(f"  حجم القاموس:   {stats['correction_dict_size']}")
    print(f"  الوقت:         {stats['time_seconds']:.2f} ثانية")
    print(f"  متوسط/كلمة:    {stats['avg_time_per_word']:.4f} ثانية")
    print("=" * 50)

    # عرض ملفات المراقبة
    print(f"\nملفات المراقبة:")
    print(f"  سجل الأحداث:     {config.log_file}")
    print(f"  إحصائيات:        {config.stats_json}")
    print(f"  تصحيحات:         {config.feedback_csv}")
    print(f"  قاموس التصحيح:   {config.correction_dict_path}")

    # تشغيل واجهة المراجعة
    print("\nتشغيل واجهة المراجعة...")
    review_ui = ReviewUI(db, config.feedback_csv)
    review_ui.launch()


if __name__ == "__main__":
    main()
