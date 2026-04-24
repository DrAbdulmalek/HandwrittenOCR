"""
HandwrittenOCR - نقطة الدخول السريعة
=======================================
تشغيل التطبيق من الجذر:
    python run.py
    python run.py --pdf input.pdf --pages 1 5
    python run.py --hf-token hf_xxx --cache-dir ./models_cache
"""

import argparse
import sys
from pathlib import Path

# إضافة مجلد المشروع إلى مسار Python
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.main import main


def parse_args():
    parser = argparse.ArgumentParser(
        description="HandwrittenOCR - استخراج وتصحيح نصوص الخط اليدوي"
    )
    parser.add_argument(
        "--pdf", type=str, default=None,
        help="مسار ملف PDF (الافتراضي: input.pdf)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="مجلد الإخراج (الافتراضي: ~/Handwriting_Dataset)"
    )
    parser.add_argument(
        "--pages", type=int, nargs=2, default=None,
        metavar=("START", "END"),
        help="نطاق الصفحات (مثال: --pages 1 5)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI لتحويل PDF (الافتراضي: 300)"
    )
    parser.add_argument(
        "--hf-token", type=str, default="",
        help="توكن Hugging Face للنماذج المحمية"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="",
        help="مسار التخزين المؤقت للنماذج (cache_dir)"
    )
    parser.add_argument(
        "--colab", action="store_true",
        help="وضع Google Colab (استخدام Google Drive + cache)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.colab:
        config = Config.from_colab_drive(hf_token=args.hf_token)
    else:
        overrides = {}
        if args.pdf:
            overrides["pdf_path"] = args.pdf
        if args.output:
            overrides["output_dir"] = args.output
        if args.pages:
            overrides["pages_start"], overrides["pages_end"] = args.pages
        overrides["dpi"] = args.dpi
        if args.hf_token:
            overrides["hf_token"] = args.hf_token
        if args.cache_dir:
            overrides["model_cache_dir"] = args.cache_dir
        config = Config.from_dict(overrides)

    main(config)
