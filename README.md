# HandwrittenOCR

> مشروع استخراج وتصحيح نصوص الخط اليدوي من ملفات PDF - نظام التحسين المستمر

## المميزات

- **تعرف متعدد المحركات**: TrOCR (أساسي) + EasyOCR (بديل) + Ensemble
- **دعم ثنائي اللغة**: العربية والإنجليزية
- **تصحيح إملائي ذكي**: ar-corrector + pyspellchecker
- **قاموس تصحيح مستمر**: يتعلم من مراجعات المستخدم تلقائياً
- **تجزئة ذكية**: EasyOCR أولاً، الكنتورات كبديل
- **معالجة مسبقة**: تسوية الميل، CLAHE، إزالة الضوضاء، Thresholding
- **واجهة مراجعة**: Jupyter (ipywidgets) أو CLI - تعرض غير المراجعة أولاً
- **تصدير بيانات التدريب**: JSONL مع تقسيم train/val
- **رفع إلى HuggingFace**: مباشرة من الكود
- **تدريب LoRA**: Fine-tune TrOCR على تصحيحات المستخدم
- **إعادة تجميع الجمل**: مع دعم RTL للعربية
- **تخزين مؤقت**: cache_dir + EasyOCR symlink على Drive

## التثبيت

```bash
sudo apt-get install -y poppler-utils tesseract-ocr
pip install -r requirements.txt
```

## الاستخدام

### التشغيل الأساسي

```bash
python run.py --pdf document.pdf --pages 1 10
```

### في Google Colab

```python
from google.colab import userdata
from config import Config
from src.main import main

config = Config.from_colab_drive(
    pdf_name="document.pdf",
    hf_token=userdata.get("HF_TOKEN")
)
main(config)
```

### نظام التحسين المستمر

```python
from config import Config
from src.database import HandwritingDB
from src.export import export_finetuning_dataset, push_to_huggingface
from src.finetuning import finetune_trocr_lora
from src.reconstruction import reconstruct_sentences
from src.recognition import OCREngine

# 1. معالجة ومراجعة
# ... (run main first, review words)

# 2. تصدير بيانات التدريب
config = Config.from_colab_drive(hf_token="hf_xxx")
db = HandwritingDB(config.db_path)
export_finetuning_dataset(db, config.export_dir)

# 3. رفع إلى HuggingFace
push_to_huggingface(config.export_dir, "user/handwriting-dataset", config.hf_token)

# 4. تدريب LoRA
ocr_engine = OCREngine(...)
finetune_trocr_lora(
    ocr_engine.trocr_model, ocr_engine.trocr_processor,
    db, ocr_engine.device, config.lora_save_path
)

# 5. إعادة تجميع الجمل
df_sentences = reconstruct_sentences(db)
```

## هيكل المشروع

```
HandwrittenOCR/
├── config.py              # إعدادات مركزية
├── run.py                 # نقطة الدخول CLI
├── requirements.txt
├── src/
│   ├── main.py            # التشغيل الرئيسي
│   ├── preprocessing.py   # معالجة الصور + تجزئة ذكية
│   ├── recognition.py     # Ensemble التعرف
│   ├── correction.py      # تصحيح + قاموس مستمر
│   ├── database.py        # SQLite v2 (مع إحداثيات)
│   ├── pdf_processor.py   # معالج PDF
│   ├── review_ui.py       # واجهة المراجعة
│   ├── export.py          # تصدير + رفع HF
│   ├── finetuning.py      # LoRA training
│   ├── reconstruction.py  # تجميع الجمل
│   └── logger.py          # تسجيل
├── notebooks/
│   └── handwritten_ocr_colab.ipynb
└── tests/
```

## التصحيحات المطبقة (التي يجب حفظها للمرات القادمة)

1. `!mv`/`!rm`/`!ln` shell commands -> `shutil.move`/`shutil.rmtree`/`os.symlink`
2. `SpellChecker.correction()` على جمل كاملة -> كلمة بكلمة مع حفظ الترقيم
3. `preprocess_image` ترجع binary فقط -> `(binary, enhanced)`
4. الكلمات تُقطع من الصورة الثنائية -> من `img_bgr` الأصلية
5. EasyOCR يأخذ أول نتيجة -> `max(results, key=lambda r: r[2])`
6. `cv2_imshow` من Colab -> `cv2.imwrite` + مسارات عامة
7. المسارات المرمَّجة يدوياً -> `Config` dataclass
8. `df` محلي يخرج عن التزامن مع DB -> `HandwritingDB` مباشرة
9. Status: `'yes'/'no'` -> `'verified'/'unverified'`
10. DB schema v1 -> v2 مع ترقية تلقائية (migration)

## الترخيص

MIT License
