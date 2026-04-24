# HandwrittenOCR

> مشروع استخراج وتصحيح نصوص الخط اليدوي من ملفات PDF

## المميزات

- **تعرف متعدد المحركات**: TrOCR (أساسي) + EasyOCR (بديل) + Tesseract
- **دعم ثنائي اللغة**: العربية والإنجليزية
- **تصحيح إملائي ذكي**: ar-corrector للعربية + pyspellchecker للإنجليزية
- **معالجة مسبقة متقدمة**: تسوية الميل، CLAHE، إزالة الضوضاء، Thresholding
- **واجهة مراجعة تفاعلية**: Jupyter (ipywidgets) أو CLI
- **تخزين منظم**: SQLite + CSV + JSON للإحصائيات
- **وضعان**: محلي أو Google Colab

## التثبيت

### المتطلبات النظامية

```bash
# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract
```

### تثبيت المكتبات

```bash
git clone https://github.com/DrAbdulmalek/HandwrittenOCR.git
cd HandwrittenOCR
pip install -r requirements.txt
```

## الاستخدام

### التشغيل الأساسي

```bash
# تشغيل مع ملف PDF محدد
python run.py --pdf path/to/document.pdf

# تحديد نطاق الصفحات
python run.py --pdf document.pdf --pages 1 10

# تحديد مجلد الإخراج
python run.py --pdf document.pdf --output ./results
```

### في Google Colab

```python
from config import Config
from src.main import main

config = Config.from_colab_drive(pdf_name="document.pdf")
main(config)
```

أو استخدم الدفتر الجاهز: `notebooks/handwritten_ocr_colab.ipynb`

### كوحدة Python

```python
from config import Config
from src.main import main

config = Config(
    pdf_path="input.pdf",
    output_dir="./results",
    pages_start=1,
    pages_end=5,
    dpi=300
)
main(config)
```

### واجهة المراجعة فقط (بدون معالجة)

```python
from config import Config
from src.database import HandwritingDB
from src.review_ui import ReviewUI

config = Config(output_dir="./results")
db = HandwritingDB(config.db_path)
ui = ReviewUI(db, config.feedback_csv)
ui.launch()
```

## هيكل المشروع

```
HandwrittenOCR/
├── README.md              # هذا الملف
├── requirements.txt        # المكتبات المطلوبة
├── config.py              # إعدادات المشروع المركزية
├── run.py                 # نقطة الدخول (CLI)
├── src/
│   ├── __init__.py
│   ├── main.py            # التشغيل الرئيسي
│   ├── preprocessing.py   # معالجة الصور المسبقة
│   ├── recognition.py     # محرك التعرف (TrOCR + EasyOCR)
│   ├── correction.py      # التصحيح الإملائي
│   ├── database.py        # إدارة قاعدة البيانات (SQLite)
│   ├── pdf_processor.py   # معالج PDF
│   ├── review_ui.py       # واجهة المراجعة (Jupyter + CLI)
│   └── logger.py          # نظام التسجيل
├── notebooks/
│   └── handwritten_ocr_colab.ipynb  # دفتر Google Colab
├── tests/
│   └── __init__.py
└── output/                # مجلد الإخراج (gitignored)
```

## التصحيحات المطبقة على الكود الأصلي

| المشكلة | التصحيح |
|---------|---------|
| اعتماد كامل على Google Colab | هيكل مشروع Python مستقل مع وضع Colab اختياري |
| مسارات مُرمَّجة يدوياً | نظام Config قابل للتخصيص |
| `SpellChecker.correction()` تُستخدم على جمل كاملة | تصحيح كلمة بكلمة مع حفظ علامات الترقيم |
| `preprocess_image` ترجع binary لكن `recognize_word` تتوقع BGR | فصل الصورة الثنائية عن الصورة المحسنة |
| `df` متغير محلي يخرج عن التزامن مع DB عند الحذف | استخدام `HandwritingDB` مباشرة |
| عدم وجود معالجة أخطاء | try/except في كل نقطة فشل محتملة |
| كل شيء في خلايا Jupyter | وحدات Python منفصلة ومختبرة |
| `cv2_imshow` من Colab فقط | دعم matplotlib + CLI + Jupyter |

## الترخيص

MIT License
