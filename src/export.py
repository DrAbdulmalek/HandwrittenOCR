"""
HandwrittenOCR - تصدير بيانات التدريب ورفع إلى HuggingFace
==============================================================
تصدير الكلمات الموثقة (verified) كبيانات تدريب JSONL
مع تقسيم train/val ورفع إلى HuggingFace Hub.
"""

import os
import json
import random
import logging

logger = logging.getLogger("HandwrittenOCR")


def export_finetuning_dataset(
    db,
    output_dir: str,
    val_ratio: float = 0.1
) -> str | None:
    """
    تصدير البيانات الموثقة كبيانات تدريب JSONL.

    Parameters:
        db: كائن قاعدة البيانات
        output_dir: مجلد الإخراج
        val_ratio: نسبة بيانات التحقق (الافتراضي 10%)

    Returns:
        مسار مجلد الإخراج أو None عند الفشل
    """
    verified = db.get_verified()

    # تضمين الكلمات المصححة على مستوى الكلمات والجمل
    verified = [w for w in verified if w.get('status') in ('verified', 'sentence_corrected')]

    if not verified:
        logger.warning("لا توجد بيانات موثقة (verified أو sentence_corrected) للتصدير")
        return None

    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # كتابة الصور وإنشاء سجلات JSONL
    jsonl_records = []
    for row in verified:
        filename = f"img_{row['image_id']}.png"
        filepath = os.path.join(img_dir, filename)
        with open(filepath, "wb") as f:
            f.write(row["image_data"])

        text = (row["predicted_text"] or "").strip()
        if text:
            jsonl_records.append({
                "image": filename,
                "text": text,
                "verified": True,
            })

    if not jsonl_records:
        logger.warning("لا توجد سجلات صالحة بعد التصفية")
        return None

    # تقسيم عشوائي
    random.shuffle(jsonl_records)
    split_idx = int(len(jsonl_records) * (1 - val_ratio))
    train_data = jsonl_records[:split_idx]
    val_data = jsonl_records[split_idx:]

    def save_jsonl(data, fname):
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    train_path = save_jsonl(train_data, "train.jsonl")
    val_path = save_jsonl(val_data, "val.jsonl")

    logger.info(
        f"تم التصدير: {len(jsonl_records)} عينة "
        f"(train={len(train_data)}, val={len(val_data)})"
    )
    print(f"تم التصدير بنجاح: {len(jsonl_records)} عينة موثقة")
    print(f"القسم: train={len(train_data)}, val={len(val_data)}")
    print(f"المسار: {os.path.abspath(output_dir)}")

    return output_dir


def push_to_huggingface(
    local_dataset_dir: str,
    hf_repo_id: str,
    hf_token: str = ""
) -> bool:
    """
    رفع البيانات الموثقة إلى HuggingFace Hub.

    Parameters:
        local_dataset_dir: مجلد البيانات المحلي
        hf_repo_id: معرف المستودع (مثال: username/handwriting-dataset)
        hf_token: توكن HuggingFace

    Returns:
        True عند النجاح
    """
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        logger.error("huggingface_hub غير مثبت. ثبّته بـ: pip install huggingface_hub")
        return False

    if not os.path.exists(local_dataset_dir):
        logger.error(f"المجلد غير موجود: {local_dataset_dir}")
        return False

    # تسجيل الدخول
    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            logger.error(f"فشل تسجيل الدخول: {e}")
            return False

    api = HfApi()

    # إنشاء المستودع إذا لم يكن موجوداً
    try:
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        logger.debug(f"معلومات المستودع: {e}")

    # رفع الملفات
    from datetime import datetime
    try:
        api.upload_folder(
            folder_path=local_dataset_dir,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Update dataset - {datetime.now().strftime('%Y-%m-%d')}"
        )
        print(f"تم الرفع بنجاح: https://huggingface.co/datasets/{hf_repo_id}")
        logger.info(f"تم رفع البيانات إلى {hf_repo_id}")
        return True
    except Exception as e:
        logger.error(f"فشل الرفع: {e}")
        return False
