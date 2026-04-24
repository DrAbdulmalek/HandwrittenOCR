"""
HandwrittenOCR - تدريب LoRA على TrOCR
========================================
Fine-tune TrOCR باستخدام LoRA على تصحيحات المستخدم.
"""

import os
import io
import logging
from PIL import Image

logger = logging.getLogger("HandwrittenOCR")


def finetune_trocr_lora(
    trocr_model,
    trocr_processor,
    db,
    device,
    save_path: str,
    min_samples: int = 100,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: list | None = None,
) -> bool:
    """
    تدريب TrOCR باستخدام LoRA على البيانات الموثقة.

    Parameters:
        trocr_model: نموذج TrOCR
        trocr_processor: معالج TrOCR
        db: قاعدة البيانات
        device: الجهاز (cuda/cpu)
        save_path: مسار حفظ النموذج
        min_samples: الحد الأدنى من العينات للتدريب
        epochs: عدد الحقب
        batch_size: حجم الدفعة
        lr: معدل التعلم
        lora_r: بعد LoRA
        lora_alpha: معامل LoRA
        lora_dropout: نسبة التسرب
        lora_target_modules: الوحدات المستهدفة

    Returns:
        True عند النجاح
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        from torch.optim import AdamW
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        logger.error(
            "peft غير مثبت. ثبّته بـ: pip install peft"
        )
        return False

    if lora_target_modules is None:
        lora_target_modules = ["query", "value"]

    # فحص عدد العينات
    verified = db.get_verified()
    if len(verified) < min_samples:
        print(
            f"لديك {len(verified)} عينة فقط. "
            f"الحد الأدنى المطلوب: {min_samples}."
        )
        return False

    print(f"بدء التدريب على {len(verified)} عينة...")

    # إعداد LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
    )
    model = get_peft_model(trocr_model, lora_config)
    model.train()

    # إنشاء مجموعة البيانات
    import pandas as pd

    class HandwritingDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            row = self.records[idx]
            img = Image.open(io.BytesIO(row["image_data"])).convert("RGB")
            pixel_values = trocr_processor(
                images=img, return_tensors="pt"
            ).pixel_values.squeeze()

            text = row["predicted_text"] or ""
            labels = trocr_processor.tokenizer(
                text, return_tensors="pt",
                padding="max_length", max_length=50
            ).input_ids.squeeze()
            labels[labels == trocr_processor.tokenizer.pad_token_id] = -100

            return {"pixel_values": pixel_values, "labels": labels}

    dataset = HandwritingDataset(verified)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)

    # التدريب
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for batch in loader:
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device)
            )
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
            batch_count += 1

        avg_loss = total_loss / max(batch_count, 1)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # حفظ النموذج
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    trocr_processor.save_pretrained(save_path)

    print(f"تم حفظ النموذج في: {save_path}")
    logger.info(f"تم تدريب LoRA وحفظه في: {save_path}")

    return True
