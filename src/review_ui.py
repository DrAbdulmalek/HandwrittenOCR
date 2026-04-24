"""
HandwrittenOCR - الواجهة التفاعلية للمراجعة
==============================================
واجهة لمراجعة وتصحيح نتائج OCR يدوياً.
تدعم Jupyter (ipywidgets) ووضع CLI.

v2: تعرض فقط الكلمات غير المراجعة (unverified)
مرتبة حسب الثقة (الأقل أولاً) لتسهيل المراجعة.
"""

import logging
import pandas as pd
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("HandwrittenOCR")

try:
    import ipywidgets as widgets
    from IPython.display import display
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False
    logger.info("ipywidgets غير متاح - وضع CLI فقط")


class ReviewUI:
    """
    واجهة مراجعة تفاعلية لنتائج OCR.

    v2: تعرض الكلمات غير المراجعة مرتبة حسب الثقة.
    """

    def __init__(self, db, feedback_csv: str):
        self.db = db
        self.feedback_csv = feedback_csv

    def launch(self) -> None:
        if HAS_IPYWIDGETS:
            logger.info("تشغيل واجهة Jupyter التفاعلية")
            self._launch_jupyter_ui()
        else:
            logger.info("ipywidgets غير متاح - تشغيل واجهة CLI")
            self._launch_cli_ui()

    def log_correction(
        self,
        image_id: int,
        original: str,
        corrected: str,
        status: str
    ) -> None:
        """تسجيل التصحيح في ملف CSV"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "image_id": image_id,
            "original_text": original,
            "corrected_text": corrected,
            "status": status,
        }
        file_exists = os.path.exists(self.feedback_csv)
        pd.DataFrame([record]).to_csv(
            self.feedback_csv,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
        )
        logger.info(
            f"Feedback: ID={image_id}, "
            f"'{original}' -> '{corrected}'"
        )

    # --- واجهة Jupyter (ipywidgets) ---

    def _launch_jupyter_ui(self) -> None:
        """تشغيل واجهة Jupyter - تعرض unverified مرتبة حسب الثقة"""
        words = self.db.get_unverified(order_by_confidence=True)

        if not words:
            print("لا توجد كلمات جديدة للمراجعة.")
            return

        current_index = [0]

        img_widget = widgets.Image(format="png", width=350)
        txt_input = widgets.Text(
            description="النص الصحيح:",
            layout=widgets.Layout(width="95%"),
        )
        status_check = widgets.Checkbox(
            value=True,
            description="تضمين في التدريب",
        )
        progress = widgets.IntProgress(
            min=0,
            max=len(words) - 1,
            bar_style="info",
            layout=widgets.Layout(width="95%"),
        )
        info_label = widgets.Label()
        conf_label = widgets.Label()

        def update_view():
            idx = current_index[0]
            if 0 <= idx < len(words):
                row = words[idx]
                img_widget.value = row["image_data"]
                txt_input.value = row["predicted_text"] or ""
                status_check.value = True
                progress.value = idx
                conf = row.get("confidence", 0)
                src = row.get("model_source", "none")
                info_label.value = (
                    f"السجل {idx + 1} من {len(words)} "
                    f"(ID: {row['image_id']})"
                )
                conf_label.value = f"الثقة: {conf:.2f} | المصدر: {src}"

        def on_confirm(b):
            idx = current_index[0]
            if idx >= len(words):
                return
            row = words[idx]
            rid = row["image_id"]
            original = row["predicted_text"] or ""
            corrected = txt_input.value
            new_status = "verified" if status_check.value else "unverified"

            self.db.update_word(rid, corrected, new_status)

            if original != corrected:
                self.log_correction(rid, original, corrected, new_status)

            # انتقل للتالي (أو أعلن النهاية)
            current_index[0] = idx + 1
            if current_index[0] < len(words):
                update_view()
            else:
                print("اكتملت المراجعة")

        def on_prev(b):
            current_index[0] = max(0, current_index[0] - 1)
            update_view()

        def on_skip(b):
            current_index[0] = min(len(words) - 1, current_index[0] + 1)
            if current_index[0] < len(words):
                update_view()
            else:
                print("اكتملت المراجعة")

        def on_delete(b):
            idx = current_index[0]
            if idx >= len(words):
                return
            rid = words[idx]["image_id"]
            self.db.delete_word(rid)
            words.pop(idx)
            progress.max = max(0, len(words) - 1)
            if idx >= len(words) and idx > 0:
                current_index[0] = len(words) - 1
            update_view()

        btn_confirm = widgets.Button(description="تأكيد", button_style="success")
        btn_prev = widgets.Button(description="السابق", button_style="info")
        btn_skip = widgets.Button(description="تخطي", button_style="warning")
        btn_del = widgets.Button(description="حذف", button_style="danger")
        btn_confirm.on_click(on_confirm)
        btn_prev.on_click(on_prev)
        btn_skip.on_click(on_skip)
        btn_del.on_click(on_delete)

        ui = widgets.VBox([
            widgets.HTML("<h3>مراجعة وتصحيح نصوص الخط اليدوي</h3>"),
            progress,
            conf_label,
            info_label,
            widgets.Box(
                [img_widget],
                layout=widgets.Layout(
                    display="flex", justify_content="center", padding="10px"
                ),
            ),
            txt_input,
            status_check,
            widgets.HBox([btn_prev, btn_confirm, btn_skip, btn_del]),
        ])

        display(ui)
        update_view()

    # --- واجهة CLI ---

    def _launch_cli_ui(self) -> None:
        words = self.db.get_unverified(order_by_confidence=True)

        if not words:
            print("لا توجد كلمات جديدة للمراجعة.")
            return

        total = len(words)
        print(f"\nكلمات للمراجعة: {total}")
        print("الأوامر: [n] التالي | [p] السابق | [s] تخطي | [d] حذف | [q] خروج")
        print("للتصحيح: اكتب النص الجديد ثم اضغط Enter\n")

        idx = 0
        while 0 <= idx < total:
            row = words[idx]
            rid = row["image_id"]
            text = row["predicted_text"] or "(فارغ)"
            conf = row.get("confidence", 0)
            src = row.get("model_source", "none")

            print(f"[{idx + 1}/{total}] ID: {rid} | النص: {text} | ثقة: {conf:.2f} | {src}")

            preview_path = f"/tmp/ocr_preview_{rid}.png"
            with open(preview_path, "wb") as f:
                f.write(row["image_data"])
            print(f"معاينة: {preview_path}")

            user_input = input("تصحيح (أو أمر): ").strip()

            if user_input == "q":
                break
            elif user_input == "n" or user_input == "s":
                idx = min(total - 1, idx + 1)
            elif user_input == "p":
                idx = max(0, idx - 1)
            elif user_input == "d":
                self.db.delete_word(rid)
                words.pop(idx)
                total = len(words)
                if idx >= total and idx > 0:
                    idx = total - 1
                print("تم الحذف")
            elif user_input:
                original = row["predicted_text"] or ""
                self.db.update_word(rid, user_input, "verified")
                if original != user_input:
                    self.log_correction(rid, original, user_input, "verified")
                print(f"تم التحديث: '{original}' -> '{user_input}'")
                idx = min(total - 1, idx + 1)

        print("\nانتهت المراجعة.")
