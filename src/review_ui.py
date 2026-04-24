"""
HandwrittenOCR - الواجهة التفاعلية للمراجعة
==============================================
واجهة لمراجعة وتصحيح نتائج OCR يدوياً.
تدعم Jupyter (ipywidgets) ووضع CLI.

v3: إزالة العناصر المؤكدة/المحذوفة من العرض فوراً
مع تعامل صحيح مع الحالات الحدية (df فارغ).
"""

import logging
import pandas as pd
import os
from datetime import datetime

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

    v3: العناصر المؤكدة/المحذوفة تُزال من العرض فوراً
    مع تعامل صحيح مع قائمة فارغة.
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
        """واجهة Jupyter v3 - إزالة العناصر من العرض فوراً"""
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
                progress.value = idx
                conf = row.get("confidence", 0)
                src = row.get("model_source", "none")
                info_label.value = (
                    f"السجل {idx + 1} من {len(words)} "
                    f"(ID: {row['image_id']})"
                )
                conf_label.value = f"الثقة: {conf:.2f} | المصدر: {src}"
            else:
                # حالة: جميع العناصر تمت مراجعتها أو حذفها
                img_widget.value = b""
                txt_input.value = ""
                conf_label.value = ""
                if len(words) == 0:
                    info_label.value = "اكتملت المراجعة"
                else:
                    info_label.value = "لا توجد عناصر متبقية لعرضها"
                progress.value = progress.max

        def on_confirm(b):
            idx = current_index[0]
            if not (0 <= idx < len(words)):
                print("لا توجد عناصر للمراجعة أو اكتملت المراجعة.")
                return

            row = words[idx]
            rid = row["image_id"]
            original = row["predicted_text"] or ""
            corrected = txt_input.value

            # تحديث قاعدة البيانات
            self.db.update_word(rid, corrected, "verified")

            # تسجيل التصحيح
            if original != corrected:
                self.log_correction(rid, original, corrected, "verified")

            # إزالة من العرض المحلي فوراً
            words.pop(idx)
            progress.max = max(0, len(words) - 1)

            # تعديل المؤشر
            if len(words) == 0:
                current_index[0] = 0
            elif idx >= len(words):
                current_index[0] = len(words) - 1

            update_view()
            if len(words) == 0:
                print("اكتملت المراجعة")

        def on_prev(b):
            current_index[0] = max(0, current_index[0] - 1)
            update_view()

        def on_next(b):
            current_index[0] = min(len(words) - 1, current_index[0] + 1)
            update_view()

        def on_delete(b):
            idx = current_index[0]
            if not (0 <= idx < len(words)):
                print("لا توجد عناصر للحذف.")
                return

            rid = words[idx]["image_id"]
            self.db.delete_word(rid)

            # إزالة من العرض المحلي فوراً
            words.pop(idx)
            progress.max = max(0, len(words) - 1)

            # تعديل المؤشر
            if len(words) == 0:
                current_index[0] = 0
            elif idx >= len(words):
                current_index[0] = len(words) - 1

            update_view()
            if len(words) == 0:
                print("اكتملت المراجعة")

        btn_prev = widgets.Button(description="السابق", button_style="info")
        btn_confirm = widgets.Button(description="تأكيد", button_style="success")
        btn_next = widgets.Button(description="التالي", button_style="info")
        btn_del = widgets.Button(description="حذف", button_style="danger")
        btn_prev.on_click(on_prev)
        btn_confirm.on_click(on_confirm)
        btn_next.on_click(on_next)
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
            widgets.HBox([btn_prev, btn_confirm, btn_del, btn_next]),
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
        print("الأوامر: [n] التالي | [p] السابق | [d] حذف | [q] خروج")
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
            elif user_input == "n":
                idx = min(total - 1, idx + 1)
            elif user_input == "p":
                idx = max(0, idx - 1)
            elif user_input == "d":
                self.db.delete_word(rid)
                words.pop(idx)
                total = len(words)
                if idx >= total and idx > 0:
                    idx = total - 1
                if total == 0:
                    print("اكتملت المراجعة")
                    break
                print("تم الحذف")
            elif user_input:
                original = row["predicted_text"] or ""
                self.db.update_word(rid, user_input, "verified")
                if original != user_input:
                    self.log_correction(rid, original, user_input, "verified")
                print(f"تم التحديث: '{original}' -> '{user_input}'")
                words.pop(idx)
                total = len(words)
                if idx >= total and idx > 0:
                    idx = total - 1
                if total == 0:
                    print("اكتملت المراجعة")
                    break

        if total > 0:
            print("\nانتهت المراجعة.")
