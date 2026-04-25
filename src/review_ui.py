"""
HandwrittenOCR - الواجهات التفاعلية للمراجعة
================================================
واجهات لمراجعة وتصحيح نتائج OCR:
1. مراجعة الكلمات (Word-level)
2. مراجعة الجمل (Sentence-level)
3. تعديل قاموس التصحيحات

تدعم Jupyter (ipywidgets) ووضع CLI.
"""

import logging
import json
import os
import pandas as pd
import io as _io
from datetime import datetime

logger = logging.getLogger("HandwrittenOCR")

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from PIL import Image as PILImage
    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False
    logger.info("ipywidgets غير متاح - وضع CLI فقط")


class ReviewUI:
    """واجهة مراجعة تفاعلية لنتائج OCR على مستوى الكلمات."""

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

    def log_correction(self, image_id, original: str, corrected: str, status: str) -> None:
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
            self.feedback_csv, mode="a",
            header=not file_exists,
            index=False, encoding="utf-8",
        )
        logger.info(f"Feedback: ID={image_id}, '{original}' -> '{corrected}'")

    def _launch_jupyter_ui(self) -> None:
        """واجهة Jupyter v3 - إزالة العناصر من العرض فوراً"""
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM handwriting_data WHERE status='unverified' ORDER BY confidence ASC",
                conn
            )
        if df.empty:
            print("لا توجد كلمات جديدة للمراجعة.")
            return

        current = [0]
        img = widgets.Image(format='png', width=350)
        text = widgets.Text(description="النص:")
        conf_bar = widgets.FloatProgress(min=0, max=1.0, description='الثقة:', layout={'width': '95%'})
        conf_label = widgets.HTML(value="")
        prog = widgets.IntProgress(min=0, max=len(df)-1, bar_style='info')
        info = widgets.Label()

        def update():
            if 0 <= current[0] < len(df):
                row = df.iloc[current[0]]
                img.value = row['image_data']
                text.value = str(row['predicted_text'] or '')
                c = float(row['confidence'])
                conf_bar.value = c
                conf_bar.bar_style = 'danger' if c < 0.5 else ('warning' if c < 0.8 else 'success')
                conf_label.value = f"<b>{c:.2%}</b>"
                info.value = f"{current[0]+1}/{len(df)} | صفحة {row['page_num']}"
                prog.value = current[0]
            else:
                img.value = b''; text.value = ''; info.value = "اكتملت المراجعة"

        def on_confirm(b):
            if not (0 <= current[0] < len(df)): return
            rid = int(df.iloc[current[0]]['image_id'])
            original = df.iloc[current[0]]['predicted_text']
            corrected = text.value
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("UPDATE handwriting_data SET predicted_text=?, status='verified' WHERE image_id=?", (corrected, rid))
            if original != corrected:
                self.log_correction(rid, original, corrected, "verified")
            df.drop(df.index[current[0]], inplace=True)
            df.reset_index(drop=True, inplace=True)
            prog.max = max(0, len(df)-1)
            if current[0] >= len(df) and len(df) > 0: current[0] = len(df) - 1
            update()

        def on_next(b):
            current[0] = min(len(df)-1, current[0] + 1)
            update()

        def on_prev(b):
            current[0] = max(0, current[0] - 1)
            update()

        def on_delete(b):
            if not (0 <= current[0] < len(df)): return
            rid = int(df.iloc[current[0]]['image_id'])
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("DELETE FROM handwriting_data WHERE image_id=?", (rid,))
            df.drop(df.index[current[0]], inplace=True)
            df.reset_index(drop=True, inplace=True)
            prog.max = max(0, len(df)-1)
            if current[0] >= len(df) and len(df) > 0: current[0] = len(df)-1
            update()

        btn_confirm = widgets.Button(description="تأكيد", button_style='success')
        btn_next = widgets.Button(description="التالي", button_style='info')
        btn_prev = widgets.Button(description="السابق", button_style='info')
        btn_del = widgets.Button(description="حذف", button_style='danger')
        btn_confirm.on_click(on_confirm)
        btn_next.on_click(on_next)
        btn_prev.on_click(on_prev)
        btn_del.on_click(on_delete)

        display(widgets.VBox([
            prog, info, img, text,
            widgets.HBox([conf_bar, conf_label]),
            widgets.HBox([btn_prev, btn_confirm, btn_del, btn_next])
        ]))
        update()

    def _launch_cli_ui(self) -> None:
        words = self.db.get_unverified(order_by_confidence=True)
        if not words:
            print("لا توجد كلمات جديدة للمراجعة.")
            return

        total = len(words)
        print(f"\nكلمات للمراجعة: {total}")
        print("الأوامر: [n] التالي | [p] السابق | [d] حذف | [q] خروج")
        idx = 0
        while 0 <= idx < total:
            row = words[idx]
            rid = row["image_id"]
            text = row["predicted_text"] or "(فارغ)"
            conf = row.get("confidence", 0)
            print(f"[{idx + 1}/{total}] ID: {rid} | النص: {text} | ثقة: {conf:.2f}")
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
            elif user_input:
                original = row["predicted_text"] or ""
                self.db.update_word(rid, user_input, "verified")
                if original != user_input:
                    self.log_correction(rid, original, user_input, "verified")
                words.pop(idx)
                total = len(words)
                if idx >= total and idx > 0:
                    idx = total - 1
                if total == 0:
                    print("اكتملت المراجعة")
                    break


class SentenceReviewUI:
    """واجهة مراجعة تفاعلية على مستوى الجمل."""

    def __init__(self, db, feedback_csv: str):
        self.db = db
        self.feedback_csv = feedback_csv

    def launch(self, y_tolerance: int = 25) -> None:
        if not HAS_IPYWIDGETS:
            print("واجهة مراجعة الجمل تتطلب ipywidgets (Jupyter/Colab)")
            return
        self._launch_jupyter_ui(y_tolerance)

    def _launch_jupyter_ui(self, y_tolerance: int = 25) -> None:
        """واجهة مراجعة الجمل في Jupyter/Colab"""
        import sqlite3

        with sqlite3.connect(self.db.db_path) as conn:
            words_df = pd.read_sql_query(
                "SELECT * FROM handwriting_data ORDER BY page_num, y, x", conn
            )

        if words_df.empty:
            print("لا توجد بيانات للمراجعة.")
            return

        # تجميع الجمل
        sentences = []
        for page in words_df['page_num'].unique():
            p_words = words_df[words_df['page_num'] == page].copy()
            if p_words.empty:
                continue
            curr_line = [p_words.iloc[0].to_dict()]
            for i in range(1, len(p_words)):
                row = p_words.iloc[i].to_dict()
                if abs(row['y'] - curr_line[-1]['y']) <= y_tolerance:
                    curr_line.append(row)
                else:
                    sentences.append(curr_line)
                    curr_line = [row]
            sentences.append(curr_line)

        current_idx = [0]

        img_container = widgets.HBox(layout={'overflow_x': 'scroll', 'padding': '10px'})
        sentence_input = widgets.Textarea(description="الجملة:", layout={'width': '95%', 'height': '80px'})
        info_label = widgets.Label()
        progress = widgets.IntProgress(min=0, max=len(sentences)-1, layout={'width': '95%', 'height': '20px'})

        def get_img_widget(blob):
            img = PILImage.open(_io.BytesIO(blob))
            buf = _io.BytesIO()
            img.save(buf, format='PNG')
            return widgets.Image(value=buf.getvalue(), format='png', width=120)

        def update_ui():
            if not (0 <= current_idx[0] < len(sentences)):
                return
            sent = sentences[current_idx[0]]
            img_container.children = [get_img_widget(w['image_data']) for w in sent]
            original_text = " ".join([str(w['predicted_text'] or "") for w in sent])
            sentence_input.value = original_text
            info_label.value = f"جملة {current_idx[0]+1} من {len(sentences)} | صفحة {sent[0]['page_num']}"
            progress.value = current_idx[0]

        def save_current(b):
            sent = sentences[current_idx[0]]
            original = " ".join([str(w['predicted_text'] or "") for w in sent])
            corrected = sentence_input.value.strip()
            if not corrected:
                return

            # تحديث حالة الكلمات
            with sqlite3.connect(self.db.db_path) as conn:
                for w in sent:
                    conn.execute(
                        "UPDATE handwriting_data SET status='sentence_corrected' WHERE image_id=?",
                        (w['image_id'],)
                    )

            # تسجيل تصحيح الجملة
            sent_id = f"p{sent[0]['page_num']}_y{sent[0]['y']}"
            if original != corrected:
                pd.DataFrame([{
                    'timestamp': datetime.now().isoformat(),
                    'image_id': None,
                    'original_text': original,
                    'corrected_text': corrected,
                    'status': f'sent_rev_{sent_id}'
                }]).to_csv(self.feedback_csv, mode='a', header=not os.path.exists(self.feedback_csv), index=False, encoding='utf-8')

                # استخراج تصحيحات مشتقة على مستوى الكلمات
                orig_words = original.split()
                corr_words = corrected.split()
                if len(orig_words) == len(corr_words):
                    derived = []
                    for o, c in zip(orig_words, corr_words):
                        if o != c:
                            derived.append({
                                'timestamp': datetime.now().isoformat(),
                                'image_id': None,
                                'original_text': o,
                                'corrected_text': c,
                                'status': 'sentence_derived'
                            })
                    if derived:
                        pd.DataFrame(derived).to_csv(
                            self.feedback_csv, mode='a',
                            header=False, index=False, encoding='utf-8'
                        )

            print(f"تم حفظ الجملة {current_idx[0]+1}")
            current_idx[0] = min(len(sentences)-1, current_idx[0] + 1)
            update_ui()

        def on_next(b):
            current_idx[0] = min(len(sentences)-1, current_idx[0] + 1)
            update_ui()

        def on_prev(b):
            current_idx[0] = max(0, current_idx[0] - 1)
            update_ui()

        btn_save = widgets.Button(description="حفظ وتأكيد", button_style='success')
        btn_next = widgets.Button(description="التالي", button_style='info')
        btn_prev = widgets.Button(description="السابق", button_style='info')
        btn_save.on_click(save_current)
        btn_next.on_click(on_next)
        btn_prev.on_click(on_prev)

        display(widgets.VBox([
            progress, info_label, img_container,
            sentence_input,
            widgets.HBox([btn_prev, btn_save, btn_next])
        ]))
        update_ui()


class CorrectionDictUI:
    """واجهة تعديل قاموس التصحيحات يدوياً."""

    def __init__(self, correction_dict_path: str):
        self.dict_path = correction_dict_path

    def launch(self) -> None:
        if not HAS_IPYWIDGETS:
            print("واجهة تعديل القاموس تتطلب ipywidgets (Jupyter/Colab)")
            return
        self._launch_jupyter_ui()

    def _launch_jupyter_ui(self) -> None:
        data = {}
        if os.path.exists(self.dict_path):
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        out = widgets.Output()
        key_in = widgets.Text(description="أصلي:")
        val_in = widgets.Text(description="تصحيح:")
        status_label = widgets.Label(value=f"عدد التصحيحات: {len(data)}")

        def save(b):
            if key_in.value:
                data[key_in.value] = val_in.value
                with open(self.dict_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                status_label.value = f"عدد التصحيحات: {len(data)}"
                with out:
                    clear_output()
                    print(f"تم حفظ: '{key_in.value}' -> '{val_in.value}'")

        btn = widgets.Button(description="حفظ", button_style='success')
        btn.on_click(save)
        display(widgets.VBox([status_label, key_in, val_in, btn, out]))
