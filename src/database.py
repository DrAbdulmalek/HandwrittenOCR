"""
HandwrittenOCR - إدارة قاعدة البيانات
=======================================
عمليات CRUD على قاعدة بيانات SQLite لتخزين نتائج OCR.
"""

import sqlite3
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("HandwrittenOCR")


class HandwritingDB:
    """
    مدير قاعدة بيانات SQLite لتخزين صور الكلمات ونصوصها.
    """

    def __init__(self, db_path: str):
        """
        تهيئة قاعدة البيانات.

        Parameters:
            db_path: مسار ملف قاعدة البيانات
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def _get_conn(self) -> sqlite3.Connection:
        """الحصول على اتصال بقاعدة البيانات"""
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        """إنشاء جدول البيانات إذا لم يكن موجوداً"""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS handwriting_data (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_data BLOB NOT NULL,
                    predicted_text TEXT DEFAULT '',
                    status TEXT DEFAULT 'yes'
                )
            ''')
            conn.commit()
        logger.info(f"قاعدة البيانات جاهزة: {self.db_path}")

    def insert_word(
        self,
        image_data: bytes,
        predicted_text: str,
        status: str = "yes"
    ) -> int:
        """
        إضافة كلمة جديدة إلى قاعدة البيانات.

        Parameters:
            image_data: بيانات الصورة (PNG bytes)
            predicted_text: النص المعترف
            status: حالة التضمين ('yes' أو 'no')

        Returns:
            معرف الصورة المُنشأ
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                'INSERT INTO handwriting_data '
                '(image_data, predicted_text, status) '
                'VALUES (?, ?, ?)',
                (image_data, predicted_text, status)
            )
            conn.commit()
            return cursor.lastrowid

    def update_word(
        self,
        image_id: int,
        predicted_text: Optional[str] = None,
        status: Optional[str] = None
    ) -> None:
        """
        تحديث نص أو حالة كلمة.

        Parameters:
            image_id: معرف الصورة
            predicted_text: النص الجديد (اختياري)
            status: الحالة الجديدة (اختياري)
        """
        updates = []
        params = []

        if predicted_text is not None:
            updates.append("predicted_text = ?")
            params.append(predicted_text)
        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if not updates:
            return

        params.append(image_id)
        sql = f"UPDATE handwriting_data SET {', '.join(updates)} WHERE image_id = ?"

        with self._get_conn() as conn:
            conn.execute(sql, params)
            conn.commit()

    def delete_word(self, image_id: int) -> bool:
        """
        حذف كلمة من قاعدة البيانات.

        Parameters:
            image_id: معرف الصورة

        Returns:
            True إذا تم الحذف بنجاح
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                'DELETE FROM handwriting_data WHERE image_id = ?',
                (image_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_word(self, image_id: int) -> Optional[dict]:
        """الحصول على بيانات كلمة واحدة"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT * FROM handwriting_data WHERE image_id = ?',
                (image_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_all_words(self) -> list[dict]:
        """الحصول على جميع الكلمات"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM handwriting_data ORDER BY image_id'
            ).fetchall()
            return [dict(row) for row in rows]

    def get_count(self) -> int:
        """الحصول على عدد الكلمات المخزنة"""
        with self._get_conn() as conn:
            result = conn.execute(
                'SELECT COUNT(*) FROM handwriting_data'
            ).fetchone()
            return result[0]

    def clear_all(self) -> int:
        """حذف جميع البيانات وإرجاع عدد الصفوف المحذوفة"""
        with self._get_conn() as conn:
            cursor = conn.execute('DELETE FROM handwriting_data')
            conn.commit()
            return cursor.rowcount
