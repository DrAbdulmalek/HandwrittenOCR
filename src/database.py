"""
HandwrittenOCR - إدارة قاعدة البيانات
=======================================
عمليات CRUD على قاعدة بيانات SQLite لتخزين نتائج OCR.
مخطط v2: يدعم confidence, model_source, إحداثيات الموقع, رقم الصفحة.
"""

import sqlite3
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("HandwrittenOCR")

# إصدار مخطط قاعدة البيانات
DB_SCHEMA_VERSION = 2

# أسماء الأعمدة في المخطط الجديد
DB_COLUMNS = [
    "image_id", "image_data", "predicted_text", "status",
    "confidence", "model_source",
    "x", "y", "w", "h", "page_num"
]


class HandwritingDB:
    """
    مدير قاعدة بيانات SQLite لتخزين صور الكلمات ونصوصها.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_table()
        self._migrate_if_needed()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        """إنشاء جدول البيانات إذا لم يكن موجوداً (المخطط v2)"""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS handwriting_data (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_data BLOB NOT NULL,
                    predicted_text TEXT DEFAULT '',
                    status TEXT DEFAULT 'unverified',
                    confidence REAL DEFAULT 0.0,
                    model_source TEXT DEFAULT 'none',
                    x INTEGER DEFAULT 0,
                    y INTEGER DEFAULT 0,
                    w INTEGER DEFAULT 0,
                    h INTEGER DEFAULT 0,
                    page_num INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
        logger.info(f"قاعدة البيانات جاهزة: {self.db_path}")

    def _migrate_if_needed(self) -> None:
        """
        ترقية مخطط قاعدة البيانات من v1 إلى v2.
        المخطط القديم: (image_id, image_data, predicted_text, status)
        المخطط الجديد: يضيف confidence, model_source, x, y, w, h, page_num
        """
        with self._get_conn() as conn:
            # فحص الأعمدة الموجودة
            cursor = conn.execute("PRAGMA table_info(handwriting_data)")
            existing_cols = {row[1] for row in cursor.fetchall()}

            new_cols = {
                "confidence": "REAL DEFAULT 0.0",
                "model_source": "TEXT DEFAULT 'none'",
                "x": "INTEGER DEFAULT 0",
                "y": "INTEGER DEFAULT 0",
                "w": "INTEGER DEFAULT 0",
                "h": "INTEGER DEFAULT 0",
                "page_num": "INTEGER DEFAULT 0",
            }

            migrated = False
            for col_name, col_type in new_cols.items():
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE handwriting_data "
                        f"ADD COLUMN {col_name} {col_type}"
                    )
                    migrated = True

            # تحويل القيم القديمة: 'yes' -> 'verified', 'no' -> 'unverified'
            if "status" in existing_cols:
                conn.execute(
                    "UPDATE handwriting_data "
                    "SET status = 'verified' WHERE status = 'yes'"
                )
                conn.execute(
                    "UPDATE handwriting_data "
                    "SET status = 'unverified' WHERE status = 'no'"
                )
                conn.execute(
                    "UPDATE handwriting_data "
                    "SET status = 'unverified' WHERE status IS NULL "
                    "OR status NOT IN ('verified', 'unverified')"
                )

            if migrated:
                conn.commit()
                logger.info("تم ترقية مخطط قاعدة البيانات إلى v2")

    def insert_word(
        self,
        image_data: bytes,
        predicted_text: str,
        status: str = "unverified",
        confidence: float = 0.0,
        model_source: str = "none",
        x: int = 0,
        y: int = 0,
        w: int = 0,
        h: int = 0,
        page_num: int = 0,
    ) -> int:
        """إضافة كلمة جديدة إلى قاعدة البيانات (المخطط v2)"""
        with self._get_conn() as conn:
            cursor = conn.execute(
                '''INSERT INTO handwriting_data
                   (image_data, predicted_text, status, confidence,
                    model_source, x, y, w, h, page_num)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (image_data, predicted_text, status, confidence,
                 model_source, x, y, w, h, page_num)
            )
            conn.commit()
            return cursor.lastrowid

    def update_word(
        self,
        image_id: int,
        predicted_text: Optional[str] = None,
        status: Optional[str] = None
    ) -> None:
        """تحديث نص أو حالة كلمة"""
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
        """حذف كلمة من قاعدة البيانات"""
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
            return dict(row) if row else None

    def get_all_words(self) -> list[dict]:
        """الحصول على جميع الكلمات"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM handwriting_data ORDER BY image_id'
            ).fetchall()
            return [dict(row) for row in rows]

    def get_unverified(self, order_by_confidence: bool = True) -> list[dict]:
        """
        الحصول على الكلمات غير المراجعة.
        ترتيب حسب الثقة (الأقل أولاً) لتسهيل المراجعة.
        """
        order = "ORDER BY confidence ASC" if order_by_confidence else "ORDER BY image_id"
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f'''SELECT * FROM handwriting_data
                    WHERE status = 'unverified' {order}'''
            ).fetchall()
            return [dict(row) for row in rows]

    def get_verified(self) -> list[dict]:
        """الحصول على الكلمات المراجعة فقط (للتصدير والتدريب)"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                '''SELECT * FROM handwriting_data
                    WHERE status = 'verified' ORDER BY image_id'''
            ).fetchall()
            return [dict(row) for row in rows]

    def get_count(self) -> int:
        """الحصول على عدد الكلمات المخزنة"""
        with self._get_conn() as conn:
            result = conn.execute(
                'SELECT COUNT(*) FROM handwriting_data'
            ).fetchone()
            return result[0]

    def get_verified_count(self) -> int:
        """عدد الكلمات المراجعة"""
        with self._get_conn() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM handwriting_data WHERE status = 'verified'"
            ).fetchone()
            return result[0]

    def get_unverified_count(self) -> int:
        """عدد الكلمات غير المراجعة"""
        with self._get_conn() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM handwriting_data WHERE status = 'unverified'"
            ).fetchone()
            return result[0]

    def clear_all(self) -> int:
        """حذف جميع البيانات وإرجاع عدد الصفوف المحذوفة"""
        with self._get_conn() as conn:
            cursor = conn.execute('DELETE FROM handwriting_data')
            conn.commit()
            return cursor.rowcount
