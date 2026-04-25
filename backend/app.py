"""
HandwrittenOCR - FastAPI Backend
=================================
REST API for the HandwrittenOCR project, designed to run on Google Colab.
Serves as the API for the web frontend.
"""

import os
import sys
import json
import csv
import sqlite3
import shutil
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Ensure project root is importable (backend/ lives inside the project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from src.database import HandwritingDB
from src.recognition import OCREngine
from src.correction import (
    init_correctors,
    load_correction_dict,
    correct_text,
)
from src.reconstruction import reconstruct_sentences
from src.export import export_finetuning_dataset, push_to_huggingface
from src.finetuning import finetune_trocr_lora
from src.pdf_processor import PDFProcessor
from src.logger import setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HandwrittenOCR.API")

# ---------------------------------------------------------------------------
# Global state (lazily initialised on first request or via lifespan)
# ---------------------------------------------------------------------------
_config: Optional[Config] = None
_db: Optional[HandwritingDB] = None
_ocr_engine: Optional[OCREngine] = None
_models_loaded = False
_processing_lock = threading.Lock()
_finetuning_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lazy-initialisation helpers
# ---------------------------------------------------------------------------
def _ensure_config() -> Config:
    """Return the global Config, creating it on first call."""
    global _config
    if _config is None:
        _config = Config()
        _config.apply_hf_token()
        _config.apply_cache_env()
        _config.ensure_dirs()
    return _config


def _ensure_db() -> HandwritingDB:
    """Return the global DB handle, creating it on first call."""
    global _db
    cfg = _ensure_config()
    if _db is None:
        _db = HandwritingDB(cfg.db_path)
    return _db


def _ensure_ocr() -> OCREngine:
    """Return the global OCREngine, creating it on first call."""
    global _ocr_engine, _models_loaded
    cfg = _ensure_config()
    if _ocr_engine is None:
        logger.info("Loading OCR models (first request)...")
        init_correctors()
        _ocr_engine = OCREngine(
            trocr_model_name=cfg.trocr_model_name,
            ocr_languages=cfg.ocr_languages,
            max_text_length=cfg.max_text_length,
            cache_dir=cfg.model_cache_dir,
            hf_token=cfg.hf_token,
            trocr_default_confidence=cfg.trocr_default_confidence,
            lora_save_path=cfg.lora_save_path,
        )
        _models_loaded = True
        logger.info("OCR models loaded successfully.")
    return _ocr_engine


def _get_device() -> str:
    """Return a human-readable device string."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _checkpoint_path(cfg: Config) -> str:
    return os.path.join(cfg.output_dir, "checkpoint.json")


def _load_checkpoint(cfg: Optional[Config] = None) -> Optional[dict]:
    cfg = cfg or _ensure_config()
    path = _checkpoint_path(cfg)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_checkpoint(data: dict, cfg: Optional[Config] = None) -> None:
    cfg = cfg or _ensure_config()
    path = _checkpoint_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _clear_checkpoint(cfg: Optional[Config] = None) -> bool:
    cfg = cfg or _ensure_config()
    path = _checkpoint_path(cfg)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ---------------------------------------------------------------------------
# Feedback CSV helpers
# ---------------------------------------------------------------------------
def _append_feedback(
    image_id: int,
    original_text: str,
    corrected_text: str,
    status: str,
    cfg: Optional[Config] = None,
) -> None:
    """Append a correction entry to the feedback CSV."""
    cfg = cfg or _ensure_config()
    os.makedirs(cfg.logs_dir, exist_ok=True)
    file_exists = os.path.isfile(cfg.feedback_csv)
    with open(cfg.feedback_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "image_id", "original_text",
                "corrected_text", "status",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            image_id,
            original_text,
            corrected_text,
            status,
        ])


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class ProcessPDFRequest(BaseModel):
    pdf_path: str
    pages_start: int = 1
    pages_end: int = 2
    resume: bool = False


class UpdateWordRequest(BaseModel):
    predicted_text: str
    status: str


class SentenceItem(BaseModel):
    word_ids: list[int]
    original: str
    corrected: str
    page: int


class SaveSentencesRequest(BaseModel):
    sentences: list[SentenceItem]


class CorrectionRequest(BaseModel):
    original: str
    corrected: str


class ExportDatasetRequest(BaseModel):
    val_ratio: float = 0.1


class FinetuneRequest(BaseModel):
    min_samples: int = 100


class PushHFRequest(BaseModel):
    repo_id: str
    token: str


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="HandwrittenOCR API",
    description="REST API for handwritten text OCR, review, fine-tuning and export.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# 1. GET /api/health
# ===========================================================================
@app.get("/api/health")
async def health_check():
    """Health check – returns status, device and model readiness."""
    return {
        "status": "ok",
        "device": _get_device(),
        "models_loaded": _models_loaded,
    }


# ===========================================================================
# 2. GET /api/stats
# ===========================================================================
@app.get("/api/stats")
async def get_stats():
    """Aggregate processing statistics from JSON + DB counts."""
    cfg = _ensure_config()
    db = _ensure_db()

    # --- stats JSON ---
    stats_from_file = {}
    if os.path.isfile(cfg.stats_json):
        try:
            with open(cfg.stats_json, "r", encoding="utf-8") as f:
                stats_from_file = json.load(f)
        except Exception:
            pass

    # --- DB counts (raw SQL for statuses not exposed by the ORM) ---
    conn = db._get_conn()
    try:
        total_words = conn.execute(
            "SELECT COUNT(*) FROM handwriting_data"
        ).fetchone()[0]

        verified_words = conn.execute(
            "SELECT COUNT(*) FROM handwriting_data WHERE status = 'verified'"
        ).fetchone()[0]

        unverified_words = conn.execute(
            "SELECT COUNT(*) FROM handwriting_data WHERE status = 'unverified'"
        ).fetchone()[0]

        sentence_corrected = conn.execute(
            "SELECT COUNT(*) FROM handwriting_data WHERE status = 'sentence_corrected'"
        ).fetchone()[0]

        # Count distinct pages
        pages_row = conn.execute(
            "SELECT COUNT(DISTINCT page_num) FROM handwriting_data WHERE page_num > 0"
        ).fetchone()
        total_pages = pages_row[0] if pages_row else 0
    finally:
        conn.close()

    return {
        "total_words": total_words,
        "verified_words": verified_words,
        "unverified_words": unverified_words,
        "sentence_corrected": sentence_corrected,
        "total_pages": total_pages,
        "last_run": stats_from_file,
    }


# ===========================================================================
# 3. POST /api/process-pdf
# ===========================================================================
@app.post("/api/process-pdf")
async def process_pdf(req: ProcessPDFRequest):
    """Start PDF processing in a background thread."""
    if not os.path.isfile(req.pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")

    if _processing_lock.locked():
        raise HTTPException(status_code=409, detail="PDF processing already in progress.")

    cfg = _ensure_config()
    cfg.pdf_path = req.pdf_path
    cfg.pages_start = req.pages_start
    cfg.pages_end = req.pages_end

    def _run():
        try:
            ocr = _ensure_ocr()
            db = _ensure_db()
            processor = PDFProcessor(cfg, ocr, db)

            # Save checkpoint before processing
            _save_checkpoint({
                "pdf_path": req.pdf_path,
                "pages_start": req.pages_start,
                "pages_end": req.pages_end,
                "started_at": datetime.now().isoformat(),
                "status": "processing",
            })

            stats = processor.process()

            # Update checkpoint with result
            _save_checkpoint({
                "pdf_path": req.pdf_path,
                "pages_start": req.pages_start,
                "pages_end": req.pages_end,
                "started_at": datetime.now().isoformat(),
                "status": "completed",
                "stats": stats,
            })
            logger.info(f"PDF processing completed: {stats.get('total_words', 0)} words")
        except Exception as exc:
            logger.error(f"PDF processing failed: {exc}", exc_info=True)
            _save_checkpoint({
                "pdf_path": req.pdf_path,
                "pages_start": req.pages_start,
                "pages_end": req.pages_end,
                "started_at": datetime.now().isoformat(),
                "status": "failed",
                "error": str(exc),
            })
        finally:
            # Lock is released when the thread exits the 'with' block,
            # but we use a non-context-manager lock so release explicitly.
            _processing_lock.release()

    _processing_lock.acquire()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "started"}


# ===========================================================================
# 4. GET /api/checkpoint
# ===========================================================================
@app.get("/api/checkpoint")
async def get_checkpoint():
    """Return the current processing checkpoint or null."""
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        return {"checkpoint": None}
    return {"checkpoint": checkpoint}


# ===========================================================================
# 5. DELETE /api/checkpoint
# ===========================================================================
@app.delete("/api/checkpoint")
async def delete_checkpoint():
    """Clear the processing checkpoint."""
    cleared = _clear_checkpoint()
    return {"success": cleared}


# ===========================================================================
# 6. GET /api/words
# ===========================================================================
@app.get("/api/words")
async def get_words(
    status: str = Query("unverified"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    sort_by: str = Query("confidence"),
    sort_order: str = Query("asc"),
):
    """
    Return a paginated list of words.

    Excludes the ``image_data`` BLOB to keep payloads small.
    """
    db = _ensure_db()

    # Validate sort direction
    order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    # Whitelist sortable columns
    allowed_sort = {"confidence", "image_id", "page_num", "y", "predicted_text"}
    if sort_by not in allowed_sort:
        sort_by = "confidence"

    conn = db._get_conn()
    try:
        conn.row_factory = sqlite3.Row

        # Count total matching rows
        total = conn.execute(
            "SELECT COUNT(*) FROM handwriting_data WHERE status = ?",
            (status,),
        ).fetchone()[0]

        # Fetch paginated rows (exclude image_data)
        offset = (page - 1) * limit
        cols = "image_id, predicted_text, status, confidence, model_source, x, y, w, h, page_num"
        rows = conn.execute(
            f"SELECT {cols} FROM handwriting_data "
            f"WHERE status = ? ORDER BY {sort_by} {order_dir} "
            f"LIMIT ? OFFSET ?",
            (status, limit, offset),
        ).fetchall()

        words = [dict(r) for r in rows]
    finally:
        conn.close()

    return {
        "words": words,
        "total": total,
        "page": page,
        "limit": limit,
    }


# ===========================================================================
# 7. GET /api/words/{image_id}/image
# ===========================================================================
@app.get("/api/words/{image_id}/image")
async def get_word_image(image_id: int):
    """Return the word image as a PNG binary response."""
    db = _ensure_db()
    word = db.get_word(image_id)
    if word is None:
        raise HTTPException(status_code=404, detail=f"Word {image_id} not found.")

    image_data = word.get("image_data")
    if not image_data:
        raise HTTPException(status_code=404, detail="No image data for this word.")

    return Response(
        content=bytes(image_data),
        media_type="image/png",
    )


# ===========================================================================
# 8. PUT /api/words/{image_id}
# ===========================================================================
@app.put("/api/words/{image_id}")
async def update_word(image_id: int, req: UpdateWordRequest):
    """Update a word's text and/or status."""
    db = _ensure_db()
    cfg = _ensure_config()

    word = db.get_word(image_id)
    if word is None:
        raise HTTPException(status_code=404, detail=f"Word {image_id} not found.")

    original_text = word["predicted_text"]

    db.update_word(
        image_id=image_id,
        predicted_text=req.predicted_text,
        status=req.status,
    )

    # Log to feedback CSV if text changed AND status is verified or sentence_corrected
    if req.predicted_text != original_text and req.status in (
        "verified",
        "sentence_corrected",
    ):
        _append_feedback(
            image_id=image_id,
            original_text=original_text,
            corrected_text=req.predicted_text,
            status=req.status,
            cfg=cfg,
        )

    return {"success": True}


# ===========================================================================
# 9. DELETE /api/words/{image_id}
# ===========================================================================
@app.delete("/api/words/{image_id}")
async def delete_word(image_id: int):
    """Delete a word from the database."""
    db = _ensure_db()
    deleted = db.delete_word(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Word {image_id} not found.")
    return {"success": True}


# ===========================================================================
# 10. GET /api/sentences
# ===========================================================================
@app.get("/api/sentences")
async def get_sentences(
    y_tolerance: int = Query(25, ge=5, le=100),
):
    """
    Reconstruct sentences from verified + sentence_corrected words.

    Returns sentences grouped by page and Y-line, with ``word_ids`` so
    the frontend can trace corrections back to individual words.
    """
    db = _ensure_db()

    # Fetch all relevant words (verified + sentence_corrected)
    conn = db._get_conn()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM handwriting_data "
            "WHERE status IN ('verified', 'sentence_corrected') "
            "ORDER BY page_num, y, x"
        ).fetchall()
        words = [dict(r) for r in rows]
    finally:
        conn.close()

    if not words:
        return {"sentences": []}

    sentences = []
    pages = sorted(set(w["page_num"] for w in words if w["page_num"] > 0))

    for page_num in pages:
        p_words = [w for w in words if w["page_num"] == page_num]
        p_words.sort(key=lambda k: (k["y"], k["x"]))

        if not p_words:
            continue

        # Split into lines by y tolerance
        lines: list[list[dict]] = []
        current_line = [p_words[0]]
        for i in range(1, len(p_words)):
            row = p_words[i]
            if abs(row["y"] - current_line[-1]["y"]) <= y_tolerance:
                current_line.append(row)
            else:
                lines.append(current_line)
                current_line = [row]
        lines.append(current_line)

        for line in lines:
            text_preview = " ".join(str(w["predicted_text"]) for w in line)

            # Detect language
            lang = "en"
            try:
                from langdetect import detect
                lang = detect(text_preview)
            except Exception:
                pass

            # RTL ordering for Arabic
            sorted_line = sorted(
                line,
                key=lambda k: k["x"],
                reverse=(lang == "ar"),
            )

            sentence_text = " ".join(str(w["predicted_text"]) for w in sorted_line)
            word_ids = [w["image_id"] for w in sorted_line]

            sentences.append({
                "page": page_num,
                "text": sentence_text,
                "lang": lang,
                "word_ids": word_ids,
            })

    return {"sentences": sentences}


# ===========================================================================
# 11. PUT /api/sentences
# ===========================================================================
@app.put("/api/sentences")
async def save_sentence_corrections(req: SaveSentencesRequest):
    """
    Save sentence-level corrections.

    - Updates all referenced words' status to ``sentence_corrected``.
    - Logs corrections to the feedback CSV.
    - If word counts match between original and corrected, extracts
      derived word-level corrections and updates them as well.
    """
    db = _ensure_db()
    cfg = _ensure_config()
    total_updated = 0

    for sentence in req.sentences:
        word_ids = sentence.word_ids
        original = sentence.original
        corrected = sentence.corrected
        page = sentence.page

        if not word_ids:
            continue

        # Update status for all words in this sentence
        for wid in word_ids:
            word = db.get_word(wid)
            if word is None:
                continue

            db.update_word(wid, status="sentence_corrected")

            # Log to feedback CSV if text is being corrected at word level
            if word["predicted_text"] != corrected:
                _append_feedback(
                    image_id=wid,
                    original_text=word["predicted_text"],
                    corrected_text=corrected,
                    status="sentence_corrected",
                    cfg=cfg,
                )

        # Derive word-level corrections if counts match
        orig_words = original.split()
        corr_words = corrected.split()

        if len(orig_words) == len(corr_words) and len(orig_words) == len(word_ids):
            for i, wid in enumerate(word_ids):
                if orig_words[i] != corr_words[i]:
                    word = db.get_word(wid)
                    if word and word["predicted_text"] != corr_words[i]:
                        db.update_word(wid, predicted_text=corr_words[i])
                        _append_feedback(
                            image_id=wid,
                            original_text=word["predicted_text"],
                            corrected_text=corr_words[i],
                            status="sentence_corrected",
                            cfg=cfg,
                        )

        total_updated += len(word_ids)

    return {"success": True, "updated": total_updated}


# ===========================================================================
# 12. GET /api/correction-dict
# ===========================================================================
@app.get("/api/correction-dict")
async def get_correction_dict():
    """Return the full correction dictionary."""
    cfg = _ensure_config()
    data = load_correction_dict(cfg.correction_dict_path)
    return {"corrections": data}


# ===========================================================================
# 13. POST /api/correction-dict
# ===========================================================================
@app.post("/api/correction-dict")
async def add_correction(req: CorrectionRequest):
    """Add or update a single correction in the dictionary."""
    cfg = _ensure_config()

    existing = load_correction_dict(cfg.correction_dict_path)
    existing[req.original] = req.corrected

    os.makedirs(os.path.dirname(cfg.correction_dict_path), exist_ok=True)
    with open(cfg.correction_dict_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return {"success": True}


# ===========================================================================
# 14. DELETE /api/correction-dict/{original}
# ===========================================================================
@app.delete("/api/correction-dict/{original}")
async def delete_correction(original: str):
    """Delete a correction entry from the dictionary."""
    cfg = _ensure_config()

    existing = load_correction_dict(cfg.correction_dict_path)
    if original not in existing:
        raise HTTPException(status_code=404, detail=f"Correction '{original}' not found.")

    del existing[original]

    os.makedirs(os.path.dirname(cfg.correction_dict_path), exist_ok=True)
    with open(cfg.correction_dict_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return {"success": True}


# ===========================================================================
# 15. POST /api/export-dataset
# ===========================================================================
@app.post("/api/export-dataset")
async def export_dataset(req: ExportDatasetRequest):
    """Export the fine-tuning dataset (train/val JSONL + images)."""
    cfg = _ensure_config()
    db = _ensure_db()

    output_dir = export_finetuning_dataset(
        db=db,
        output_dir=cfg.export_dir,
        val_ratio=req.val_ratio,
    )

    if output_dir is None:
        raise HTTPException(
            status_code=400,
            detail="No verified data available for export.",
        )

    # Count train / val samples
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    train_count = _count_jsonl(train_path)
    val_count = _count_jsonl(val_path)

    return {
        "success": True,
        "path": os.path.abspath(output_dir),
        "train_count": train_count,
        "val_count": val_count,
    }


def _count_jsonl(path: str) -> int:
    """Count the number of lines in a JSONL file."""
    if not os.path.isfile(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ===========================================================================
# 16. POST /api/finetune
# ===========================================================================
@app.post("/api/finetune")
async def start_finetune(req: FinetuneRequest):
    """Start LoRA fine-tuning in a background thread."""
    if _finetuning_lock.locked():
        raise HTTPException(status_code=409, detail="Fine-tuning already in progress.")

    cfg = _ensure_config()

    def _run():
        try:
            ocr = _ensure_ocr()
            db = _ensure_db()
            success = finetune_trocr_lora(
                ocr_engine=ocr,
                db=db,
                save_path=cfg.lora_save_path,
                min_samples=req.min_samples,
                epochs=cfg.finetune_epochs,
                batch_size=cfg.finetune_batch_size,
                lr=cfg.finetune_lr,
                lora_r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                lora_target_modules=cfg.lora_target_modules,
            )
            if success:
                logger.info("LoRA fine-tuning completed successfully.")
            else:
                logger.warning("LoRA fine-tuning did not complete successfully.")
        except Exception as exc:
            logger.error(f"Fine-tuning failed: {exc}", exc_info=True)
        finally:
            _finetuning_lock.release()

    _finetuning_lock.acquire()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "started"}


# ===========================================================================
# 17. POST /api/push-huggingface
# ===========================================================================
@app.post("/api/push-huggingface")
async def push_to_hf(req: PushHFRequest):
    """Push the exported dataset to HuggingFace Hub."""
    cfg = _ensure_config()

    dataset_dir = cfg.export_dir
    if not os.path.isdir(dataset_dir):
        raise HTTPException(
            status_code=400,
            detail="No exported dataset found. Run /api/export-dataset first.",
        )

    success = push_to_huggingface(
        local_dataset_dir=dataset_dir,
        hf_repo_id=req.repo_id,
        hf_token=req.token,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to push to HuggingFace.")

    url = f"https://huggingface.co/datasets/{req.repo_id}"
    return {"success": True, "url": url}
