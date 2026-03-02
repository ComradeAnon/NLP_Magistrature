from __future__ import annotations

from typing import Dict, List, Tuple
import re

import pandas as pd
import pytesseract
from PIL import Image

from .utils import normalize_spaces

# Символы/паттерны, которые считаем "математическими триггерами"
# (подберите при необходимости)
MATH_TRIGGERS = [
    "=", "≠", "≈",
    "<", ">", "≤", "≥",
    "±", "×", "⋅",
    "∑", "∫", "√", "∞",
    "→", "↦", "∝",
]

# Иногда OCR отдаёт странные варианты для равно/минуса
MATH_TRIGGER_REGEX = re.compile(
    r"(" +
    r"|".join(re.escape(t) for t in MATH_TRIGGERS) +
    r"|<=|>=|!=|=="
    r")"
)

def tesseract_to_df(img: Image.Image, lang: str = "rus+eng", psm: int = 6) -> pd.DataFrame:
    """
    psm=6: Assume a single uniform block of text.
    Можно пробовать 6/11, но 6 обычно стабильно для строк.
    """
    config = f"--psm {psm}"
    df = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    if "conf" in df.columns:
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(float)
    return df

def _group_line_words(df: pd.DataFrame) -> List[Dict]:
    """
    Группируем слова по строкам. Возвращаем список line-структур:
    {
      "bbox_px": [x0,y0,x1,y1],
      "line_text_ocr": "...",
      "mean_conf": ...,
      "words": [ {text,left,top,width,height,conf}, ... ],
      "block_num","par_num","line_num"
    }
    """
    if df.empty:
        return []

    out = []
    grp_cols = ["block_num", "par_num", "line_num"]

    for key, g in df.groupby(grp_cols):
        g = g.copy()
        g["text"] = g["text"].astype(str)

        words = []
        for _, r in g.iterrows():
            txt = str(r["text"]).strip()
            if not txt:
                continue
            words.append({
                "text": txt,
                "left": int(r["left"]),
                "top": int(r["top"]),
                "width": int(r["width"]),
                "height": int(r["height"]),
                "conf": float(r["conf"]) if "conf" in r else None
            })

        if not words:
            continue

        # bbox всей строки
        x0 = min(w["left"] for w in words)
        y0 = min(w["top"] for w in words)
        x1 = max(w["left"] + w["width"] for w in words)
        y1 = max(w["top"] + w["height"] for w in words)

        line_text = normalize_spaces(" ".join(w["text"] for w in words))

        mean_conf = None
        if "conf" in g.columns:
            mean_conf = float(g["conf"].replace(-1, 0).mean())

        out.append({
            "bbox_px": [x0, y0, x1, y1],
            "line_text_ocr": line_text,
            "mean_conf": mean_conf,
            "words": words,
            "block_num": int(key[0]),
            "par_num": int(key[1]),
            "line_num": int(key[2]),
        })

    return out

def find_math_line_candidates(df: pd.DataFrame, min_mean_conf: float = 30.0) -> List[Dict]:
    """
    Возвращает строки, где есть мат.триггеры (не только '=').
    """
    lines = _group_line_words(df)
    out = []
    for ln in lines:
        if ln["mean_conf"] is not None and ln["mean_conf"] < min_mean_conf:
            continue
        # проверка по словам (так надёжнее)
        has_math = any(MATH_TRIGGER_REGEX.search(w["text"]) for w in ln["words"])
        if not has_math:
            # иногда в line_text символы сохраняются лучше, чем по словам
            if not MATH_TRIGGER_REGEX.search(ln["line_text_ocr"]):
                continue
        out.append(ln)
    return out

def refine_formula_bbox_from_line(words: List[Dict], margin: int = 8, neighbor_words: int = 3) -> List[int]:
    """
    Сужаем bbox строки до "формульной части":
    - ищем слова, содержащие триггеры
    - берём их + несколько соседей слева/справа (по порядку X)
    """
    if not words:
        return [0, 0, 0, 0]

    # сортируем слева направо
    ws = sorted(words, key=lambda w: w["left"])

    anchor_idx = [i for i, w in enumerate(ws) if MATH_TRIGGER_REGEX.search(w["text"])]
    if not anchor_idx:
        # fallback: вся строка
        x0 = min(w["left"] for w in ws)
        y0 = min(w["top"] for w in ws)
        x1 = max(w["left"] + w["width"] for w in ws)
        y1 = max(w["top"] + w["height"] for w in ws)
        return [x0, y0, x1, y1]

    i0 = max(0, min(anchor_idx) - neighbor_words)
    i1 = min(len(ws) - 1, max(anchor_idx) + neighbor_words)

    chosen = ws[i0:i1 + 1]
    x0 = min(w["left"] for w in chosen) - margin
    y0 = min(w["top"] for w in chosen) - margin
    x1 = max(w["left"] + w["width"] for w in chosen) + margin
    y1 = max(w["top"] + w["height"] for w in chosen) + margin

    # не даём отрицательных
    x0 = max(0, x0)
    y0 = max(0, y0)
    return [x0, y0, x1, y1]
