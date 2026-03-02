import argparse
from pathlib import Path

import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from src.preprocess import preprocess_page
from src.layout_tesseract import find_eq_line_bboxes

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def save_overlay(pil_img: Image.Image, candidates, out_path: Path):
    bgr = pil_to_bgr(pil_img)
    for c in candidates:
        x0, y0, x1, y1 = c["bbox_px"]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        txt = c.get("line_text_ocr", "")[:40]
        cv2.putText(bgr, txt, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)

def tesseract_df(img: Image.Image, lang: str, psm: int) -> pd.DataFrame:
    config = f"--psm {psm}"
    df = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    if "conf" in df.columns:
        df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1).astype(float)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="docs/Пособие по КВ-5-10.pdf")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--lang", default="rus+eng")
    ap.add_argument("--tesseract-cmd", default=None)
    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    pdf_path = Path(args.pdf)
    doc = fitz.open(pdf_path)
    page = doc.load_page(args.page - 1)

    pp = preprocess_page(page, dpi=args.dpi)

    out_dir = Path("debug")
    out_dir.mkdir(exist_ok=True)
    pp.page_img.save(out_dir / "page.png")
    pp.page_img_pre.save(out_dir / "page_pre.png")

    print("=== Stage1: page type ===")
    print("text_char_count:", pp.text_char_count)
    print("page_type:", pp.page_type)
    print("saved:", (out_dir / "page.png"), (out_dir / "page_pre.png"))

    for psm in (6, 11):
        print(f"\n=== Stage2: tesseract words (psm={psm}) ===")
        df = tesseract_df(pp.page_img_pre, lang=args.lang, psm=psm)
        tokens = df["text"].tolist()
        has_eq = any("=" in t for t in tokens)
        print("tokens:", len(tokens), "has '=' token?", has_eq)
        # покажем кусок токенов
        print("sample tokens:", [t for t in tokens[:50]])

        # попробуем найти строки с '=' по вашему алгоритму
        cands = find_eq_line_bboxes(df, min_mean_conf=0.0)  # для дебага не фильтруем по conf
        print("eq-line candidates:", len(cands))

        overlay_path = out_dir / f"overlay_psm{psm}.png"
        save_overlay(pp.page_img, cands, overlay_path)
        print("overlay saved:", overlay_path)

    doc.close()
    print("\nDone. Open debug/page.png, debug/page_pre.png, debug/overlay_psm6.png, debug/overlay_psm11.png")

if __name__ == "__main__":
    main()
