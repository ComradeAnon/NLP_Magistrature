from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import pytesseract
from tqdm import tqdm

from .schema import DocumentMeta, PageInfo
from .utils import ensure_dir, sha1_file, read_json, write_json, append_jsonl
from .preprocess import preprocess_page, extract_doc_metadata
from .layout_tesseract import tesseract_to_df, find_math_line_candidates, refine_formula_bbox_from_line
from .extract_text import extract_latex_from_text_layer
from .extract_formulas import formula_from_text_layer, formula_from_ocr


def load_pix2tex():
    from pix2tex.cli import LatexOCR
    return LatexOCR()

def is_math_latex(latex_norm: str) -> bool:
    """
    Фильтр, чтобы не сохранять мусорные кропы.
    Должно быть не пусто и содержать признаки математики.
    """
    if not latex_norm:
        return False
    s = latex_norm

    # если contains typical math operators/symbols
    math_markers = [
        "=", r"\le", r"\ge", "<", ">", r"\neq", r"\approx",
        r"\times", "+", "-", "^", "_",
        r"\frac", r"\sqrt", r"\sum", r"\int",
        "(", ")", "[", "]",
    ]
    if any(m in s for m in math_markers):
        return True

    # или хотя бы цифры (иногда короткие)
    if any(ch.isdigit() for ch in s):
        return True

    return False


def process_pdf(pdf_path: Path,
                out_dir: Path,
                formulas_jsonl: Path,
                ocr_model,
                dpi: int,
                lang: str,
                tesseract_text_only: bool,
                min_mean_conf: float,
                max_pages: Optional[int],
                save_page_png: bool,
                psm: int) -> Dict:

    doc_id = pdf_path.stem
    pdf_hash = sha1_file(pdf_path)

    doc_out = ensure_dir(out_dir / "docs" / doc_id)
    crops_out = ensure_dir(out_dir / "crops" / doc_id)

    doc = fitz.open(pdf_path)
    meta = DocumentMeta(
        doc_id=doc_id,
        pdf_file=pdf_path.name,
        pdf_sha1=pdf_hash,
        page_count=doc.page_count,
        metadata=extract_doc_metadata(doc),
    )
    write_json(doc_out / "meta.json", meta.__dict__)

    page_infos: List[Dict] = []
    per_doc_formulas: List[Dict] = []

    page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    pages_iter = tqdm(range(page_count), desc=f"{pdf_path.name}", unit="page", leave=False)

    for page_i in pages_iter:
        page = doc.load_page(page_i)
        pp = preprocess_page(page, dpi=dpi)

        page_info = PageInfo(page=page_i + 1, page_type=pp.page_type, text_char_count=pp.text_char_count)
        page_infos.append(page_info.__dict__)

        if save_page_png:
            pp.page_img.save(doc_out / f"page_{page_i+1:04d}.png")

        # Stage 3a: LaTeX from text layer (если реально присутствует как сырой LaTeX)
        latex_hits = extract_latex_from_text_layer(page)
        for hit in latex_hits:
            fr = formula_from_text_layer(doc_id, pdf_path.name, page_i + 1, hit["latex_raw"])
            per_doc_formulas.append(fr.to_dict())
            append_jsonl(formulas_jsonl, fr.to_dict())

        # Stage 2: Tesseract layout candidates
        # По умолчанию tesseract запускается НА ВСЕХ страницах.
        # Если включён tesseract_text_only=True -> запускаем только для scanned/mixed.
        run_tess = True
        if tesseract_text_only:
            run_tess = (pp.page_type in ("scanned", "mixed"))
        if not run_tess:
            continue

        df = tesseract_to_df(pp.page_img_pre, lang=lang, psm=psm)
        line_candidates = find_math_line_candidates(df, min_mean_conf=min_mean_conf)

        pages_iter.set_postfix({
            "type": pp.page_type,
            "math_lines": len(line_candidates),
            "latex_hits": len(latex_hits)
        })

        # Stage 3b: Extract by OCR pix2tex on refined bbox
        for idx, ln in enumerate(line_candidates):
            # refine bbox to formula-ish part
            bbox = refine_formula_bbox_from_line(ln["words"], margin=8, neighbor_words=3)

            # crop from ORIGINAL rendered page (not preprocessed)
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(pp.page_img.width, x1); y1 = min(pp.page_img.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = pp.page_img.crop((x0, y0, x1, y1))

            # OCR formula
            try:
                latex_raw = ocr_model(crop)
            except Exception:
                latex_raw = ""

            latex_raw = (latex_raw or "").strip()

            # если вообще не распознано — не сохраняем мусор
            # и не пишем кроп (по вашему пункту 2)
            from .postprocess import normalize_latex
            latex_norm = normalize_latex(latex_raw)
            if not is_math_latex(latex_norm):
                continue

            # сохраняем кроп только для распознанной формулы
            crop_name = f"p{page_i+1:04d}_{idx:04d}.png"
            crop_path = crops_out / crop_name
            crop.save(crop_path)
            crop_rel = str(Path("crops") / doc_id / crop_name)

            fr = formula_from_ocr(
                doc_id=doc_id,
                pdf_file=pdf_path.name,
                page=page_i + 1,
                page_img=pp.page_img,
                bbox_px=[x0, y0, x1, y1],
                crop_rel_path=crop_rel,
                line_text_ocr=ln.get("line_text_ocr"),
                tesseract_conf=ln.get("mean_conf"),
                latex_raw=latex_raw,
                ocr_confidence=None,
            )
            per_doc_formulas.append(fr.to_dict())
            append_jsonl(formulas_jsonl, fr.to_dict())

    doc.close()

    doc_json = {
        "document": meta.__dict__,
        "pages": page_infos,
        "formulas": per_doc_formulas,
    }
    write_json(doc_out / "result.json", doc_json)

    counts = {}
    for fr in per_doc_formulas:
        counts[fr["cls"]] = counts.get(fr["cls"], 0) + 1

    return {
        "doc_id": doc_id,
        "pdf_file": pdf_path.name,
        "pdf_sha1": pdf_hash,
        "pages_processed": page_count,
        "formula_counts": counts,
        "result_json": str((doc_out / "result.json").resolve()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dpi", type=int, default=350)
    ap.add_argument("--lang", default="rus+eng")
    ap.add_argument("--min-mean-conf", type=float, default=30.0)
    ap.add_argument("--max-pages", type=int, default=None)

    ap.add_argument("--tesseract-cmd", default=None, help="e.g. /opt/homebrew/bin/tesseract")
    ap.add_argument("--tesseract-text-only", action="store_true",
                    help="If set, run Tesseract only on scanned/mixed pages. Default: run on ALL pages.")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract psm (try 6 or 11)")

    ap.add_argument("--save-page-png", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    docs_dir = Path(args.docs)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "docs")
    ensure_dir(out_dir / "crops")

    state_path = out_dir / "state.json"
    formulas_jsonl = out_dir / "formulas.jsonl"
    summary_path = out_dir / "summary.json"

    if args.reset:
        if state_path.exists():
            state_path.unlink()
        if formulas_jsonl.exists():
            formulas_jsonl.unlink()

    state = read_json(state_path, default={"processed": {}})

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF in {docs_dir.resolve()}")
        return

    ocr_model = load_pix2tex()

    run_summary = {
        "docs_dir": str(docs_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "processed": [],
        "skipped": [],
        "started_at": time.time(),
    }

    pdf_iter = tqdm(pdfs, desc="PDFs", unit="pdf")
    for pdf_path in pdf_iter:
        pdf_hash = sha1_file(pdf_path)
        key = pdf_path.name
        if state["processed"].get(key) == pdf_hash:
            run_summary["skipped"].append(key)
            continue

        t0 = time.perf_counter()
        result = process_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir,
            formulas_jsonl=formulas_jsonl,
            ocr_model=ocr_model,
            dpi=args.dpi,
            lang=args.lang,
            tesseract_text_only=args.tesseract_text_only,
            min_mean_conf=args.min_mean_conf,
            max_pages=args.max_pages,
            save_page_png=args.save_page_png,
            psm=args.psm,
        )
        dt = time.perf_counter() - t0
        result["time_sec"] = dt
        run_summary["processed"].append(result)

        state["processed"][key] = pdf_hash
        write_json(state_path, state)

        pdf_iter.set_postfix({"last_sec": f"{dt:.1f}", "doc": pdf_path.name})

    run_summary["finished_at"] = time.time()
    write_json(summary_path, run_summary)

    print(f"Done. formulas: {formulas_jsonl.resolve()}")
    print(f"Summary: {summary_path.resolve()}")

if __name__ == "__main__":
    main()
