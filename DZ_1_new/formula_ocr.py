"""
Боевой парсер сложных формул.
"""

import warnings
import logging
from pathlib import Path
from typing import List, Dict
import re

import fitz
import cv2
import numpy as np
from PIL import Image

# Глушим системный спам
warnings.filterwarnings("ignore")
logging.getLogger("albumentations").setLevel(logging.ERROR)

class FormulaOCR:
    def __init__(self):
        self.model = None
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
        except ImportError:
            print("❌ pix2tex не установлен. Введи: pip install pix2tex[gui]")

    def _is_math_text(self, text: str) -> tuple[bool, str]:
        text = text.strip()
        if not text:
            return False, "Пустой текст"
            
        clean_text = text.replace(" ", "").replace("\n", "")
        total_chars = len(clean_text)
        
        if re.fullmatch(r"^[\(\[]?[A-Za-z]?\d+([.,\-]\d+)*[\)\]]?[.,]?$", clean_text):
            return False, "Номер формулы или число"
            
        letters = len(re.findall(r'[a-zA-Zа-яА-Я]', text))
        
        if total_chars > 30 and letters / total_chars > 0.7:
            return False, "Сплошной текст (много букв)"
            
        math_symbols = ['∫', '∑', '∂', '∆', '∇', 'lim', 'max', 'min', 'dx', 'dy', '∈', '⊂', '∞', '≈', '≠', '≤', '≥']
        if any(sym in text for sym in math_symbols):
            return True, "Найден спецсимвол вышмата"
            
        if any(sym in text for sym in ['=', '<', '>', '≈']):
            if letters / (total_chars + 1) < 0.6:
                return True, "Найдено уравнение (=)"
                
        operators = set("+-/*^|") 
        op_count = sum(1 for c in clean_text if c in operators)
        
        if total_chars > 5 and (op_count / total_chars) > 0.10:
            return True, "Много математических операторов"
            
        return False, "Нет явных признаков формулы"

    def _merge_horizontal_boxes(self, boxes, max_gap=120):
        if not boxes:
            return []

        boxes.sort(key=lambda b: (b.y0, b.x0))
        merged = []
        for box in boxes:
            if not merged:
                merged.append(box)
                continue

            last_box = merged[-1]
            overlap_y = max(0, min(last_box.y1, box.y1) - max(last_box.y0, box.y0))
            min_height = min(last_box.height, box.height)

            if min_height > 0 and overlap_y > 0.3 * min_height:
                gap = box.x0 - last_box.x1
                if -20 < gap < max_gap:
                    new_rect = fitz.Rect(
                        min(last_box.x0, box.x0),
                        min(last_box.y0, box.y0),
                        max(last_box.x1, box.x1),
                        max(last_box.y1, box.y1)
                    )
                    merged[-1] = new_rect
                    continue

            merged.append(box)
        return merged

    def extract_from_pdf(self, pdf_path: Path) -> List[Dict]:
        if not self.model:
            return []

        results = []
        doc = fitz.open(str(pdf_path))

        for page_num, page in enumerate(doc):
            dpi_zoom = 3.0
            matrix = fitz.Matrix(dpi_zoom, dpi_zoom)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            page_width = pix.w
            raw_boxes = []
            
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                pdf_rect = fitz.Rect(x, y, x+w, y+h) / dpi_zoom
                raw_boxes.append(pdf_rect)

            merged_boxes = self._merge_horizontal_boxes(raw_boxes, max_gap=40)
            
            for pdf_rect in merged_boxes:
                w = pdf_rect.width
                h = pdf_rect.height
                
                if w > (page_width / dpi_zoom) * 0.90:
                    continue
                if h < 20 / dpi_zoom or w < 20 / dpi_zoom:
                    continue

                text_rect = pdf_rect + (-2, -2, 2, 2)
                raw_text = page.get_text("text", clip=text_rect)
                
                is_math, _ = self._is_math_text(raw_text)
                
                if is_math:
                    final_rect = pdf_rect + (-5, -5, 5, 5) 
                    final_rect.intersect(page.rect)
                    
                    try:
                        p = page.get_pixmap(matrix=matrix, clip=final_rect)
                        
                        if p.width == 0 or p.height == 0:
                            continue
                            
                        img_pil = Image.frombytes("RGB", [p.width, p.height], p.samples)
                        latex_code = self.model(img_pil)
                        
                        if latex_code and len(latex_code.strip()) > 2:
                            clean_t = raw_text.replace('\n', ' ').strip()
                            results.append({
                                "text": clean_t,
                                "latex": latex_code,
                                "source": pdf_path.stem
                            })
                    except Exception:
                        pass # Игнорируем ошибки рендера картинки

        doc.close()
        return results
