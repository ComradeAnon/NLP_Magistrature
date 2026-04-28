import argparse
import fitz
import cv2
import numpy as np
from pathlib import Path
import re

def is_math_text(text: str) -> tuple[bool, str]:
    text = text.strip()
    if not text:
        return False, "Пустой текст"
        
    clean_text = text.replace(" ", "").replace("\n", "")
    total_chars = len(clean_text)
    
    # Отбрасываем номера формул (1), [2], (A1) и т.д.
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

def merge_horizontal_boxes(boxes, max_gap=120):
    """
    Склеивает рамки, которые находятся на одной горизонтальной линии
    и разрыв между которыми не превышает max_gap.
    """
    if not boxes:
        return []

    # Сортируем рамки сверху вниз, а затем слева направо
    boxes.sort(key=lambda b: (b.y0, b.x0))

    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue

        last_box = merged[-1]

        # Проверяем перекрытие по вертикали (находятся ли они на одной строке)
        overlap_y = max(0, min(last_box.y1, box.y1) - max(last_box.y0, box.y0))
        min_height = min(last_box.height, box.height)

        # Если рамки перекрываются хотя бы на 30% по высоте
        if min_height > 0 and overlap_y > 0.3 * min_height:
            # Считаем разрыв между правой гранью первой и левой гранью второй
            gap = box.x0 - last_box.x1

            # Если разрыв небольшой (покрывает выравнивание по '=', но не достает до '(5)')
            if -20 < gap < max_gap:
                # Создаем новую рамку, поглощающую обе
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

def analyze_page_boxes(page, dpi_zoom=3.0):
    matrix = fitz.Matrix(dpi_zoom, dpi_zoom)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
    
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    page_width = pix.w
    raw_boxes = []
    
    # 1. Собираем все первичные рамки от OpenCV
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        pdf_rect = fitz.Rect(x, y, x+w, y+h) / dpi_zoom
        raw_boxes.append(pdf_rect)

    # 2. УМНАЯ СКЛЕЙКА РАЗОРВАННЫХ ФОРМУЛ (max_gap=120 пикселей / dpi_zoom = 40 единиц PDF)
    merged_boxes = merge_horizontal_boxes(raw_boxes, max_gap=40)
    
    results = []
    # 3. Принимаем решение по каждой склеенной рамке
    for pdf_rect in merged_boxes:
        w = pdf_rect.width
        h = pdf_rect.height
        
        # Геометрические фильтры
        if w > (page_width / dpi_zoom) * 0.90:
            results.append({"rect": pdf_rect, "status": "REJECT", "color": (1, 0, 0), "reason": "[REJECT] Блок на всю ширину"})
            continue
        if h < 20 / dpi_zoom or w < 20 / dpi_zoom:
            results.append({"rect": pdf_rect, "status": "REJECT", "color": (1, 0.5, 0), "reason": "[REJECT] Мелкий мусор"})
            continue

        # Читаем текст внутри итоговой рамки
        text_rect = pdf_rect + (-2, -2, 2, 2)
        raw_text = page.get_text("text", clip=text_rect)
        
        is_math, text_reason = is_math_text(raw_text)
        
        if is_math:
            status = "ACCEPTED"
            color = (0, 0.7, 0)
            final_rect = pdf_rect + (-5, -5, 5, 5) 
            reason = f"[ACCEPT] {text_reason}"
        else:
            status = "REJECT"
            color = (1, 0, 0)
            final_rect = pdf_rect
            reason = f"[REJECT] {text_reason}"

        results.append({
            "rect": final_rect,
            "status": status,
            "reason": reason,
            "color": color
        })
        
    return results

def process_debug_pdf(pdf_path, max_pages):
    print(f"🔍 Запуск дебага (с умной склейкой) для файла: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    pages_to_process = min(max_pages, len(doc)) if max_pages > 0 else len(doc)
    stats = {"ACCEPTED": 0, "REJECT": 0}
    
    for i in range(pages_to_process):
        page = doc[i]
        boxes_info = analyze_page_boxes(page)
        
        for info in boxes_info:
            rect = info["rect"]
            color = info["color"]
            reason = info["reason"]
            
            stats[info["status"]] += 1
            
            page.draw_rect(rect, color=color, width=1.5)
            text_y = rect.y0 - 2 if rect.y0 > 10 else rect.y1 + 10
            page.insert_text(fitz.Point(rect.x0, text_y), reason, fontsize=8, color=color)

    out_path = f"DEBUG_VISION_{Path(pdf_path).name}"
    doc.save(out_path)
    
    print("-" * 40)
    print("✅ Дебаг завершен!")
    print(f"   🟩 Формулы: {stats['ACCEPTED']}")
    print(f"   🟥 Текст/Мусор: {stats['REJECT']}")
    print(f"📂 Сохранено в: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--pages", type=int, default=0)
    args = parser.parse_args()
    process_debug_pdf(args.file, args.pages)
