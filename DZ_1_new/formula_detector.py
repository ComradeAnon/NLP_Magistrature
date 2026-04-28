"""
Классификатор математических формул по главному оператору в LaTeX коде.
Отбрасывает мусор.
"""

import re
import logging
from typing import List, Dict, Optional
from config import CLASS_KEYS

logger = logging.getLogger(__name__)

class FormulaClassifier:
    def classify(self, latex: str) -> Optional[str]:
        if not latex or len(latex.strip()) < 3:
            return None
            
        has_int  = bool(re.search(r'\\int\b|\\oint\b', latex))
        has_sum  = bool(re.search(r'\\sum\b|\\prod\b', latex))
        has_diff = bool(re.search(r'\\partial\b|\\nabla\b|\\lim\b|\\mathrm\{d\}', latex))
        
        ops_count = sum([has_int, has_sum, has_diff])
        
        if ops_count > 1:
            return "mixed"
            
        if has_int:
            return "integral"
        if has_sum:
            return "summation"
        if has_diff:
            return "derivative_limit"
            
        return "equation"

    def classify_batch(self, candidates: List[Dict]) -> List[Dict]:
        results = []
        for item in candidates:
            cls = self.classify(item.get("latex", ""))
            if cls:
                item_copy = dict(item)
                item_copy["class"] = cls
                results.append(item_copy)
            else:
                print(f"  [Отбраковано] Слишком короткий/плохой LaTeX: {item.get('latex', '')}")
                
        return results
