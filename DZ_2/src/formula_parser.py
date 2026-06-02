"""
Парсер LaTeX формул.
Концепция:
  - Вершина графа = атомарный математический элемент
    (переменная, константа, функция, оператор типа ∫/lim/∂)
  - Ребро графа = операция между элементами (+, -, =, <, ∂, ∫, ^ ...)
  - Каждое ребро помечено ID формулы из которой оно взято
  - Одинаковые элементы (x, y, f) переиспользуются между формулами
"""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Структуры данных
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MathNode:
    """Вершина графа — атомарный математический элемент."""
    id: str
    label: str       # читаемое название: x, f, ∫, lim, e, π ...
    node_type: str   # variable | constant | function | operator | number
    latex: str
    description: str = ""

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, MathNode) and self.id == other.id

@dataclass
class MathEdge:
    """Ребро графа — операция между двумя элементами."""
    source_id: str
    target_id: str
    operation: str   # addition, subtraction, equals, less_than, power, ...
    formula_id: str  # из какой формулы это ребро
    description: str = ""
    weight: float = 1.0


@dataclass
class FormulaGraph:
    """Граф одной формулы."""
    formula_id: str
    latex: str
    nodes: list[MathNode] = field(default_factory=list)
    edges: list[MathEdge] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# Словари символов
# ═════════════════════════════════════════════════════════════════════════════

GREEK = {
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
    r'\theta': 'θ', r'\vartheta': 'θ', r'\kappa': 'κ', r'\lambda': 'λ',
    r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\varpi': 'π',
    r'\rho': 'ρ', r'\varrho': 'ρ', r'\sigma': 'σ', r'\varsigma': 'σ',
    r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ', r'\varphi': 'φ',
    r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
    r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
    r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
    r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
}

# label, node_type
CONSTANTS: dict[str, tuple[str, str]] = {
    r'\infty':    ('∞',  'constant'),
    r'\emptyset': ('∅',  'constant'),
    'ℝ':          ('ℝ',  'constant'),
    'ℕ':          ('ℕ',  'constant'),
    'ℤ':          ('ℤ',  'constant'),
    'ℂ':          ('ℂ',  'constant'),
}

# label, node_type
OPERATORS: dict[str, tuple[str, str]] = {
    r'\int':     ('∫',   'operator'),
    r'\iint':    ('∬',   'operator'),
    r'\iiint':   ('∭',   'operator'),
    r'\oint':    ('∮',   'operator'),
    r'\sum':     ('Σ',   'operator'),
    r'\prod':    ('Π',   'operator'),
    r'\lim':     ('lim', 'operator'),
    r'\partial': ('∂',   'operator'),
    r'\nabla':   ('∇',   'operator'),
    r'\sqrt':    ('√',   'operator'),
    r'\sup':     ('sup', 'operator'),
    r'\inf':     ('inf', 'operator'),
    r'\max':     ('max', 'operator'),
    r'\min':     ('min', 'operator'),
    r'\det':     ('det', 'operator'),
    r'\exp':     ('exp', 'function'),
    r'\log':     ('log', 'function'),
    r'\ln':      ('ln',  'function'),
    r'\sin':     ('sin', 'function'),
    r'\cos':     ('cos', 'function'),
    r'\tan':     ('tan', 'function'),
    r'\cot':     ('cot', 'function'),
    r'\arcsin':  ('arcsin', 'function'),
    r'\arccos':  ('arccos', 'function'),
    r'\arctan':  ('arctan', 'function'),
    r'\sinh':    ('sinh', 'function'),
    r'\cosh':    ('cosh', 'function'),
    r'\tanh':    ('tanh', 'function'),
}

# операция → название ребра
OP_SYMBOLS: dict[str, str] = {
    '+':             'addition',
    '-':             'subtraction',
    '*':             'multiplication',
    r'\cdot':        'multiplication',
    r'\times':       'multiplication',
    r'\otimes':      'tensor_product',
    r'\oplus':       'direct_sum',
    '/':             'division',
    '=':             'equals',
    '<':             'less_than',
    '>':             'greater_than',
    r'\leq':         'less_or_equal',
    r'\geq':         'greater_or_equal',
    r'\neq':         'not_equal',
    r'\approx':      'approx_equal',
    r'\sim':         'similar',
    r'\simeq':       'similar',
    r'\equiv':       'equivalent',
    r'\in':          'belongs_to',
    r'\notin':       'not_belongs_to',
    r'\subset':      'subset',
    r'\subseteq':    'subset_or_equal',
    r'\cup':         'union',
    r'\cap':         'intersection',
    r'\to':          'maps_to',
    r'\rightarrow':  'maps_to',
    r'\leftarrow':   'maps_from',
    r'\Rightarrow':  'implies',
    r'\Leftarrow':   'implied_by',
    r'\leftrightarrow': 'iff',
    r'\Leftrightarrow': 'iff',
    '^':             'power',
    '_':             'subscript',
}

# Операции-отношения — разбивают формулу на части
RELATION_OPS = {
    '=', '<', '>',
    r'\leq', r'\geq', r'\neq',
    r'\approx', r'\sim', r'\simeq', r'\equiv',
    r'\in', r'\notin', r'\subset', r'\subseteq',
    r'\to', r'\rightarrow', r'\Rightarrow',
    r'\leftrightarrow', r'\Leftrightarrow',
}

# LaTeX команды которые нужно пропустить (декоративные)
SKIP_COMMANDS = {
    r'\left', r'\right', r'\quad', r'\qquad',
    r'\,', r'\;', r'\!', r'\:', r'\ ',
    r'\mathrm', r'\mathbf', r'\mathit', r'\mathsf',
    r'\bf', r'\rm', r'\it', r'\sf',
    r'\text', r'\mbox',
    r'\bigl', r'\bigr', r'\Bigl', r'\Bigr',
    r'\big', r'\Big', r'\bigg', r'\Bigg',
    r'\displaystyle', r'\textstyle', r'\scriptstyle',
    r'\limits', r'\nolimits',
    r'\nonumber', r'\notag',
    r'\label', r'\tag',
}

# ═════════════════════════════════════════════════════════════════════════════
# Парсер
# ═════════════════════════════════════════════════════════════════════════════
class LaTeXParser:
    """
    Парсер LaTeX → граф математических элементов.

    Алгоритм:
      1. Предобработка: разворачиваем \\frac, убираем декоративные команды
      2. Токенизация: разбиваем на атомарные токены
      3. Построение графа: токены → вершины + рёбра
    """

    def __init__(self):
        self._counter = 0
        # Реестр узлов: "type:label" → MathNode
        # Позволяет переиспользовать x, y, f из разных формул
        self._node_registry: dict[str, MathNode] = {}

    # ─── Публичный интерфейс ─────────────────────────────────────────────────
    def parse(
        self,
        latex: str,
        formula_id: str,
    ) -> FormulaGraph:
        """Парсит LaTeX строку в FormulaGraph, используя только поле latex."""
        graph = FormulaGraph(
            formula_id=formula_id,
            latex=latex,
        )

        try:
            clean  = self._preprocess(latex)
            tokens = self._tokenize(clean)
            self._build_graph(tokens, graph)
        except Exception as e:
            logger.warning(f"Ошибка парсинга {formula_id}: {e}")
            # Fallback — хотя бы один узел
            node = self._get_or_create_node(
                label=latex[:30] or "formula",
                node_type="expression",
                latex=latex,
                description=f"Формула {formula_id}",
            )
            if node not in graph.nodes:
                graph.nodes.append(node)

        return graph

    # ─── Предобработка ───────────────────────────────────────────────────────
    def _preprocess(self, latex: str) -> str:
        """
        Приводит LaTeX к виду удобному для токенизации:
          \\frac{a}{b}        →  (a) / (b)
          \\mathbb{R}         →  ℝ
          \\mathcal{L}        →  L
          \\operatorname{inf} →  \\inf
          \\left( ... \\right) →  убираем \\left \\right
        """
        # 1. Разворачиваем \frac
        latex = self._expand_frac(latex)

        # 2. \mathbb{X} → ℝ/ℕ/ℤ/ℂ или просто X
        def replace_mathbb(m):
            letter = m.group(1)
            mapping = {'R': 'ℝ', 'N': 'ℕ', 'Z': 'ℤ', 'C': 'ℂ', 'Q': 'ℚ'}
            return mapping.get(letter, letter)
        latex = re.sub(r'\\mathbb\{([^}]+)\}', replace_mathbb, latex)

        # 3. \mathcal{X} → X (просто убираем оформление)
        latex = re.sub(r'\\math(?:cal|scr|frak)\{([^}]+)\}', r'\1', latex)

        # 4. \operatorname{xxx} → \xxx
        latex = re.sub(r'\\operatorname\*?\{([^}]+)\}', r'\\\1', latex)
        # 5. \mathrm{xxx}, \mathbf{xxx} → xxx
        latex = re.sub(r'\\math(?:rm|bf|it|sf)\{([^}]+)\}', r'\1', latex)


        # 6. \text{xxx} → убираем
        latex = re.sub(r'\\text\{[^}]*\}', '', latex)

        # 7. Убираем \left \right (скобки оставляем)
        latex = re.sub(r'\\(?:left|right)\s*', '', latex)

        # 8. Убираем \begin{...}...\end{...} блоки (матрицы, системы)
        latex = re.sub(r'\\begin\{[^}]+\}', '', latex)
        latex = re.sub(r'\\end\{[^}]+\}', '', latex)

        # 9. \quad, \qquad, \, и т.п. → пробел
        latex = re.sub(
            r'\\(?:quad|qquad|,|;|!|:|thinspace|medspace|thickspace)', ' ', latex
        )
        # 10. Убираем \\ (перенос строки в LaTeX)
        latex = latex.replace('\\\\', ' ')
        return latex

    def _expand_frac(self, latex: str) -> str:
        """Разворачивает \\frac{числитель}{знаменатель} → (числитель)/(знаменатель)."""
        for _ in range(10):  # до 10 вложенных дробей
            match = re.search(r'\\frac\s*\{', latex)
            if not match:
                break
            pos = match.start()
            brace_pos = match.end() - 1  # позиция открывающей {

            # Числитель
            num_content, num_len = self._extract_braces(latex, brace_pos)
            after_num = brace_pos + num_len

            # Пропускаем пробелы между {} {}
            j = after_num
            while j < len(latex) and latex[j] in ' \t\n':
                j += 1

            if j >= len(latex) or latex[j] != '{':
                break

            # Знаменатель
            den_content, den_len = self._extract_braces(latex, j)
            after_den = j + den_len
            replacement = f"({num_content}) / ({den_content})"
            latex = latex[:pos] + replacement + latex[after_den:]
        return latex
    def _extract_braces(self, latex: str, pos: int) -> tuple[str, int]:
        """
        Извлекает содержимое {…} начиная с позиции pos (pos указывает на '{').
        Возвращает (содержимое, длина_включая_скобки).
        """
        if pos >= len(latex) or latex[pos] != '{':
            return '', 0
        depth = 1
        i = pos + 1
        while i < len(latex) and depth > 0:
            if latex[i] == '{':
                depth += 1
            elif latex[i] == '}':
                depth -= 1
            i += 1
        return latex[pos + 1: i - 1], i - pos


    # ─── Токенизация ─────────────────────────────────────────────────────────
    def _tokenize(self, latex: str) -> list[dict]:
        """
        Разбивает LaTeX на токены:
          {'type': str, 'value': str, 'latex': str}

        Типы токенов:
          var      — одиночная буква-переменная (x, y, f, ...)
          greek    — греческая буква (α, β, ...)
          num      — число (1, 2.5, ...)
          constant — математическая константа (∞, ℝ, ...)
          operator — математический оператор (∫, lim, ∂, ...)
          function — математическая функция (sin, cos, ...)
          sym      — символ операции (+, -, =, <, ^, _, ...)
        """
        tokens: list[dict] = []
        i = 0

        while i < len(latex):
            ch = latex[i]

            # ── Пробелы ───────────────────────────────────────────────────────
            if ch in ' \t\n\r':
                i += 1
                continue

            # ── Фигурные скобки — рекурсивно токенизируем содержимое ─────────
            if ch == '{':
                content, length = self._extract_braces(latex, i)
                inner = self._tokenize(content)
                tokens.extend(inner)
                i += length
                continue
            if ch == '}':
                i += 1
                continue

            # ── Круглые и квадратные скобки — прозрачны ──────────────────────
            if ch in '()[]':
                i += 1
                continue

            # ── Специальные одиночные символы ─────────────────────────────────
            if ch == '&':   # разделитель колонок в матрицах
                i += 1
                continue
            if ch == '#':
                i += 1
                continue

            # ── Десятичная точка/запятая внутри числа — обрабатываем с числом
            if ch == '.' and i + 1 < len(latex) and latex[i+1].isdigit():
                i += 1
                continue

            # ── LaTeX команды: \something ─────────────────────────────────────
            if ch == '\\':
                tok, length = self._parse_command(latex, i)
                if tok is not None:
                    tokens.append(tok)
                i += length
                continue

            # ── Числа ─────────────────────────────────────────────────────────
            num_match = re.match(r'\d+(?:\.\d+)?', latex[i:])
            if num_match:
                tokens.append({
                    'type':  'num',
                    'value': num_match.group(),
                    'latex': num_match.group(),
                })
                i += len(num_match.group())
                continue

            # ── Многосимвольные операторы ─────────────────────────────────────
            multi_sym = None
            for sym in ('<=', '>=', '!=', '->', '<-', '<=>'):
                if latex[i:i+len(sym)] == sym:
                    multi_sym = sym
                    break
            if multi_sym:
                tokens.append({
                    'type':  'sym',
                    'value': multi_sym,
                    'latex': multi_sym,
                })
                i += len(multi_sym)
                continue

            # ── Одиночные символы операций ────────────────────────────────────
            if ch in '+=<>|^_,':
                tokens.append({
                    'type':  'sym',
                    'value': ch,
                    'latex': ch,
                })
                i += 1
                continue

            if ch == '-':
                tokens.append({
                    'type':  'sym',
                    'value': '-',
                    'latex': '-',
                })
                i += 1
                continue

            if ch == '/':
                tokens.append({
                    'type':  'sym',
                    'value': '/',
                    'latex': '/',
                })
                i += 1
                continue

            if ch == '=':
                tokens.append({
                    'type':  'sym',
                    'value': '=',
                    'latex': '=',
                })
                i += 1
                continue

            # ── Специальные Unicode символы (после предобработки) ─────────────
            unicode_match = re.match(r'[ℝℕℤℂℚ∞∅]', latex[i:])
            if unicode_match:
                sym = unicode_match.group()
                label, ntype = CONSTANTS.get(sym, (sym, 'constant'))
                tokens.append({
                    'type':  ntype,
                    'value': label,
                    'latex': sym,
                })
                i += len(sym)
                continue

            # ── Одиночные буквы — переменные ──────────────────────────────────
            if ch.isalpha():
                tokens.append({
                    'type':  'var',
                    'value': ch,
                    'latex': ch,
                })
                i += 1
                continue

            # Всё остальное пропускаем
            i += 1

        return tokens

    def _parse_command(self, latex: str, pos: int) -> tuple[dict | None, int]:
        """
        Парсит LaTeX команду начиная с pos (latex[pos] == '\\').
        Возвращает (токен | None, сколько символов занимает команда).
        """
        if pos + 1 >= len(latex):
            return None, 1

        next_ch = latex[pos + 1]

        # \\ — перенос строки
        if next_ch == '\\':
            return None, 2

        # \{ \} \[ \] — скобки
        if next_ch in '{}[]()| ':
            return None, 2

        # \число — пропускаем
        if next_ch.isdigit():
            return None, 2

        # \commandName
        cmd_match = re.match(r'\\([a-zA-Z]+\*?)', latex[pos:])
        if not cmd_match:
            return None, 1

        cmd     = '\\' + cmd_match.group(1)
        cmd_len = len(cmd_match.group())

        # Пропускаем декоративные команды
        if cmd in SKIP_COMMANDS:
            rest = pos + cmd_len
            while rest < len(latex) and latex[rest] == ' ':
                rest += 1
            if rest < len(latex) and latex[rest] == '{':
                _, blen = self._extract_braces(latex, rest)
                return None, rest - pos + blen
            return None, cmd_len

        # Греческие буквы
        if cmd in GREEK:
            return {
                'type':  'var',
                'value': GREEK[cmd],
                'latex': cmd,
            }, cmd_len

        # Операторы (∫, lim, ∂, ...)
        if cmd in OPERATORS:
            label, ntype = OPERATORS[cmd]
            return {
                'type':  ntype,
                'value': label,
                'latex': cmd,
            }, cmd_len

        # Константы (\infty, \emptyset)
        if cmd in CONSTANTS:
            label, ntype = CONSTANTS[cmd]
            return {
                'type':  ntype,
                'value': label,
                'latex': cmd,
            }, cmd_len

        # Символы операций (\leq, \geq, \cdot, \to, ...)
        if cmd in OP_SYMBOLS:
            return {
                'type':  'sym',
                'value': cmd,
                'latex': cmd,
            }, cmd_len

        # \array, \begin, \end и прочие блочные команды — пропускаем с аргументом
        if cmd_match.group(1) in (
            'array', 'begin', 'end', 'hline',
            'displaystyle', 'label', 'tag',
            'nonumber', 'notag', 'cr',
        ):
            rest = pos + cmd_len
            while rest < len(latex) and latex[rest] == ' ':
                rest += 1
            if rest < len(latex) and latex[rest] == '{':
                _, blen = self._extract_braces(latex, rest)
                return None, rest - pos + blen
            return None, cmd_len

        # Неизвестная команда — пропускаем
        return None, cmd_len

    # ─── Построение графа ─────────────────────────────────────────────────────

    def _build_graph(self, tokens: list[dict], graph: FormulaGraph):
        """
        Строит граф из списка токенов.

        Стратегия:
          1. Разбиваем токены на сегменты по операциям-отношениям (=, <, →)
          2. Каждый сегмент обрабатываем отдельно → получаем главный узел
          3. Главные узлы соседних сегментов связываем ребром-отношением
        """
        fid      = graph.formula_id
        segments = self._split_by_relations(tokens)

        if len(segments) == 1:
            self._process_segment(segments[0][0], graph, fid)
            return

        prev_node: MathNode | None = None

        for seg_tokens, left_op in segments:
            seg_node = self._process_segment(seg_tokens, graph, fid)

            if prev_node is not None and seg_node is not None and left_op:
                op_name = OP_SYMBOLS.get(left_op, left_op)
                self._add_edge(
                    graph, fid,
                    prev_node, seg_node,
                    operation=op_name,
                    weight=2.0,
                )

            if seg_node is not None:
                prev_node = seg_node

    def _split_by_relations(
        self,
        tokens: list[dict],
    ) -> list[tuple[list[dict], str | None]]:
        """
        Делит токены на сегменты по операциям-отношениям (=, <, >, →, ...).
        Возвращает список (токены_сегмента, разделитель_слева_от_сегмента).
        Первый сегмент имеет разделитель None.
        """
        segments: list[tuple[list[dict], str | None]] = []
        current:  list[dict] = []
        left_op:  str | None = None

        for tok in tokens:
            if tok['type'] == 'sym' and tok['value'] in RELATION_OPS:
                segments.append((current, left_op))
                current = []
                left_op = tok['value']
            else:
                current.append(tok)

        segments.append((current, left_op))
        return segments

    def _process_segment(
        self,
        tokens: list[dict],
        graph: FormulaGraph,
        formula_id: str,
    ) -> MathNode | None:
        """
        Обрабатывает один сегмент выражения (между знаками отношения).
        Строит внутренние рёбра (+, -, *, /, ^, ∫, ∂, lim ...).
        Возвращает главный узел сегмента.
        """
        if not tokens:
            return None

        # Помечаем переменные-функции: f x → f(x)
        tokens = self._mark_function_calls(tokens)

        elements:     list[MathNode] = []
        operations:   list[str]      = []
        last_op_node: MathNode | None = None
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            # ── Операторы (∫, ∂, lim, √, Σ, ...) ────────────────────────────
            if tok['type'] == 'operator':
                op_node = self._get_or_create_node(
                    label=tok['value'],
                    node_type='operator',
                    latex=tok['latex'],
                    description=self._op_description(tok['value']),
                )
                self._ensure_in_graph(op_node, graph)
                last_op_node = op_node

                # Нижний индекс: lim_{n→∞}, \int_{0}
                if (i + 1 < len(tokens)
                        and tokens[i+1]['type'] == 'sym'
                        and tokens[i+1]['value'] == '_'):
                    i += 2
                    if i < len(tokens) and tokens[i]['type'] != 'sym':
                        param = self._token_to_node(tokens[i], graph)
                        if param:
                            self._add_edge(
                                graph, formula_id,
                                op_node, param,
                                operation='parameter',
                            )

                # Верхний индекс: \int^{b}
                elif (i + 1 < len(tokens)
                        and tokens[i+1]['type'] == 'sym'
                        and tokens[i+1]['value'] == '^'):
                    i += 2
                    if i < len(tokens) and tokens[i]['type'] != 'sym':
                        bound = self._token_to_node(tokens[i], graph)
                        if bound:
                            self._add_edge(
                                graph, formula_id,
                                op_node, bound,
                                operation='upper_bound',
                            )

                # Аргумент оператора — следующий не-sym токен
                if (i + 1 < len(tokens)
                        and tokens[i+1]['type'] not in ('sym',)):
                    i += 1
                    arg = self._token_to_node(tokens[i], graph)
                    if arg:
                        self._add_edge(
                            graph, formula_id,
                            op_node, arg,
                            operation='applied_to',
                        )

                elements.append(op_node)

            # ── Функции (sin, cos, exp, ...) ──────────────────────────────────
            elif tok['type'] == 'function':
                fn_node = self._get_or_create_node(
                    label=tok['value'],
                    node_type='function',
                    latex=tok['latex'],
                    description=f"Функция {tok['value']}",
                )
                self._ensure_in_graph(fn_node, graph)

                if (i + 1 < len(tokens)
                        and tokens[i+1]['type'] not in ('sym',)):
                    i += 1
                    arg = self._token_to_node(tokens[i], graph)
                    if arg:
                        self._add_edge(
                            graph, formula_id,
                            fn_node, arg,
                            operation='function_of',
                        )

                elements.append(fn_node)

            # ── Переменная-функция: f(x) помечена как var_func ───────────────
            elif tok['type'] == 'var_func':
                fn_node = self._get_or_create_node(
                    label=tok['value'],
                    node_type='function',
                    latex=tok['latex'],
                    description=f"Функция {tok['value']}",
                )
                self._ensure_in_graph(fn_node, graph)

                if (i + 1 < len(tokens)
                        and tokens[i+1]['type'] in ('var', 'num', 'constant')):
                    i += 1
                    arg = self._token_to_node(tokens[i], graph)
                    if arg:
                        self._add_edge(
                            graph, formula_id,
                            fn_node, arg,
                            operation='function_of',
                        )

                elements.append(fn_node)

            # ── Степень и индекс: ^ _ ─────────────────────────────────────────
            elif tok['type'] == 'sym' and tok['value'] in ('^', '_'):
                op_name = 'power' if tok['value'] == '^' else 'subscript'
                if elements and i + 1 < len(tokens):
                    base = elements[-1]
                    i   += 1
                    exp  = self._token_to_node(tokens[i], graph)
                    if exp and exp.id != base.id:
                        self._add_edge(
                            graph, formula_id,
                            base, exp,
                            operation=op_name,
                        )

            # ── Арифметические операции (+, -, *, /) ──────────────────────────
            elif tok['type'] == 'sym' and tok['value'] in OP_SYMBOLS:
                op_name = OP_SYMBOLS[tok['value']]
                operations.append(op_name)

            # ── Обычные элементы (var, num, constant) ─────────────────────────
            else:
                node = self._token_to_node(tok, graph)
                if node:
                    elements.append(node)

            i += 1

        # ── Связываем элементы через арифметические операции ─────────────────
        op_idx = 0
        for j in range(len(elements) - 1):
            left  = elements[j]
            right = elements[j + 1]

            if left.id == right.id:
                continue

            op = operations[op_idx] if op_idx < len(operations) else 'adjacent'
            op_idx += 1

            # Пропускаем adjacent если левый — оператор
            # (связь уже создана через applied_to)
            if op == 'adjacent' and left.node_type == 'operator':
                continue

            self._add_edge(graph, formula_id, left, right, operation=op)

        # Главный узел: предпочитаем оператор если он есть
        if last_op_node is not None and last_op_node in elements:
            return last_op_node
        return elements[0] if elements else None

    def _mark_function_calls(self, tokens: list[dict]) -> list[dict]:
        """
        Обнаруживает паттерн переменная + переменная где первая
        является функцией второй: f x → f_func x.
        Эвристика: буквы f, g, h, F, G, H перед переменной — функции.
        """
        if len(tokens) < 2:
            return tokens

        result = list(tokens)
        for i in range(len(result) - 1):
            cur = result[i]
            nxt = result[i + 1]

            if (cur['type'] == 'var'
                    and cur['value'] in 'fghFGH'
                    and nxt['type'] in ('var', 'num', 'constant')
                    and nxt['value'] not in OP_SYMBOLS):
                result[i] = {**cur, 'type': 'var_func'}

        return result

    # ─── Вспомогательные методы ───────────────────────────────────────────────

    def _uid(self, prefix: str) -> str:
        """Генерирует уникальный ID."""
        self._counter += 1
        clean = re.sub(r'[^a-zA-Z0-9]', '_', prefix)[:20]
        return f"{clean}_{self._counter}"

    def _get_or_create_node(
        self,
        label: str,
        node_type: str,
        latex: str,
        description: str = "",
    ) -> MathNode:
        """
        Возвращает существующий узел или создаёт новый.
        Ключ = type:label — одинаковые элементы из разных формул
        указывают на один узел графа.
        """
        key = f"{node_type}:{label}"
        if key not in self._node_registry:
            self._node_registry[key] = MathNode(
                id=self._uid(label),
                label=label,
                node_type=node_type,
                latex=latex,
                description=description,
            )
        return self._node_registry[key]

    def _ensure_in_graph(self, node: MathNode, graph: FormulaGraph):
        """Добавляет узел в граф если его там ещё нет."""
        if node not in graph.nodes:
            graph.nodes.append(node)

    def _token_to_node(
        self,
        tok: dict,
        graph: FormulaGraph,
    ) -> MathNode | None:
        """Конвертирует токен в узел графа."""
        if tok is None:
            return None

        ttype = tok['type']
        value = tok['value']

        type_map = {
            'var':      'variable',
            'num':      'number',
            'constant': 'constant',
            'operator': 'operator',
            'function': 'function',
            'var_func': 'function',
        }

        ntype = type_map.get(ttype)
        if ntype is None:
            return None

        desc_map = {
            'variable': f"Переменная {value}",
            'number':   f"Число {value}",
            'constant': f"Константа {value}",
            'operator': self._op_description(value),
            'function': f"Функция {value}",
        }

        node = self._get_or_create_node(
            label=value,
            node_type=ntype,
            latex=tok['latex'],
            description=desc_map.get(ntype, value),
        )
        self._ensure_in_graph(node, graph)
        return node

    def _add_edge(
        self,
        graph: FormulaGraph,
        formula_id: str,
        source: MathNode,
        target: MathNode,
        operation: str,
        weight: float = 1.0,
    ):
        """Добавляет ребро если его ещё нет (без петель и дублей)."""
        if source.id == target.id:
            return

        for e in graph.edges:
            if (e.source_id  == source.id
                    and e.target_id == target.id
                    and e.operation == operation
                    and e.formula_id == formula_id):
                return

        graph.edges.append(MathEdge(
            source_id=source.id,
            target_id=target.id,
            operation=operation,
            formula_id=formula_id,
            description=f"{source.label} --[{operation}]--> {target.label}",
            weight=weight,
        ))

    def _op_description(self, op: str) -> str:
        """Описание оператора."""
        descriptions = {
            '∫':   'Оператор интегрирования',
            '∬':   'Двойной интеграл',
            '∭':   'Тройной интеграл',
            '∮':   'Контурный интеграл',
            'Σ':   'Оператор суммирования',
            'Π':   'Оператор произведения',
            'lim': 'Оператор предела',
            '∂':   'Оператор частной производной',
            '∇':   'Оператор набла (градиент)',
            'Δ':   'Оператор Лапласа / разность',
            '√':   'Оператор квадратного корня',
            '/':   'Дробь (деление)',
            'sup': 'Супремум',
            'inf': 'Инфимум',
            'max': 'Максимум',
            'min': 'Минимум',
            'det': 'Определитель',
        }
        return descriptions.get(op, f"Оператор {op}")

