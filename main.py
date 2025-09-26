import re
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set


# --------------------------
# Data models
# --------------------------


@dataclass
class CaretPosition:
    list_id: int
    paragraph_id: int
    char_pos: int


@dataclass
class TableCell:
    text: str
    row_span: int = 1
    col_span: int = 1


@dataclass
class TableBlock:
    rows: List[List[TableCell]]  # 2-D logical grid with spans
    grid_size: Tuple[int, int]  # (n_rows, n_cols)
    raw_spans: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )  # (r,c,row_span,col_span)


@dataclass
class ParagraphBlock:
    text: str


@dataclass
class Document:
    section_count: int = 0
    page_start_num: int = 0
    footnote_start_num: int = 0
    endnote_start_num: int = 0
    picture_start_num: int = 0
    table_start_num: int = 0
    equation_start_num: int = 0
    caret_pos: CaretPosition = field(default_factory=lambda: CaretPosition(0, 0, 0))
    blocks: List[Any] = field(
        default_factory=list
    )  # sequence of ParagraphBlock | TableBlock
    full_text: str = ""  # paragraphs + tables rendered as Markdown
    tables: List[TableBlock] = field(default_factory=list)


# --------------------------
# Parser
# --------------------------


class HwpXParser:
    XML_FILENAME_HEADER = "Contents/header.xml"
    XML_FILENAME_SETTINGS = "settings.xml"
    XML_FILENAME_CONTENT = "Contents/content.hpf"
    CONTENTS_BASE = "Contents/"

    CTRL_TEXT_TAGS = {
        "edit",
        "comboBox",
        "listItem",
        "caption",
        "btn",
        "radio",
        "check",
        "datePicker",
        "timePicker",
        "number",
        "inputText",
    }
    CTRL_TEXT_ATTRS = ("text", "value", "displayText", "label")

    @staticmethod
    def _ln(tag: str) -> str:
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    @staticmethod
    def extract_namespaces(xml_bytes) -> dict:
        ns = {}
        for _, (prefix, uri) in ET.iterparse(xml_bytes, events=("start-ns",)):
            ns[prefix] = uri
        return ns

    @classmethod
    def read_header(cls, zipf: zipfile.ZipFile, doc: Document):
        if cls.XML_FILENAME_HEADER not in zipf.namelist():
            return
        ns = cls.extract_namespaces(BytesIO(zipf.read(cls.XML_FILENAME_HEADER)))
        root = ET.parse(BytesIO(zipf.read(cls.XML_FILENAME_HEADER))).getroot()
        sec_cnt = root.get("secCnt")
        if sec_cnt is not None:
            doc.section_count = int(sec_cnt)
        begin = root.find("hh:beginNum", ns)
        if begin is not None:
            doc.page_start_num = int(begin.get("page", 0))
            doc.footnote_start_num = int(begin.get("footnote", 0))
            doc.endnote_start_num = int(begin.get("endnote", 0))
            doc.picture_start_num = int(begin.get("pic", 0))
            doc.table_start_num = int(begin.get("tbl", 0))
            doc.equation_start_num = int(begin.get("equation", 0))

    @classmethod
    def read_settings(cls, zipf: zipfile.ZipFile, doc: Document):
        if cls.XML_FILENAME_SETTINGS not in zipf.namelist():
            return
        ns = cls.extract_namespaces(BytesIO(zipf.read(cls.XML_FILENAME_SETTINGS)))
        root = ET.parse(BytesIO(zipf.read(cls.XML_FILENAME_SETTINGS))).getroot()
        caret = root.find(".//ha:CaretPosition", ns)
        if caret is not None:
            doc.caret_pos = CaretPosition(
                int(caret.get("listIDRef") or 0),
                int(caret.get("paraIDRef") or 0),
                int(caret.get("pos") or 0),
            )

    @classmethod
    def _read_manifest_and_spine(cls, zipf: zipfile.ZipFile) -> List[str]:
        if cls.XML_FILENAME_CONTENT not in zipf.namelist():
            return []
        xml = zipf.read(cls.XML_FILENAME_CONTENT)
        ns = cls.extract_namespaces(BytesIO(xml))
        root = ET.fromstring(xml)

        manifest_map: Dict[str, str] = {}
        manifest = root.find("opf:manifest", ns)
        if manifest is not None:
            for item in manifest.findall("opf:item", ns):
                _id = item.get("id")
                href = item.get("href")
                if _id and href:
                    if not href.startswith(cls.CONTENTS_BASE):
                        href = cls.CONTENTS_BASE + href
                    manifest_map[_id] = href

        ordered: List[str] = []
        spine = root.find("opf:spine", ns)
        if spine is not None:
            for itemref in spine.findall("opf:itemref", ns):
                ref = itemref.get("idref")
                if ref and ref in manifest_map and manifest_map[ref].endswith(".xml"):
                    if re.search(r"/section\d+\.xml$", manifest_map[ref]):
                        ordered.append(manifest_map[ref])
        return ordered

    # -------- utils --------

    @staticmethod
    def _norm(s: str) -> str:
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def _tokens(s: str) -> List[str]:
        # 한글/영문/숫자 토큰 기준. 기호는 공백 처리.
        s = re.sub(r"[\t\r\n]+", " ", s)
        s = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+", " ", s)
        toks = [t for t in s.split() if t]
        return toks

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        A, B = set(a), set(b)
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0

    # -------- text collectors --------

    @classmethod
    def _collect_text_from_node(cls, node) -> str:
        """가시 텍스트 수집. run/t/ctrl/lineBreak/tab + tail, 실패시 itertext() 폴백."""
        parts: List[str] = []
        for n in node.iter():
            name = cls._ln(n.tag)
            if name == "t" and n.text:
                parts.append(n.text)
            elif name == "lineBreak":
                parts.append("\n")
            elif name == "tab":
                parts.append("\t")
            elif name in ("pageBreak", "columnBreak", "sectionBreak"):
                parts.append("\n")
            elif name in cls.CTRL_TEXT_TAGS:
                for a in cls.CTRL_TEXT_ATTRS:
                    v = n.get(a)
                    if v:
                        parts.append(v)
                        break
            else:
                v = n.get("text")
                if v:
                    parts.append(v)
            # tail도 수집(인라인 컨트롤 뒤 텍스트 보호)
            if n.tail:
                parts.append(n.tail)

        s = "".join(parts).strip()
        if s:
            return s
        # 폴백: 도형/특수컨트롤 내 텍스트 전체 긁기
        return "".join(node.itertext()).strip()

    @classmethod
    def _normalize_para(cls, s: str) -> str:
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    # -------- table parsing --------

    @classmethod
    def _parse_table(cls, tbl_el) -> TableBlock:
        logical_rows: List[List[TableCell]] = []
        raw_spans: List[Tuple[int, int, int, int]] = []

        row_idx = 0
        for tr in tbl_el:
            if cls._ln(tr.tag) != "tr":
                continue
            row_cells: List[TableCell] = []
            for tc in tr:
                if cls._ln(tc.tag) != "tc":
                    continue
                texts: List[str] = []
                for el in tc.iter():
                    if cls._ln(el.tag) == "p":
                        t = cls._collect_text_from_node(el)
                        t = cls._normalize_para(t)
                        if t:
                            texts.append(t)
                cell_text = "\n".join(texts).strip()
                if not cell_text:
                    # 셀 안에 p가 없거나 t가 없을 때 도형/컨트롤 텍스트 폴백
                    cell_text = cls._normalize_para("".join(tc.itertext()).strip())
                rspan = cls._safe_int(tc.get("rowSpan") or tc.get("rowspan") or "1", 1)
                cspan = cls._safe_int(tc.get("colSpan") or tc.get("colspan") or "1", 1)
                row_cells.append(TableCell(cell_text, rspan, cspan))
                if rspan > 1 or cspan > 1:
                    raw_spans.append((row_idx, len(row_cells) - 1, rspan, cspan))
            logical_rows.append(row_cells)
            row_idx += 1

        max_cols = 0
        for r in logical_rows:
            width = 0
            for c in r:
                width += c.col_span if c.col_span > 1 else 1
            max_cols = max(max_cols, width)
        n_rows = len(logical_rows)

        return TableBlock(
            rows=logical_rows, grid_size=(n_rows, max_cols), raw_spans=raw_spans
        )

    @staticmethod
    def _safe_int(s: str, default: int) -> int:
        try:
            return int(s)
        except Exception:
            return default

    # -------- section -> blocks --------

    def _parse_section_blocks(self, xml_bytes: bytes) -> List[Any]:
        root = ET.fromstring(xml_bytes)
        blocks: List[Any] = []

        def walk(node, inside_tbl: bool):
            name = self._ln(node.tag)

            if name == "tbl":
                blocks.append(self._parse_table(node))
                return  # do not collect <p> inside table

            if name == "p" and not inside_tbl:
                txt = self._collect_text_from_node(node)
                txt = self._normalize_para(txt)
                if txt:
                    blocks.append(ParagraphBlock(txt))

            next_inside = inside_tbl or (name == "tc")
            for child in list(node):
                walk(child, next_inside)

        walk(root, inside_tbl=False)
        return blocks

    # -------- de-duplication --------
    # 정책:
    # 1) 표에서 읽힌 내용과 겹치는 문단은 전부 제거. 포함관계 양방향 + 토큰 유사도>=0.8.
    # 2) 여러 개(기본 2개 이상)의 표 시그니처를 포함하는 초장문 문단은 "전체 텍스트본"으로 간주하고 제거.

    @staticmethod
    def _tokens(s: str) -> List[str]:
        s = re.sub(r"[\t\r\n]+", " ", s)
        s = re.sub(r"[^\w\u3131-\u318E\uAC00-\uD7A3]+", " ", s)
        return [t for t in s.split() if t]

    @staticmethod
    def _ngrams(tokens: List[str], n_min: int, n_max: int) -> Set[str]:
        out: Set[str] = set()
        L = len(tokens)
        for n in range(n_min, min(n_max, L) + 1):
            for i in range(L - n + 1):
                out.add(" ".join(tokens[i : i + n]))
        return out

    @staticmethod
    def _strip_space(s: str) -> str:
        return re.sub(r"\s+", "", s or "")

    @staticmethod
    def _char_ngrams(s: str, n_min: int = 6, n_max: int = 24) -> Set[str]:
        """공백 제거 후 문자 n-gram. 셀 경계/구분자 소실 대비."""
        s = HwpXParser._strip_space(s)
        out: Set[str] = set()
        L = len(s)
        if L == 0:
            return out
        for n in range(n_min, min(n_max, L) + 1):
            for i in range(L - n + 1):
                out.add(s[i : i + n])
        return out

    @staticmethod
    def _block_fingerprint(block: Any) -> str:
        if isinstance(block, ParagraphBlock):
            return "P::" + re.sub(r"\s+", " ", (block.text or "").strip())
        if isinstance(block, TableBlock):
            flat = []
            for r in block.rows:
                flat.append(
                    "|".join(
                        f"{(c.text or '').strip()}[{c.row_span}x{c.col_span}]"
                        for c in r
                    )
                )
            return "T::" + "||".join(flat)
        return "?"

    @staticmethod
    def _table_atoms(tb: TableBlock) -> Tuple[Set[str], Set[str], Set[str]]:
        """토큰 n-gram: cell/row/whole"""
        cell_atoms: Set[str] = set()
        row_atoms: Set[str] = set()

        for r in tb.rows:
            # cell
            for c in r:
                t = HwpXParser._norm(c.text or "")
                cell_atoms |= HwpXParser._ngrams(HwpXParser._tokens(t), 2, 8)

            # row
            row_txt = " ".join(
                (c.text or "").strip() for c in r if (c.text or "").strip()
            )
            row_txt = HwpXParser._norm(row_txt)
            row_atoms |= HwpXParser._ngrams(HwpXParser._tokens(row_txt), 2, 10)

        whole = " ".join(
            " ".join((c.text or "").strip() for c in r if (c.text or "").strip())
            for r in tb.rows
        )
        whole = HwpXParser._norm(whole)
        whole_atoms = HwpXParser._ngrams(HwpXParser._tokens(whole), 2, 12)
        return cell_atoms, row_atoms, whole_atoms

    @staticmethod
    def _table_char_atoms(tb: TableBlock) -> Tuple[Set[str], Set[str]]:
        """문자 n-gram: row/whole (공백 제거)"""
        row_chars: Set[str] = set()
        for r in tb.rows:
            row_txt = " ".join(
                (c.text or "").strip() for c in r if (c.text or "").strip()
            )
            row_chars |= HwpXParser._char_ngrams(row_txt, 6, 24)
        whole = " ".join(
            " ".join((c.text or "").strip() for c in r if (c.text or "").strip())
            for r in tb.rows
        )
        whole_chars = HwpXParser._char_ngrams(whole, 6, 28)
        return row_chars, whole_chars

    @staticmethod
    def _cover(a: Set[str], b: Set[str]) -> float:
        return (len(a & b) / len(a)) if a else 0.0

    @staticmethod
    def _jacc(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def dedupe_blocks(blocks: List[Any]) -> List[Any]:
        # 0) exact dedupe
        seen = set()
        base: List[Any] = []
        for b in blocks:
            fp = HwpXParser._block_fingerprint(b)
            if fp in seen:
                continue
            seen.add(fp)
            base.append(b)

        # 1) 표 atom 집합 수집 (토큰+문자)
        cell_all: Set[str] = set()
        row_all: Set[str] = set()
        whole_all: Set[str] = set()
        row_chars_all: Set[str] = set()
        whole_chars_all: Set[str] = set()
        for b in base:
            if isinstance(b, TableBlock):
                ca, ra, wa = HwpXParser._table_atoms(b)
                cell_all |= ca
                row_all |= ra
                whole_all |= wa
                rc, wc = HwpXParser._table_char_atoms(b)
                row_chars_all |= rc
                whole_chars_all |= wc

        def is_mega_para(pt: str) -> bool:
            if len(pt) < 600:
                return False
            grams_tok = HwpXParser._ngrams(HwpXParser._tokens(pt), 2, 10)
            grams_char = HwpXParser._char_ngrams(pt, 6, 20)
            hits = 0
            if HwpXParser._cover(grams_tok, row_all) > 0.08:
                hits += 1
            if HwpXParser._cover(grams_tok, whole_all) > 0.08:
                hits += 1
            if HwpXParser._cover(grams_char, row_chars_all) > 0.06:
                hits += 1
            if HwpXParser._cover(grams_char, whole_chars_all) > 0.06:
                hits += 1
            return hits >= 2

        out: List[Any] = []
        for b in base:
            if not isinstance(b, ParagraphBlock):
                out.append(b)
                continue

            pt = HwpXParser._norm(b.text or "")
            if not pt:
                continue
            if is_mega_para(pt):
                continue

            toks = HwpXParser._tokens(pt)
            if len(toks) < 5:
                out.append(b)
                continue

            grams_tok = HwpXParser._ngrams(toks, 2, 12)
            grams_char = HwpXParser._char_ngrams(pt, 6, 24)

            # 토큰 기준 임계
            cov_cell = HwpXParser._cover(grams_tok, cell_all)
            cov_row = HwpXParser._cover(grams_tok, row_all)
            cov_whole = HwpXParser._cover(grams_tok, whole_all)
            jac_cell = HwpXParser._jacc(grams_tok, cell_all)
            jac_row = HwpXParser._jacc(grams_tok, row_all)
            jac_whole = HwpXParser._jacc(grams_tok, whole_all)

            # 문자 기준 임계(공백 무시)
            cov_row_c = HwpXParser._cover(grams_char, row_chars_all)
            cov_whole_c = HwpXParser._cover(grams_char, whole_chars_all)
            jac_row_c = HwpXParser._jacc(grams_char, row_chars_all)
            jac_whole_c = HwpXParser._jacc(grams_char, whole_chars_all)

            dup_token = (
                cov_cell >= 0.60
                or cov_row >= 0.55
                or cov_whole >= 0.50
                or jac_cell >= 0.60
                or jac_row >= 0.55
                or jac_whole >= 0.50
            )
            dup_char = (
                cov_row_c >= 0.35
                or cov_whole_c >= 0.35
                or jac_row_c >= 0.35
                or jac_whole_c >= 0.35
            )

            if dup_token or dup_char:
                continue  # 표와 실질 중복

            out.append(b)

        # 첫 문단 덤프 컷
        if out and isinstance(out[0], ParagraphBlock):
            if len(HwpXParser._norm(out[0].text)) > 300:
                out = out[1:]

        return out

    # -------- renderers --------

    @staticmethod
    def render_markdown(blocks: List[Any]) -> str:
        lines: List[str] = []
        for b in blocks:
            if isinstance(b, ParagraphBlock):
                lines.append(b.text)
                lines.append("")
            elif isinstance(b, TableBlock):
                _, max_cols = b.grid_size
                header = "|" + "|".join([" "] * max_cols) + "|"
                sep = "|" + "|".join(["---"] * max_cols) + "|"
                lines.append(header)
                lines.append(sep)
                for r in b.rows:
                    expanded: List[str] = []
                    for cell in r:
                        expanded.extend(
                            [cell.text if cell.text else ""] * max(1, cell.col_span)
                        )
                    if len(expanded) < max_cols:
                        expanded += [""] * (max_cols - len(expanded))
                    lines.append("|" + "|".join(expanded[:max_cols]) + "|")
                lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def render_csv_tables(tables: List[TableBlock]) -> List[str]:
        out: List[str] = []
        for tb in tables:
            _, max_cols = tb.grid_size
            rows: List[str] = []
            for r in tb.rows:
                expanded: List[str] = []
                for c in r:
                    val = c.text.replace('"', '""')
                    expanded.extend(
                        [
                            (
                                f'"{val}"'
                                if any(ch in val for ch in [",", "\n", '"'])
                                else val
                            )
                        ]
                        * max(1, c.col_span)
                    )
                if len(expanded) < max_cols:
                    expanded += [""] * (max_cols - len(expanded))
                rows.append(",".join(expanded[:max_cols]))
            out.append("\n".join(rows))
        return out

    # -------- main parse --------

    def parse(self, filepath: str) -> Document:
        doc = Document()
        with zipfile.ZipFile(filepath, "r") as zf:
            self.read_header(zf, doc)
            self.read_settings(zf, doc)

            section_paths = self._read_manifest_and_spine(zf)
            if not section_paths:
                section_paths = sorted(
                    [
                        n
                        for n in zf.namelist()
                        if n.startswith(self.CONTENTS_BASE + "section")
                        and n.endswith(".xml")
                    ],
                    key=lambda p: (
                        int(re.search(r"section(\d+)\.xml$", p).group(1))
                        if re.search(r"section(\d+)\.xml$", p)
                        else 10**9
                    ),
                )

            flow_blocks: List[Any] = []
            for sec in section_paths:
                xml_bytes = zf.read(sec)
                flow_blocks.extend(self._parse_section_blocks(xml_bytes))

            flow_blocks = self.dedupe_blocks(flow_blocks)

            doc.blocks = flow_blocks
            doc.tables = [b for b in flow_blocks if isinstance(b, TableBlock)]
            doc.full_text = self.render_markdown(flow_blocks)

        return doc


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    FILEPATH = "example.hwpx"
    parser = HwpXParser()
    doc = parser.parse(FILEPATH)
    print(doc.full_text)
