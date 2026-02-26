import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        # Remove common page number noise.
        if not line:
            continue
        if re.fullmatch(r"\d{1,4}", line):
            continue
        if re.fullmatch(r"第?\s*\d+\s*页", line):
            continue
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    return "\n".join(lines)


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in parts if len(p) >= 20]


def chunk_paragraphs(paragraphs: Iterable[str], max_chars: int, min_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chars:
            if current and current_len >= min_chars:
                chunks.append("\n".join(current))
            current = []
            current_len = 0
            # Hard slice very long paragraphs.
            start = 0
            while start < para_len:
                piece = para[start:start + max_chars].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
                start += max_chars
            continue

        if current_len + para_len + (1 if current else 0) > max_chars:
            if current_len >= min_chars:
                chunks.append("\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + (1 if current_len > 0 else 0)

    if current and current_len >= min_chars:
        chunks.append("\n".join(current))
    return chunks


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def extract_epub_text(path: Path) -> str:
    book = epub.read_epub(str(path))
    docs = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_body_content(), "lxml")
        docs.append(soup.get_text(separator="\n"))
    return normalize_text("\n".join(docs))


def extract_book_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix == ".epub":
        return extract_epub_text(path)
    return ""


def build_sft_seed_samples(chunks: List[str], pairs_per_book: int, max_answer_chars: int) -> List[dict]:
    templates = [
        "請根據以下內容整理重點：\n{context}",
        "請用簡短方式說明這段內容：\n{context}",
        "這段文字主要在講什麼？\n{context}",
    ]
    out = []
    if not chunks:
        return out
    step = max(1, len(chunks) // pairs_per_book)
    selected = [chunks[i] for i in range(0, len(chunks), step)][:pairs_per_book]
    for i, chunk in enumerate(selected):
        user_prompt = templates[i % len(templates)].format(context=chunk[: max_answer_chars * 2])
        assistant_answer = chunk[:max_answer_chars]
        out.append(
            {
                "conversations": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_answer},
                ]
            }
        )
    return out


def scan_books(input_dir: Path) -> List[Path]:
    files = []
    for pattern in ("**/*.pdf", "**/*.epub"):
        files.extend(input_dir.glob(pattern))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniMind datasets from PDF/EPUB books")
    parser.add_argument("--input_dir", required=True, help="Directory that contains pdf/epub books")
    parser.add_argument("--pretrain_out", default="dataset/books_pretrain.jsonl", help="Output JSONL for pretrain")
    parser.add_argument("--sft_out", default="dataset/books_sft_seed.jsonl", help="Output JSONL for SFT seed")
    parser.add_argument("--report_out", default="dataset/books_build_report.json", help="Build report path")
    parser.add_argument("--max_chars", type=int, default=1200, help="Max chars per chunk")
    parser.add_argument("--min_chars", type=int, default=200, help="Min chars per chunk")
    parser.add_argument("--pairs_per_book", type=int, default=20, help="How many SFT pairs to create per book")
    parser.add_argument("--max_answer_chars", type=int, default=320, help="Max assistant chars in SFT seed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pretrain_out = Path(args.pretrain_out)
    sft_out = Path(args.sft_out)
    report_out = Path(args.report_out)
    pretrain_out.parent.mkdir(parents=True, exist_ok=True)
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    books = scan_books(input_dir)
    if not books:
        raise RuntimeError("No PDF/EPUB files found.")

    total_chunks = 0
    total_sft = 0
    report = {"books": [], "totals": {}}

    with pretrain_out.open("w", encoding="utf-8") as f_pre, sft_out.open("w", encoding="utf-8") as f_sft:
        for book_path in books:
            raw_text = extract_book_text(book_path)
            paragraphs = split_paragraphs(raw_text)
            chunks = chunk_paragraphs(paragraphs, max_chars=args.max_chars, min_chars=args.min_chars)
            sft_rows = build_sft_seed_samples(
                chunks,
                pairs_per_book=args.pairs_per_book,
                max_answer_chars=args.max_answer_chars,
            )

            for chunk in chunks:
                f_pre.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
            for row in sft_rows:
                f_sft.write(json.dumps(row, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            total_sft += len(sft_rows)
            report["books"].append(
                {
                    "file": str(book_path),
                    "paragraphs": len(paragraphs),
                    "pretrain_chunks": len(chunks),
                    "sft_pairs": len(sft_rows),
                }
            )

    report["totals"] = {
        "books": len(books),
        "pretrain_chunks": total_chunks,
        "sft_pairs": total_sft,
        "pretrain_out": str(pretrain_out),
        "sft_out": str(sft_out),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["totals"], ensure_ascii=False))
    print(f"Report written to: {report_out}")


if __name__ == "__main__":
    main()
