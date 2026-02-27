import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader
from pypdf.errors import LimitReachedError
try:
    import fitz  # type: ignore[import-not-found]  # PyMuPDF
except Exception:
    fitz = None


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


def _extract_pdf_text_fitz(path: Path, max_pages: int, max_seconds: float) -> str:
    if fitz is None:
        return ""
    start = time.time()
    docs = []
    try:
        with fitz.open(str(path)) as doc:
            total = len(doc)
            limit = total if max_pages <= 0 else min(total, max_pages)
            for idx in range(limit):
                if max_seconds > 0 and (time.time() - start) > max_seconds:
                    break
                docs.append(doc[idx].get_text("text") or "")
    except Exception:
        return ""
    return normalize_text("\n".join(docs))


def _extract_pdf_text_pypdf(path: Path, max_pages: int, max_seconds: float) -> str:
    start = time.time()
    pages = []
    try:
        reader = PdfReader(str(path), strict=False)
        total = len(reader.pages)
        limit = total if max_pages <= 0 else min(total, max_pages)
        for idx in range(limit):
            if max_seconds > 0 and (time.time() - start) > max_seconds:
                break
            page = reader.pages[idx]
            try:
                pages.append(page.extract_text() or "")
            except LimitReachedError:
                continue
            except Exception as ex:
                # Skip known unsupported encodings quickly.
                if "B5pc-H" in str(ex):
                    return ""
                continue
    except Exception:
        return ""
    return normalize_text("\n".join(pages))


def extract_pdf_text(path: Path, max_pages: int = 50, max_seconds: float = 15.0) -> str:
    # Prefer PyMuPDF first for better robustness on legacy CJK encodings.
    text = _extract_pdf_text_fitz(path, max_pages=max_pages, max_seconds=max_seconds)
    if text:
        return text
    return _extract_pdf_text_pypdf(path, max_pages=max_pages, max_seconds=max_seconds)


def extract_epub_text(path: Path) -> str:
    book = epub.read_epub(str(path))
    docs = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            soup = BeautifulSoup(item.get_body_content(), "lxml")
            docs.append(soup.get_text(separator="\n"))
        except Exception:
            continue
    return normalize_text("\n".join(docs))


def extract_book_text(path: Path, pdf_max_pages: int = 50, pdf_max_seconds: float = 15.0) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path, max_pages=pdf_max_pages, max_seconds=pdf_max_seconds)
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


def scan_books(input_dir: Path, max_depth: int = -1, max_scan_books: int = 0) -> List[Path]:
    """
    max_depth: 0=only input_dir, 1=input_dir+one level down, -1=unlimited
    max_scan_books: >0 means stop scanning once enough books are found
    """
    books: List[Path] = []
    input_dir = input_dir.absolute()
    root_depth = len(input_dir.parts)

    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        depth = len(root_path.parts) - root_depth
        if max_depth >= 0 and depth > max_depth:
            dirs[:] = []
            continue
        if max_depth >= 0 and depth == max_depth:
            # Reaching target depth: do not descend deeper.
            dirs[:] = []
        for name in files:
            lower = name.lower()
            if not (lower.endswith(".pdf") or lower.endswith(".epub")):
                continue
            books.append(root_path / name)
            if max_scan_books > 0 and len(books) >= max_scan_books:
                return books
    return books


def sample_books(books: List[Path], max_books: int, sample_mode: str, seed: int) -> List[Path]:
    if max_books <= 0 or len(books) <= max_books:
        return books
    if sample_mode == "head":
        return books[:max_books]
    rng = random.Random(seed)
    sampled = books[:]
    rng.shuffle(sampled)
    return sorted(sampled[:max_books])


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
    parser.add_argument("--max_books", type=int, default=0, help="Max number of books to process (0 = all)")
    parser.add_argument("--sample_mode", choices=["random", "head"], default="random", help="Sampling strategy when max_books > 0")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for book sampling")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in book list (for parallel/split runs)")
    parser.add_argument("--end_index", type=int, default=-1, help="End index (exclusive). -1 = all. E.g. 0,50000 then 50000,100000")
    parser.add_argument("--report_per_book", type=int, default=1, choices=[0, 1], help="0=only totals in report (save memory for large corpus)")
    parser.add_argument("--log_interval", type=int, default=1000, help="Print progress every N books")
    parser.add_argument("--append", action="store_true", help="Append to output files (for merging split runs)")
    parser.add_argument("--search_depth", type=int, default=1, help="Folder search depth: 0=input_dir only, 1=+one level, -1=unlimited")
    parser.add_argument("--early_stop_scan", type=int, default=0, choices=[0, 1],
                        help="1=stop scanning as soon as max_books is reached")
    parser.add_argument("--pdf_max_pages", type=int, default=50, help="Max pages to parse per PDF (0=all)")
    parser.add_argument("--pdf_max_seconds", type=float, default=15.0, help="Max parsing seconds per PDF (0=no limit)")
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
    write_mode = "a" if args.append else "w"

    early_stop = bool(args.early_stop_scan and args.max_books > 0)
    print(
        f"Scanning input directory (depth={args.search_depth}) for PDF/EPUB files"
        f"{' with early stop' if early_stop else ''}...",
        flush=True
    )
    scan_limit = args.max_books if early_stop else 0
    books = scan_books(input_dir, max_depth=args.search_depth, max_scan_books=scan_limit)
    if not books:
        raise RuntimeError("No PDF/EPUB files found.")
    if early_stop:
        # Already capped during scanning; keep order for fastest startup.
        pass
    else:
        books = sample_books(books, args.max_books, args.sample_mode, args.seed)

    end = args.end_index if args.end_index >= 0 else len(books)
    books = books[args.start_index:end]
    if not books:
        raise RuntimeError(f"No books in range start_index={args.start_index} end_index={end}.")

    print(f"Found {len(books)} books to process (range {args.start_index}-{end}). Processing...", flush=True)

    total_chunks = 0
    total_sft = 0
    failed_books = 0
    report = {"books": [], "totals": {}, "range": {"start_index": args.start_index, "end_index": end, "count": len(books)}}

    start_time = time.time()
    with pretrain_out.open(write_mode, encoding="utf-8") as f_pre, sft_out.open(write_mode, encoding="utf-8") as f_sft:
        for i, book_path in enumerate(books):
            try:
                raw_text = extract_book_text(
                    book_path,
                    pdf_max_pages=args.pdf_max_pages,
                    pdf_max_seconds=args.pdf_max_seconds
                )
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
                if args.report_per_book:
                    report["books"].append(
                        {
                            "file": str(book_path),
                            "paragraphs": len(paragraphs),
                            "pretrain_chunks": len(chunks),
                            "sft_pairs": len(sft_rows),
                        }
                    )
            except Exception as ex:
                failed_books += 1
                if args.report_per_book:
                    report["books"].append(
                        {
                            "file": str(book_path),
                            "paragraphs": 0,
                            "pretrain_chunks": 0,
                            "sft_pairs": 0,
                            "error": str(ex),
                        }
                    )

            n = i + 1
            show_progress = n <= 5 or n % args.log_interval == 0 or n == len(books)
            if show_progress:
                elapsed = time.time() - start_time
                rate = n / elapsed if elapsed > 0 else 0
                eta = (len(books) - n) / rate if rate > 0 else 0
                print(
                    f"[progress] {n}/{len(books)} ({100*n/len(books):.1f}%) | "
                    f"pretrain={total_chunks} sft={total_sft} failed={failed_books} | "
                    f"{elapsed:.0f}s elapsed, ~{eta:.0f}s ETA | "
                    f"{book_path.name[:40]}",
                    flush=True
                )
            if n % args.log_interval == 0:
                f_pre.flush()
                f_sft.flush()

    report["totals"] = {
        "books": len(books),
        "pretrain_chunks": total_chunks,
        "sft_pairs": total_sft,
        "failed_books": failed_books,
        "pretrain_out": str(pretrain_out),
        "sft_out": str(sft_out),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)
    print(f"Report written to: {report_out}", flush=True)


if __name__ == "__main__":
    main()
