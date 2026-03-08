import argparse
import json
import random
from pathlib import Path


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get("conversations"), list) and len(obj["conversations"]) >= 2:
                    rows.append(obj)
            except Exception:
                continue
    return rows


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def make_signature(row: dict, mode: str) -> str:
    conv = row.get("conversations", [])
    if not conv:
        return ""

    if mode == "user_only":
        user = normalize_text(conv[0].get("content", ""))
        return f"u:{user}"

    if mode == "assistant_only":
        assistant = normalize_text(conv[1].get("content", "")) if len(conv) > 1 else ""
        return f"a:{assistant}"

    user = normalize_text(conv[0].get("content", ""))
    assistant = normalize_text(conv[1].get("content", "")) if len(conv) > 1 else ""
    return f"u:{user}\na:{assistant}"


def dedup_rows(rows: list[dict], mode: str) -> tuple[list[dict], int]:
    if mode == "none":
        return rows, 0
    seen = set()
    out = []
    dropped = 0
    for row in rows:
        sig = make_signature(row, mode)
        if sig in seen:
            dropped += 1
            continue
        seen.add(sig)
        out.append(row)
    return out, dropped


def take_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or n >= len(rows):
        sampled = rows[:]
        rng.shuffle(sampled)
        return sampled
    sampled = rng.sample(rows, n)
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix books SFT and general SFT datasets")
    parser.add_argument("--books_sft", required=True, help="Books SFT jsonl path")
    parser.add_argument("--general_sft", required=True, help="General SFT jsonl path")
    parser.add_argument("--output", default="dataset/sft_mix.jsonl", help="Mixed SFT output jsonl")
    parser.add_argument("--report", default="dataset/sft_mix_report.json", help="Mix report output json")
    parser.add_argument("--books_ratio", type=float, default=0.7, help="Books ratio in final mix [0,1]")
    parser.add_argument("--max_samples", type=int, default=0, help="Final mixed sample count. 0 = auto by availability")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--dedup_mode", choices=["none", "user_only", "assistant_only", "user_assistant"], default="user_assistant",
                        help="Dedup strategy before and after mixing")
    args = parser.parse_args()

    if not (0.0 <= args.books_ratio <= 1.0):
        raise ValueError("--books_ratio must be in [0,1]")

    books_path = Path(args.books_sft)
    general_path = Path(args.general_sft)
    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    books_rows_raw = load_jsonl_rows(books_path)
    general_rows_raw = load_jsonl_rows(general_path)
    books_rows = books_rows_raw
    general_rows = general_rows_raw

    books_rows, dropped_books = dedup_rows(books_rows, args.dedup_mode)
    general_rows, dropped_general = dedup_rows(general_rows, args.dedup_mode)

    if not books_rows:
        raise RuntimeError(f"No valid rows in books_sft: {books_path}")
    if not general_rows:
        raise RuntimeError(f"No valid rows in general_sft: {general_path}")

    if args.max_samples > 0:
        target_total = args.max_samples
    else:
        if args.books_ratio in (0.0, 1.0):
            target_total = min(len(books_rows), len(general_rows)) * 2
        else:
            by_books = int(len(books_rows) / args.books_ratio)
            by_general = int(len(general_rows) / (1.0 - args.books_ratio))
            target_total = min(by_books, by_general)
        target_total = max(2, target_total)

    n_books = int(round(target_total * args.books_ratio))
    n_general = target_total - n_books

    n_books = min(n_books, len(books_rows))
    n_general = min(n_general, len(general_rows))
    final_total = n_books + n_general

    if final_total <= 1:
        raise RuntimeError("Not enough rows after ratio and sampling constraints.")

    sampled_books = take_rows(books_rows, n_books, rng)
    sampled_general = take_rows(general_rows, n_general, rng)
    mixed = sampled_books + sampled_general
    rng.shuffle(mixed)

    mixed, dropped_mixed_dup = dedup_rows(mixed, args.dedup_mode)

    with out_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "books_sft": str(books_path),
        "general_sft": str(general_path),
        "output": str(out_path),
        "seed": args.seed,
        "books_ratio_target": args.books_ratio,
        "dedup_mode": args.dedup_mode,
        "input_counts": {
            "books_raw": len(books_rows_raw),
            "general_raw": len(general_rows_raw),
            "books_after_dedup": len(books_rows),
            "general_after_dedup": len(general_rows),
            "dropped_books_dedup": dropped_books,
            "dropped_general_dedup": dropped_general,
        },
        "sampled_counts": {
            "target_total": target_total,
            "books_sampled": n_books,
            "general_sampled": n_general,
            "mixed_before_final_dedup": final_total,
            "mixed_after_final_dedup": len(mixed),
            "dropped_mixed_dedup": dropped_mixed_dup,
        },
        "books_ratio_actual": round((n_books / max(1, final_total)), 4),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_rows": len(mixed), "books_sampled": n_books, "general_sampled": n_general}, ensure_ascii=False))


if __name__ == "__main__":
    main()
