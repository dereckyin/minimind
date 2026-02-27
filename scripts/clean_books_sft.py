import argparse
import json
import re
from pathlib import Path


TEMPLATE_HINTS = (
    "請根據以下內容整理重點",
    "請用簡短方式說明這段內容",
    "這段文字主要在講什麼",
)


def strip_control_chars(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    parts = re.split(r"[。！？!?；;]", text)
    out = [p.strip() for p in parts if len(p.strip()) >= 12]
    return out


def char_overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    return len(sa & sb) / max(1, len(sb))


def build_answer_from_context(prompt: str, context: str, max_chars: int) -> str:
    sents = split_sentences(context)
    if not sents:
        return ""

    # Pick spread-out sentences to avoid simply copying the beginning.
    idxs = sorted(set([0, len(sents) // 2, len(sents) - 1]))
    picks = [sents[i] for i in idxs if i < len(sents)]
    picks = [p for p in picks if len(p) >= 12]
    if not picks:
        picks = sents[:3]

    if "整理重點" in prompt:
        lines = []
        for i, p in enumerate(picks[:3], 1):
            lines.append(f"{i}. {p}")
        ans = "重點整理：\n" + "\n".join(lines)
    elif "簡短方式" in prompt:
        core = "；".join(picks[:2])
        ans = f"簡述：{core}"
    elif "主要在講什麼" in prompt:
        core = "；".join(picks[:2])
        ans = f"主題是：{core}"
    else:
        ans = "；".join(picks[:2])

    return ans[:max_chars].strip()


def extract_context(user_text: str) -> tuple[str, str]:
    if "\n" not in user_text:
        return user_text, ""
    head, rest = user_text.split("\n", 1)
    return head.strip(), rest.strip()


def looks_like_label_leak(user_text: str, assistant_text: str) -> bool:
    _, ctx = extract_context(user_text)
    if not ctx or not assistant_text:
        return False
    if ctx.startswith(assistant_text):
        return True
    if assistant_text in user_text and len(assistant_text) > 60:
        return True
    return char_overlap_ratio(ctx, assistant_text) > 0.9 and len(assistant_text) > 120


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean books SFT seed dataset")
    parser.add_argument("--input", required=True, help="Input SFT seed jsonl")
    parser.add_argument("--output", default="dataset/books_sft_clean.jsonl", help="Output clean jsonl")
    parser.add_argument("--report", default="dataset/books_sft_clean_report.json", help="Output report json")
    parser.add_argument("--max_answer_chars", type=int, default=220, help="Max assistant chars")
    parser.add_argument("--min_answer_chars", type=int, default=24, help="Min assistant chars")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped_invalid = 0
    dropped_short = 0
    rewritten = 0
    dropped_duplicate = 0

    seen = set()
    rows = []

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                conv = obj.get("conversations", [])
                if len(conv) < 2:
                    dropped_invalid += 1
                    continue
                user = normalize_space(strip_control_chars(conv[0].get("content", "")))
                assistant = normalize_space(strip_control_chars(conv[1].get("content", "")))
                if not user:
                    dropped_invalid += 1
                    continue

                if looks_like_label_leak(user, assistant):
                    head, ctx = extract_context(user)
                    assistant_new = build_answer_from_context(head, ctx, args.max_answer_chars)
                    if assistant_new and assistant_new != assistant:
                        assistant = assistant_new
                        rewritten += 1

                if len(assistant) < args.min_answer_chars:
                    dropped_short += 1
                    continue

                key = (user, assistant)
                if key in seen:
                    dropped_duplicate += 1
                    continue
                seen.add(key)

                rows.append(
                    {
                        "conversations": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant[: args.max_answer_chars]},
                        ]
                    }
                )
                kept += 1
            except Exception:
                dropped_invalid += 1
                continue

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "total": total,
        "kept": kept,
        "dropped_invalid": dropped_invalid,
        "dropped_short": dropped_short,
        "dropped_duplicate": dropped_duplicate,
        "rewritten": rewritten,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
