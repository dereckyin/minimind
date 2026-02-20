import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def char_f1(pred: str, ref: str) -> float:
    pred_chars = list(pred.strip())
    ref_chars = list(ref.strip())
    if not pred_chars and not ref_chars:
        return 1.0
    if not pred_chars or not ref_chars:
        return 0.0
    pred_set = {}
    ref_set = {}
    for c in pred_chars:
        pred_set[c] = pred_set.get(c, 0) + 1
    for c in ref_chars:
        ref_set[c] = ref_set.get(c, 0) + 1
    overlap = 0
    for c, n in pred_set.items():
        overlap += min(n, ref_set.get(c, 0))
    precision = overlap / len(pred_chars)
    recall = overlap / len(ref_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_eval_data(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind with fixed QA set")
    parser.add_argument("--tokenizer_path", default="model", type=str)
    parser.add_argument("--weight_path", default="out/full_sft_tiny_128.pth", type=str)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--eval_data", default="dataset/eval_fixed_qa.jsonl", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--save_report", default="out/eval_fixed_qa_report.json", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(
        MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=False,
        )
    )
    weights = torch.load(args.weight_path, map_location=args.device)
    model.load_state_dict(weights, strict=False)
    model = model.to(args.device).eval()

    samples = load_eval_data(Path(args.eval_data))
    results = []
    with torch.no_grad():
        for row in samples:
            question = row["question"]
            reference = row["reference"]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
            generated = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                generated[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True,
            ).strip()
            score = char_f1(answer, reference)
            results.append(
                {
                    "question": question,
                    "reference": reference,
                    "prediction": answer,
                    "char_f1": round(score, 4),
                }
            )

    avg = sum(r["char_f1"] for r in results) / max(1, len(results))
    report = {"avg_char_f1": round(avg, 4), "count": len(results), "items": results}

    report_path = Path(args.save_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"avg_char_f1": report["avg_char_f1"], "count": report["count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
