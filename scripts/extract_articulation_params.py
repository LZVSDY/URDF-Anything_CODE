import os
import re
import json
import argparse
from typing import Any, Dict, Optional


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    brace = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                return text[start : i + 1]
    return None


def clean_seg_tokens(d: dict) -> dict:
    """Remove [SEG] tokens from link semantic names."""
    result = dict(d)
    if "links" in result and isinstance(result["links"], dict):
        result["links"] = {
            k: re.sub(r'\s*\[SEG\]\s*', '', v).strip()
            for k, v in result["links"].items()
        }
    return result


def parse_pred(content: dict) -> Optional[dict]:
    """Parse pred_answers field → dict."""
    pa = content.get("pred_answers", "")
    if not pa:
        return None
    js = extract_first_json_object(pa)
    if js is None:
        return None
    try:
        d = json.loads(js)
    except Exception:
        js2 = re.sub(r",\s*([}\]])", r"\1", js)
        try:
            d = json.loads(js2)
        except Exception:
            return None
    return clean_seg_tokens(d)


def parse_gt(content: dict) -> Optional[dict]:
    if isinstance(content.get("answer"), dict):
        return clean_seg_tokens(content["answer"])
    gt_s = content.get("gt_answers")
    if isinstance(gt_s, str):
        js = extract_first_json_object(gt_s)
        if js:
            try:
                d = json.loads(js)
                return clean_seg_tokens(d)
            except Exception:
                pass
    return None


def collect_json_files(root: str):
    skip_dirs = {"seg_result", "seg_visualization", "seg_per_part",
                 "articulation_params", "reconstructed"}
    files = []
    for r, dirs, fns in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in fns:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser(description="Extract pred/gt articulation params")
    ap.add_argument("--root", type=str, required=True,
                    help="Root dir of inference output JSONs")
    ap.add_argument("--out_dir", type=str, default="",
                    help="Output dir for param files (default: {root}/articulation_params)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "articulation_params")
    os.makedirs(out_dir, exist_ok=True)

    json_files = collect_json_files(args.root)
    print(f"Found {len(json_files)} JSON files in {args.root}")

    n_ok = 0
    n_fail = 0

    for fp in json_files:
        try:
            content = json.load(open(fp, "r"))
        except Exception as e:
            print(f"  [SKIP] Cannot read {fp}: {e}")
            n_fail += 1
            continue

        pred = parse_pred(content)
        gt = parse_gt(content)

        if pred is None and gt is None:
            n_fail += 1
            continue

        base = os.path.splitext(os.path.basename(fp))[0]

        miou = content.get("miou")
        alignment = content.get("alignment")
        normalize_info = content.get("normalize")

        if pred is not None:
            pred_out = {
                "sample_id": base,
                "miou": miou,
                "alignment": alignment,
                "normalize": normalize_info,
                "articulation": pred,
            }
            pred_path = os.path.join(out_dir, f"{base}_pred.json")
            with open(pred_path, "w") as f:
                json.dump(pred_out, f, indent=2, ensure_ascii=False)

        if gt is not None:
            gt_out = {
                "sample_id": base,
                "normalize": normalize_info,
                "articulation": gt,
            }
            gt_path = os.path.join(out_dir, f"{base}_gt.json")
            with open(gt_path, "w") as f:
                json.dump(gt_out, f, indent=2, ensure_ascii=False)

        n_ok += 1

    print(f"\nDone: {n_ok} samples extracted, {n_fail} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()