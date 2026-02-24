import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

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


def parse_pred_answer(pred_answers: str) -> Optional[Dict[str, Any]]:
    js = extract_first_json_object(pred_answers)
    if js is None:
        return None
    try:
        return json.loads(js)
    except Exception:
        js2 = re.sub(r",\s*([}\]])", r"\1", js)
        try:
            return json.loads(js2)
        except Exception:
            return None


def parse_gt(content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(content.get("answer"), dict):
        return content["answer"]
    gt_s = content.get("gt_answers")
    if isinstance(gt_s, str):
        try:
            return json.loads(gt_s)
        except Exception:
            js = extract_first_json_object(gt_s)
            if js:
                try:
                    return json.loads(js)
                except Exception:
                    return None
    return None

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = _norm(v)
    return v / n if n > eps else v * 0.0


def safe_list3(x, default=(0.0, 0.0, 0.0)):
    if isinstance(x, list) and len(x) == 3:
        try:
            return [float(x[0]), float(x[1]), float(x[2])]
        except Exception:
            return list(default)
    return list(default)


def rpy_to_quat_xyzw(rpy):
    return Rotation.from_euler("xyz", safe_list3(rpy)).as_quat()


def axis_world_from_rpy(axis_local, rpy):
    a = np.asarray(safe_list3(axis_local), dtype=float)
    if _norm(a) < 1e-12:
        return a * 0.0
    R = Rotation.from_euler("xyz", safe_list3(rpy)).as_matrix()
    return (R @ normalize(a)).astype(float)


def axis_angle_error_rad(a1, a2):
    if _norm(a1) < 1e-12 and _norm(a2) < 1e-12:
        return 0.0
    if _norm(a1) < 1e-12 or _norm(a2) < 1e-12:
        return math.pi / 2
    u, v = normalize(a1), normalize(a2)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(abs(dot)))


def quat_angle_distance_rad(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    n1, n2 = _norm(q1), _norm(q2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    q1, q2 = q1 / n1, q2 / n2
    dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
    return float(2.0 * np.arccos(abs(dot)))


def revolute_origin_axisline_distance(pos1, a1, pos2, a2):
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if _norm(a1) < 1e-12 or _norm(a2) < 1e-12:
        return float(np.linalg.norm(pos1 - pos2))
    a1, a2 = normalize(a1), normalize(a2)
    w = np.cross(a2, a1)
    if np.allclose(w, 0, atol=1e-6):
        w = np.cross(a2, [1, 0, 0]) if not np.allclose(a2, [1, 0, 0], atol=1e-6) else np.cross(a1, [0, 1, 0])
    wn = _norm(w)
    if wn < 1e-12:
        return float(np.linalg.norm(pos1 - pos2))
    return float(abs(np.dot(pos1 - pos2, w)) / wn)


def build_joint_map(joints):
    if not isinstance(joints, list):
        return {}
    m = {}
    for j in joints:
        if isinstance(j, dict):
            jid = j.get("id")
            if isinstance(jid, str) and jid:
                m[jid] = j
    return m


def aggregate(vals):
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std())}


def collect_json_files(root: str) -> List[str]:
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


def eval_root(root: str, max_samples: int = 0) -> Dict[str, Any]:
    angular_cutoff = 0.25    
    euclidean_cutoff = 0.05

    json_files = collect_json_files(root)
    if max_samples > 0:
        json_files = json_files[:max_samples]

    scanned = 0
    parsed = 0

    miou_list = []
    miou_by_category = {} 

    gt_joint_total = 0
    pred_joint_total = 0
    missing_joint_total = 0

    type_correct_list = []
    axis_err_list = []
    origin_xyz_err_list = []
    origin_orient_err_list = []
    origin_combo_err_list = []
    axis_within_list = []
    origin_within_list = []
    success_list = []

    per_sample_results = []

    for fp in json_files:
        scanned += 1
        try:
            content = json.load(open(fp, "r"))
        except Exception:
            continue

        miou_val = content.get("miou")
        if miou_val is not None:
            miou_list.append(float(miou_val))

            q = content.get("question", "")
            cat_match = re.search(r"articulated object\s+(\w[\w\s]*?)\s+consists", q, re.IGNORECASE)
            if cat_match:
                cat = cat_match.group(1).strip()
                miou_by_category.setdefault(cat, []).append(float(miou_val))

        gt = parse_gt(content)
        pred = parse_pred_answer(content.get("pred_answers", ""))

        if gt is None or pred is None:
            continue
        parsed += 1

        gt_map = build_joint_map(gt.get("joints", []))
        pred_map = build_joint_map(pred.get("joints", []))

        gt_ids = list(gt_map.keys())
        gt_joint_total += len(gt_ids)
        pred_joint_total += len(pred_map)

        sample_success = True
        for jid in gt_ids:
            gj = gt_map[jid]
            pj = pred_map.get(jid)

            if pj is None:
                missing_joint_total += 1
                type_correct_list.append(False)
                axis_within_list.append(False)
                origin_within_list.append(False)
                success_list.append(False)
                sample_success = False
                continue

            type_ok = str(pj.get("type", "")).lower() == str(gj.get("type", "")).lower()
            type_correct_list.append(bool(type_ok))

            prpy = (pj.get("origin") or {}).get("rpy", [0, 0, 0])
            grpy = (gj.get("origin") or {}).get("rpy", [0, 0, 0])

            a_pred_w = axis_world_from_rpy(pj.get("axis", [0, 0, 0]), prpy)
            a_gt_w = axis_world_from_rpy(gj.get("axis", [0, 0, 0]), grpy)

            axis_err = axis_angle_error_rad(a_gt_w, a_pred_w)
            axis_err_list.append(axis_err)
            axis_ok = axis_err <= angular_cutoff
            axis_within_list.append(bool(axis_ok))

            pxyz = np.asarray(safe_list3((pj.get("origin") or {}).get("xyz")), dtype=float)
            gxyz = np.asarray(safe_list3((gj.get("origin") or {}).get("xyz")), dtype=float)

            jt = str(gj.get("type", "")).lower()
            if jt == "revolute":
                origin_xyz_err = revolute_origin_axisline_distance(gxyz, a_gt_w, pxyz, a_pred_w)
            else:
                origin_xyz_err = float(np.linalg.norm(gxyz - pxyz))
            origin_xyz_err_list.append(origin_xyz_err)

            q_pred = rpy_to_quat_xyzw(prpy)
            q_gt = rpy_to_quat_xyzw(grpy)
            orient_err = quat_angle_distance_rad(q_gt, q_pred)
            origin_orient_err_list.append(orient_err)

            origin_combo = 0.5 * (origin_xyz_err + orient_err)
            origin_combo_err_list.append(origin_combo)
            origin_ok = origin_combo <= euclidean_cutoff
            origin_within_list.append(bool(origin_ok))

            success = bool(type_ok and axis_ok and origin_ok)
            success_list.append(success)
            if not success:
                sample_success = False

    def rate(bools):
        return float(np.mean(bools)) if bools else 0.0

    cat_summary = {}
    for cat, vals in sorted(miou_by_category.items()):
        a = np.array(vals)
        cat_summary[cat] = {
            "count": len(vals),
            "mean_miou": round(float(a.mean()), 4),
            "median_miou": round(float(np.median(a)), 4),
        }

    out = {
        "root": root,
        "num_json_scanned": scanned,
        "num_json_parsed_for_articulation": parsed,

        "miou": {
            "num_samples": len(miou_list),
            "mean": round(float(np.mean(miou_list)), 4) if miou_list else None,
            "median": round(float(np.median(miou_list)), 4) if miou_list else None,
            "std": round(float(np.std(miou_list)), 4) if miou_list else None,
            "by_category": cat_summary,
        },

        "articulation": {
            "num_gt_joints": gt_joint_total,
            "num_pred_joints": pred_joint_total,
            "missing_joints": missing_joint_total,
            "missing_rate": round(missing_joint_total / gt_joint_total, 4) if gt_joint_total else 0.0,

            "axis_angle_error_rad": aggregate(axis_err_list),
            "origin_xyz_error": aggregate(origin_xyz_err_list),
            "origin_orientation_error_rad": aggregate(origin_orient_err_list),
            "origin_combined_error": aggregate(origin_combo_err_list),

            "thresholds": {
                "angular_cutoff_rad": angular_cutoff,
                "euclidean_cutoff": euclidean_cutoff,
            },
            "rates": {
                "type_correct": round(rate(type_correct_list), 4),
                "axis_within": round(rate(axis_within_list), 4),
                "origin_within": round(rate(origin_within_list), 4),
                "success(type&axis&origin)": round(rate(success_list), 4),
            },
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Combined mIoU + articulation metrics evaluation")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory of inference output (containing per-object JSON files)")
    ap.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    res = eval_root(args.root, max_samples=args.max_samples)
    s = json.dumps(res, indent=2, ensure_ascii=False)
    print(s)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(s + "\n")
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()