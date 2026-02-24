import os
import json
import re
import copy
import argparse
import logging
from functools import partial
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import logging
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from transformers import logging as hf_logging
import torch
import torch.utils.checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset

from model.UA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_point_token
from model.llava.constants import IGNORE_INDEX
from model.llava import conversation as conversation_lib
from utils.new_dataset import URDFReasoningDatasetBack as URDFReasoningDataset, collate_fn_back as collate_fn
from model.llava.constants import POINT_TOKEN_INDEX
from utils.mesh_convert_gpu import convert_link_objs_to_mesh_gpu

from tqdm import tqdm
import numpy as np
import glob
import shutil
import warnings
from pydantic.warnings import PydanticDeprecatedSince20

def compute_projected_token_length(prompt_token_num: int,
                                    with_ape: bool,
                                    with_local: bool,
                                    with_global: bool,
                                    pos_len = 512,
                                    local_len = 512,
                                    global_len = 16) -> int:
    extra = prompt_token_num if prompt_token_num > 0 else 0
    total = 0
    if with_ape:
        total += pos_len + extra
    if with_local:
        total += local_len + extra
    if with_global:
        total += global_len + extra
    return total

def _link_sort_key(name: str):
    if name.startswith('link_') and name[5:].isdigit():
        return (0, int(name[5:]))
    elif name == 'base':
        return (-1, 0)
    return (1, name)


def parse_answer_json(text: str) -> Optional[dict]:
    ast_idx = text.rfind('ASSISTANT:')
    if ast_idx >= 0:
        text = text[ast_idx + len('ASSISTANT:'):].strip()

    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = text[start:end]
    json_str = re.sub(r'\s*\[SEG\]\s*', '', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def get_ordered_link_semantics(parsed: dict) -> List[Tuple[str, str]]:
    links = parsed.get("links", {})
    sorted_names = sorted(
        [k for k in links.keys() if k != 'base'],
        key=_link_sort_key
    )
    return [(ln, links[ln].strip()) for ln in sorted_names]


def build_semantic_permutation(
    pred_semantics: List[str],
    gt_semantics: List[str],
) -> Optional[List[int]]:
    pred_sem_to_idx = {}
    for idx, sem in enumerate(pred_semantics):
        if sem in pred_sem_to_idx:
            return None
        pred_sem_to_idx[sem] = idx

    perm = []
    for gt_sem in gt_semantics:
        if gt_sem in pred_sem_to_idx:
            perm.append(pred_sem_to_idx[gt_sem])
        else:
            return None

    if len(set(perm)) != len(perm):
        return None

    return perm


def build_hungarian_permutation(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    threshold: float = 0.5,
) -> List[int]:
    K_pred = pred_masks.shape[0]
    K_gt = gt_masks.shape[0]

    iou_matrix = torch.zeros(K_pred, K_gt)
    for i in range(K_pred):
        pred_bin = (pred_masks[i] > threshold).float()
        for j in range(K_gt):
            gt_bin = (gt_masks[j] > threshold).float()
            inter = (pred_bin * gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum() - inter
            if union > 0:
                iou_matrix[i, j] = (inter / union).item()

    row_ind, col_ind = linear_sum_assignment(-iou_matrix.cpu().numpy())

    gt_to_pred = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    perm = [gt_to_pred.get(gi, -1) for gi in range(K_gt)]
    return perm


def align_pred_to_gt(
    pred_text: str,
    gt_text: str,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], str, Optional[dict], Optional[dict]]:
    K_pred = pred_masks.shape[0]
    K_gt = gt_masks.shape[0]

    if K_pred == 0 or K_gt == 0:
        return pred_masks, list(range(max(K_pred, K_gt))), "empty", None, None

    pred_parsed = parse_answer_json(pred_text)
    gt_parsed = parse_answer_json(gt_text)

    perm = None
    method = "identity"

    if pred_parsed is not None and gt_parsed is not None:
        pred_link_sems = get_ordered_link_semantics(pred_parsed)
        gt_link_sems = get_ordered_link_semantics(gt_parsed)

        if len(pred_link_sems) == K_pred and len(gt_link_sems) == K_gt:
            pred_sems = [sem for _, sem in pred_link_sems]
            gt_sems = [sem for _, sem in gt_link_sems]

            perm = build_semantic_permutation(pred_sems, gt_sems)
            if perm is not None:
                method = "semantic"

    if perm is None:
        perm = build_hungarian_permutation(pred_masks, gt_masks)
        method = "hungarian"

    is_identity = all(
        i < len(perm) and perm[i] == i
        for i in range(min(len(perm), K_gt))
    )
    if is_identity and method == "semantic":
        method = "identity"

    aligned = torch.zeros_like(gt_masks)
    for gt_i, pred_j in enumerate(perm):
        if 0 <= pred_j < K_pred:
            aligned[gt_i] = pred_masks[pred_j]

    return aligned, perm, method, pred_parsed, gt_parsed


def reorder_pred_json(
    pred_parsed: Optional[dict],
    gt_parsed: Optional[dict],
    perm: List[int],
) -> Optional[dict]:
    if pred_parsed is None:
        return None

    old_links = pred_parsed.get("links", {})
    old_joints = pred_parsed.get("joints", [])

    pred_link_names = sorted(
        [k for k in old_links.keys() if k != 'base'],
        key=_link_sort_key
    )

    pred_link_to_joint = {}
    for j in old_joints:
        child = j.get("child", "")
        pred_link_to_joint[child] = j

    K_gt = len(perm)
    new_links = OrderedDict()
    new_joints = []

    for gt_i in range(K_gt):
        pred_j = perm[gt_i]
        new_link_name = f"link_{gt_i}"

        if 0 <= pred_j < len(pred_link_names):
            old_link_name = pred_link_names[pred_j]
            semantic = old_links.get(old_link_name, f"part_{gt_i}")
            new_links[new_link_name] = semantic

            old_joint = pred_link_to_joint.get(old_link_name)
            if old_joint is not None:
                new_joint = dict(old_joint)
                new_joint["id"] = f"joint_{gt_i}"
                new_joint["child"] = new_link_name
                new_joints.append(new_joint)
        else:
            new_links[new_link_name] = f"missing_{gt_i}"

    result = dict(pred_parsed)
    result["links"] = dict(new_links)
    result["joints"] = new_joints
    return result


import math
import numpy as np
from scipy.spatial.transform import Rotation

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
    return (R @ normalize_np(a)).astype(float)

def axis_angle_error_rad(a1, a2):
    if _norm(a1) < 1e-12 and _norm(a2) < 1e-12:
        return 0.0
    if _norm(a1) < 1e-12 or _norm(a2) < 1e-12:
        return math.pi / 2
    u, v = normalize_np(a1), normalize_np(a2)
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
    a1, a2 = normalize_np(a1), normalize_np(a2)
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

def compute_aa_counts_from_articulation(pred_art: Optional[dict], gt_art: Optional[dict],
                                       angular_cutoff: float = 0.25,
                                       euclidean_cutoff: float = 0.05) -> Dict[str, int]:
    out = dict(
        gt_joint_total=0,
        matched_joint=0,
        missing_joint=0,
        type_correct=0,
        axis_within=0,
        origin_within=0,
        success=0,
        axis_err_sum=0,
        origin_combo_sum=0,
    )
    if gt_art is None or pred_art is None:
        return out

    gt_map = build_joint_map(gt_art.get("joints", []))
    pred_map = build_joint_map(pred_art.get("joints", []))
    gt_ids = list(gt_map.keys())
    out["gt_joint_total"] = len(gt_ids)

    for jid in gt_ids:
        gj = gt_map[jid]
        pj = pred_map.get(jid)

        if pj is None:
            out["missing_joint"] += 1
            continue
        
        out["matched_joint"] += 1

        type_ok = str(pj.get("type", "")).lower() == str(gj.get("type", "")).lower()
        if str(pj.get("type", "")).lower() in str(gj.get("type", "")).lower():
            type_ok = True
        if type_ok:
            out["type_correct"] += 1

        prpy = (pj.get("origin") or {}).get("rpy", [0, 0, 0])
        grpy = (gj.get("origin") or {}).get("rpy", [0, 0, 0])

        a_pred_w = axis_world_from_rpy(pj.get("axis", [0, 0, 0]), prpy)
        a_gt_w = axis_world_from_rpy(gj.get("axis", [0, 0, 0]), grpy)

        axis_err = axis_angle_error_rad(a_gt_w, a_pred_w)
        out["axis_err_sum"] += float(axis_err)
        axis_ok = axis_err <= angular_cutoff
        if axis_ok:
            out["axis_within"] += 1

        pxyz = np.asarray(safe_list3((pj.get("origin") or {}).get("xyz")), dtype=float)
        gxyz = np.asarray(safe_list3((gj.get("origin") or {}).get("xyz")), dtype=float)

        jt = str(gj.get("type", "")).lower()
        if jt == "revolute":
            origin_xyz_err = revolute_origin_axisline_distance(gxyz, a_gt_w, pxyz, a_pred_w)
        else:
            origin_xyz_err = float(np.linalg.norm(gxyz - pxyz))

        q_pred = rpy_to_quat_xyzw(prpy)
        q_gt = rpy_to_quat_xyzw(grpy)
        orient_err = quat_angle_distance_rad(q_gt, q_pred)

        origin_combo = 0.5 * (origin_xyz_err + orient_err)
        out["origin_combo_sum"] += float(origin_combo)
        origin_ok = origin_combo <= euclidean_cutoff
        if origin_ok:
            out["origin_within"] += 1

        if type_ok and axis_ok and origin_ok:
            out["success"] += 1

    return out


def _read_obj_points(obj_path: str) -> np.ndarray:  
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (ValueError, IndexError):
                    pass
    return np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)


def _save_mesh_as_obj(mesh, obj_path: str) -> None:
    verts   = np.asarray(mesh.vertices,       dtype=np.float64)
    faces   = np.asarray(mesh.triangles,      dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    has_normals = (len(normals) == len(verts) and len(normals) > 0)

    with open(obj_path, 'w') as f:
        f.write("# Mesh reconstructed from point cloud\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_normals:
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for tri in faces:
            i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if has_normals:
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            else:
                f.write(f"f {i0} {i1} {i2}\n")


def _points_to_mesh_open3d(points_np: np.ndarray):
    try:
        import open3d as o3d
    except ImportError:
        return _scipy_convex_hull_to_o3d_like(points_np)

    n = len(points_np)
    if n < 4:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))

    if n >= 20:
        nb = min(16, n // 3)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=2.5)

    pts_clean = np.asarray(pcd.points)
    n2 = len(pts_clean)
    if n2 < 4:
        return None

    k_norm = min(30, max(5, n2 - 1))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_norm)
    )
    if n2 >= 10:
        pcd.orient_normals_consistent_tangent_plane(k=min(15, n2 - 1))

    mesh = None

    if n2 >= 80:
        try:
            m_p, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=6,
                width=0,
                scale=1.1,
                linear_fit=False,
            )
            dens_arr = np.asarray(densities)
            thresh = np.quantile(dens_arr, 0.05)
            m_p.remove_vertices_by_mask(dens_arr < thresh)
            m_p.remove_degenerate_triangles()
            m_p.remove_unreferenced_vertices()

            tri_clusters, n_tris_per_cluster, _ = m_p.cluster_connected_triangles()
            if len(n_tris_per_cluster) > 1:
                largest = int(np.argmax(n_tris_per_cluster))
                m_p.remove_triangles_by_mask(np.asarray(tri_clusters) != largest)
                m_p.remove_unreferenced_vertices()

            if len(np.asarray(m_p.triangles)) >= 20:
                mesh = m_p

        except Exception:
            pass

    if mesh is None:
        try:
            from scipy.spatial import KDTree
            pts_arr = np.asarray(pcd.points)
            k_nn = min(6, n2 - 1)
            tree = KDTree(pts_arr)
            dists, _ = tree.query(pts_arr, k=k_nn + 1) 
            avg_nn = float(dists[:, 1:].mean())

            best_mesh_a = None
            best_n_tri = 0
            for alpha_mult in (3.5, 5.0, 2.0):
                alpha = avg_nn * alpha_mult
                m_a = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                m_a.remove_degenerate_triangles()
                m_a.remove_unreferenced_vertices()
                n_tri = len(np.asarray(m_a.triangles))
                if n_tri > best_n_tri:
                    best_n_tri = n_tri
                    best_mesh_a = m_a

            if best_mesh_a is not None and best_n_tri >= 4:
                tri_clusters, n_tris_per_cluster, _ = best_mesh_a.cluster_connected_triangles()
                if len(n_tris_per_cluster) > 1:
                    largest = int(np.argmax(n_tris_per_cluster))
                    best_mesh_a.remove_triangles_by_mask(
                        np.asarray(tri_clusters) != largest
                    )
                    best_mesh_a.remove_unreferenced_vertices()
                mesh = best_mesh_a

        except Exception:
            pass 

    if mesh is None or len(np.asarray(mesh.triangles)) < 4:
        try:
            hull, _ = pcd.compute_convex_hull()
            mesh = hull
        except Exception:
            pass

    if mesh is not None and len(np.asarray(mesh.triangles)) > 0:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()

    return mesh


def _scipy_convex_hull_to_o3d_like(points_np: np.ndarray):
    from scipy.spatial import ConvexHull

    class _SimpleMesh:
        def __init__(self, verts, faces, normals=None):
            self.vertices      = verts
            self.triangles     = faces
            self.vertex_normals = normals if normals is not None else np.zeros_like(verts)

    if len(points_np) < 4:
        return None
    try:
        hull = ConvexHull(points_np.astype(np.float64))
        verts = hull.points
        faces = hull.simplices
        return _SimpleMesh(verts, faces)
    except Exception:
        return None


def _save_mesh_as_obj_generic(mesh, obj_path: str) -> None:
    verts   = np.asarray(mesh.vertices)
    faces   = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else np.array([])
    has_normals = (len(normals) == len(verts) and len(normals) > 0)

    with open(obj_path, 'w') as f:
        f.write("# Mesh reconstructed from point cloud\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_normals:
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for tri in faces:
            i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if has_normals:
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            else:
                f.write(f"f {i0} {i1} {i2}\n")

class LISADataModule(pl.LightningDataModule):
    def __init__(self, model_args, data_args, training_args, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.predict_type = data_args.predict_type
        self.point_cloud_token_len = compute_projected_token_length(
            prompt_token_num=model_args.prompt_token_num,
            with_ape=model_args.with_ape,
            with_local=model_args.with_local,
            with_global=model_args.with_global,
        )

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = URDFReasoningDataset(data_root=self.data_args.data_root,
                                                    split="train",
                                                    train_ratio=0.9,
                                                    max_samples=self.data_args.max_samples,
                                                    )
            self.val_dataset = URDFReasoningDataset(data_root=self.data_args.data_root,
                                                split="test",
                                                train_ratio=0.9,
                                                max_samples=self.data_args.max_samples,
                                                )
            self.test_dataset = self.val_dataset
        elif stage == 'test':
            self.test_dataset = URDFReasoningDataset(data_root=self.data_args.data_root,
                                                split="all",
                                                train_ratio=0.9,
                                                max_samples=self.data_args.max_samples,
                                                )
        self.pin_memory = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.training_args.dataloader_num_workers,
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                use_mm_start_end=self.model_args.mm_use_pt_start_end,
                local_rank=self.trainer.local_rank,
                point_token_len=self.point_cloud_token_len,
            ),
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.training_args.dataloader_num_workers,
            collate_fn=partial(
                collate_fn,
                tokenizer=self.tokenizer,
                use_mm_start_end=self.model_args.mm_use_pt_start_end,
                local_rank=self.trainer.local_rank,
                point_token_len=self.point_cloud_token_len,
            ),
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.training_args.dataloader_num_workers,
            collate_fn=partial(
                collate_fn, 
                tokenizer=self.tokenizer,
                use_mm_start_end=self.model_args.mm_use_pt_start_end,
                point_token_len=self.point_cloud_token_len,
                inference_mode=True
            ),
        )


class LISALightningModule(pl.LightningModule):
    def __init__(self, model_args, data_args, training_args, tokenizer, load_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters({
            "model_args": model_args.__dict__,
            "data_args": data_args.__dict__,
            "training_args": training_args.__dict__,
        })
        self.load_ckpt_path = load_ckpt_path 
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        bnb_config = None
        if training_args.bits in [4, 8]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )

        hf_logging.set_verbosity_error()

        self.model, _ = LISAForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            output_loading_info=True,
            seg_token_idx=self.seg_token_idx,
            cache_dir=training_args.cache_dir,
            quantization_config=bnb_config,
            device_map={"": "cuda"} if bnb_config else None,
            use_mm_start_end=model_args.mm_use_pt_start_end,
            vision_tower=model_args.vision_tower,
            ce_loss_weight=1.0,
            dice_loss_weight=model_args.dice_loss_weight,
            bce_loss_weight=model_args.bce_loss_weight,
            seg_hidden_dim=model_args.seg_hidden_dim,
            context_fusion=model_args.context_fusion,
            backbone3d_path=model_args.backbone3d_path,
        )

        hf_logging.set_verbosity_warning()

        if training_args.lora_enable:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            def find_linear_layers(model, target_modules):
                cls = torch.nn.Linear
                names = set()
                for name, module in model.named_modules():
                    if any(skip in name for skip in ["backbone3d", "vision_tower", "mm_projector", "text_hidden_fcs", "projection", "seg_decoder"]):
                        continue
                    if isinstance(module, cls) and any(t in name for t in target_modules):
                        names.add(name)
                return sorted(names)

            target_modules = find_linear_layers(self.model, lora_target_modules)
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()

        self.model.config.use_cache = False
        self.model.config.mm_use_pt_start_end = model_args.mm_use_pt_start_end
        self.model.config.mm_use_pt_patch_token = model_args.mm_use_pt_patch_token
        self.model.config.with_color = model_args.with_color
        self.model.config.sample_points_num = data_args.sample_points_num

        self.model.initialize_vision_tokenizer(model_args, tokenizer=self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))

        for n, p in self.model.named_parameters():
            if any(x in n for x in [
                "lm_head", "embed_tokens", "text_hidden_fcs", "seg_decoder", "seg_emb_head"
            ]):
                p.requires_grad = True

        for n, p in self.model.named_parameters():
            if any(x in n for x in [
                "mm_projector", "vision_tower", "backbone3d"
            ]):
                p.requires_grad = False
                
        for n, p in self.model.named_parameters():
            if "seg_emb_head" in n:
                p.requires_grad = True
                
        for m in self.model.seg_emb_head.propagation.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        total_loss, seg_loss, _, _ = self.model(**batch)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_llm_loss", total_loss-seg_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
        total_loss, seg_loss, _, _ = self.model(**batch)
        self.log("val_loss", total_loss, on_epoch=True, sync_dist=True)
        self.log("val_seg_loss", seg_loss, on_epoch=True, sync_dist=True)
        self.log("val_llm_loss", total_loss-seg_loss, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.training_args.warmup_ratio)
        decay_steps = total_steps - warmup_steps

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.training_args.learning_rate * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }}

    def lr_scheduler_step(self, scheduler, optimizer, metric):
        scheduler.step()

    def test_step(self, batch, batch_idx):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()
        points, colors, questions, gt_answers, seg_mask, part_indices, json_path = batch.values()

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], "")
        formatted_prompt = conv.get_prompt()

        json_path = json_path[0] if isinstance(json_path, list) else json_path
        input_ids = tokenizer_point_token(formatted_prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids, pred_masks = self.model.evaluate(
                points,
                colors,
                input_ids,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
                seg_type_ids=part_indices[0].tolist(),
            )

        output_ids = output_ids[0]
        output_ids = output_ids[output_ids != POINT_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        aligned_pred_masks, perm, method, pred_parsed, gt_parsed = align_pred_to_gt(
            pred_text=text_output,
            gt_text=gt_answers[0],
            pred_masks=pred_masks[0],
            gt_masks=seg_mask[0],
        )

        reordered_pred = reorder_pred_json(pred_parsed, gt_parsed, perm)

        rank = getattr(self.trainer, "global_rank", 0)

        timestamp = "test"
        inference_results_dir = self.training_args.output_dir
        run_output_dir = os.path.join(inference_results_dir, timestamp)

        seg_output_dir = os.path.join(run_output_dir, "seg_result")
        ply_output_dir = os.path.join(run_output_dir, "seg_visualization")

        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(seg_output_dir, exist_ok=True)
        os.makedirs(ply_output_dir, exist_ok=True)

        base_name = os.path.basename(json_path).replace(".json", "")
        obj_id = json_path.split("/")[-2]

        sample_dir = os.path.join(run_output_dir, obj_id)
        os.makedirs(sample_dir, exist_ok=True)

        save_path = os.path.join(sample_dir, f"{base_name}.json")
        seg_save_path = os.path.join(seg_output_dir, f"{base_name}_seg.txt")

        ply_gt_fused_path = os.path.join(ply_output_dir, f"{base_name}_gt_fused.ply")
        ply_pred_fused_path = os.path.join(ply_output_dir, f"{base_name}_pred_fused.ply")

        pts0 = points[0]
        gt0 = seg_mask[0]
        pred0 = aligned_pred_masks

        if pred0.numel() > 0 and gt0.numel() > 0:
            self._save_fused_seg_visualization(
                pts0, gt0, pred0,
                ply_gt_fused_path, ply_pred_fused_path
            )
        try:
            org_json = json.load(open(json_path, "r"))
        except Exception as e:
            org_json = {}
            org_json["json_load_error"] = str(e)
            org_json["json_path"] = json_path

        org_json["pred_answers_raw"] = text_output

        if reordered_pred is not None:
            org_json["pred_answers"] = json.dumps(reordered_pred, ensure_ascii=False)
        else:
            org_json["pred_answers"] = text_output

        org_json["gt_answers"] = gt_answers[0]
        org_json["alignment"] = {"method": method, "permutation": perm}

        org_json["seg_result_path"] = seg_save_path
        org_json["seg_visualization"] = {
            "gt_fused_ply": ply_gt_fused_path,
            "pred_fused_ply": ply_pred_fused_path,
        }

        org_json["ddp_rank"] = int(rank)

        with open(save_path, "w") as f:
            json.dump(org_json, f, indent=4, ensure_ascii=False)

        try:
            params_dir = os.path.join(run_output_dir, "articulation_params")
            os.makedirs(params_dir, exist_ok=True)

            from scripts.extract_articulation_params import parse_pred, parse_gt, clean_seg_tokens

            pred_art = None
            if reordered_pred is not None:
                pred_art = clean_seg_tokens(reordered_pred)
            else:
                pred_art = parse_pred(org_json)

            if pred_art is not None:
                pred_out = {
                    "sample_id": base_name,
                    "alignment": org_json.get("alignment"),
                    "normalize": org_json.get("normalize"),
                    "articulation": pred_art,
                    "ddp_rank": int(rank),
                }
                with open(os.path.join(params_dir, f"{base_name}_pred.json"), "w") as f:
                    json.dump(pred_out, f, indent=2, ensure_ascii=False)

            gt_art = parse_gt(org_json)
            if gt_art is not None:
                gt_out = {
                    "sample_id": base_name,
                    "normalize": org_json.get("normalize"),
                    "articulation": gt_art,
                    "ddp_rank": int(rank),
                }
                with open(os.path.join(params_dir, f"{base_name}_gt.json"), "w") as f:
                    json.dump(gt_out, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[rank{rank}] articulation_params write failed: {e}")
            
        try:
            do_reconstruct = True
            if do_reconstruct:
                from scripts.reconstruct_urdf import reconstruct_single
                out_dir = os.path.join(run_output_dir, "reconstructed", f"{base_name}")
                reconstruct_single(
                    content=org_json,
                    sample_id=f"{base_name}",
                    out_dir=out_dir,
                    mode="both",
                    denormalize=True,
                )

                try:
                    convert_link_objs_to_mesh_gpu(out_dir, verbose=True,
                                                  fast_mode=True,
                                                  max_workers=4,
                                                  device=self.device)
                except Exception as e_mesh:
                    print(f"[rank{rank}] convert_link_objs_to_mesh failed: {e_mesh}")

        except Exception as e:
            print(f"[rank{rank}] reconstruct_single failed: {e}")

    def _save_fused_seg_visualization(self, points, gt_mask, pred_mask, 
                                 gt_path, pred_path):
        if points.shape[0] == 3:
            points = points.permute(1, 0)
        
        xyz = points.detach().cpu().numpy()
        palette = self.model._debug_palette_10()
        
        gt_labels, _ = self.model._mask_to_labels(gt_mask, threshold=0.5)
        pred_labels, _ = self.model._mask_to_labels(pred_mask, threshold=0.5)

        gt_rgb = self.model._labels_to_rgb(gt_labels, palette)
        pred_rgb = self.model._labels_to_rgb(pred_labels, palette)
        
        self.model._write_ply_xyzrgb(gt_path, xyz, gt_rgb)
        self.model._write_ply_xyzrgb(pred_path, xyz, pred_rgb)
        


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    backbone3d_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_path: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_pt_start_end: bool = field(default=False)
    mm_use_pt_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    prompt_token_num: int = field(default=1)
    with_ape: bool = field(default=True)
    with_local: bool = field(default=True)
    with_global: bool = field(default=True)
    with_color: bool = field(default=True)
    bce_loss_weight: float = field(default=1.0)
    dice_loss_weight: float = field(default=1.0)
    seg_hidden_dim: int = field(default=512)
    context_fusion: bool = field(default=True)

@dataclass
class DataArguments:
    data_root: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    point_folder: Optional[str] = field(default=None)
    sample_points_num: int = field(default=4096)
    occlusion: bool = field(default=False)
    predict_type: str = field(default="seg")
    max_samples: int = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    auto_resume: bool = field(default=True)
    resume: str = ""
    debug: bool = field(default=False)
    train_mask_decoder: bool = field(default=False)
    deepspeed: bool = field(default=False)
    warmup_ratio: float = 0.3
    do_eval: bool = field(default=False)
    load_ckpt_path: str = field(default=None)
    debug_port: int = field(default=5678)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    pl.seed_everything(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")

    model = LISALightningModule(model_args, data_args, training_args, tokenizer, load_ckpt_path=training_args.load_ckpt_path)
    datamodule = LISADataModule(model_args, data_args, training_args, tokenizer)

    logger = TensorBoardLogger(save_dir=training_args.output_dir, name="logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(training_args.output_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
        save_weights_only=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=training_args.output_dir,
        max_epochs=training_args.num_train_epochs,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        precision="bf16",
        devices="auto",
        accelerator="gpu",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()