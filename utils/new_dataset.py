import os
import glob
import json
import random
import hashlib
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

DEFAULT_POINT_TOKEN = "<point>"
DEFAULT_POINT_PATCH_TOKEN = "<pt_patch>"
DEFAULT_PT_START_TOKEN = "<pt_start>"
DEFAULT_PT_END_TOKEN = "<pt_end>"

LONG_QUESTION_LIST = DEFAULT_POINT_TOKEN + "\n" + "{sent}"
ANSWER_LIST = "{sent}"

PART_CATEGORIES = [
    "alarm_ring", "ball", "board", "bottle_body", "box_body",
    "bucket_body", "button", "camera_body", "cap", "cart_body",
    "caster", "chair_leg", "circle", "clock_body", "coffee_machine_body",
    "connector", "container", "cover_lid", "dishwasher_body", "dispenser_body",
    "display_base", "door", "door_frame", "drawer", "fan_frame",
    "fastener", "fastener_connector", "faucet_base", "foot_pad", "furniture_body",
    "glasses_body", "globe_frame", "hand", "handle", "head",
    "kettle_body", "key", "keyboard_base", "knife_body", "knob",
    "lamp_base", "laptop_base", "leg", "lens", "lever",
    "lid", "lighter_body", "lock", "microwave_body", "mouse_body",
    "nose", "oven_body", "pen_body", "phone_base", "portafilter",
    "pot_body", "pressing_lid", "printer_body", "pump_lid", "refrigerator_body",
    "remote_base", "rotation_bar", "rotation_blade", "rotation_body", "rotation_button",
    "rotation_container", "rotation_door", "rotation_handle", "rotation_lid", "rotation_screen",
    "rotation_slider", "rotation_tray", "rotation_window", "rotor", "safe_body",
    "screen", "seat", "shelf", "slider", "slot",
    "sphere", "spout", "stapler_base", "stapler_body", "steering_wheel",
    "stem", "suitcase_body", "switch", "switch_frame", "tilt_leg",
    "toaster_body", "toggle_button", "toilet_body", "translation_bar", "translation_blade",
    "translation_door", "translation_handle", "translation_lid", "translation_screen", "translation_tray",
    "translation_window", "trashcan_body", "usb_body", "usb_rotation", "washing_machine_body",
    "wheel", "window_frame"
]

PART2IDX = {n: i for i, n in enumerate(PART_CATEGORIES)}


def load_instruction_and_response(json_file: str):
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            with open(json_file, "r", encoding=encoding) as f:
                content = json.load(f)
            return content["question"], content["answer"], content
        except (UnicodeDecodeError, KeyError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Unable to parse JSON file: {json_file}")


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-12:
        return pc
    pc = pc / m
    return pc


def _stable_hash_to_unit(x: str, seed: int = 0) -> float:
    h = hashlib.md5((str(seed) + "_" + x).encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / float(16 ** 8)  # [0,1)
    return v

def _onehot_link_idx(code):
    """one-hot tuple → link index"""
    try:
        return list(code).index(1)
    except (ValueError, AttributeError):
        return 999

class URDFReasoningDatasetBack(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_ratio: float = 0.9,
        split_seed: int = 3407,
        max_samples: Optional[int] = None,
        num_points: int = 2048,
        sample_points: bool = True,
        answer_as_json_str: bool = True,  # True: json.dumps(answer)；False: str(answer)
    ):
        self.data_root = data_root
        self.split = split
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.max_samples = max_samples
        self.num_points = int(num_points)
        self.sample_points = bool(sample_points)
        self.answer_as_json_str = bool(answer_as_json_str)

        self.json_root = os.path.join(self.data_root, "json_questions")
        self.pc_root = os.path.join(self.data_root, "point_clouds")

        self.samples: List[Tuple[str, str, str]] = []  # (json_path, point_path, obj_id)
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid (json, point) pairs found under:\n"
                f"  json_root={self.json_root}\n"
                f"  pc_root={self.pc_root}"
            )

    def _keep_obj(self, obj_id: str) -> bool:
        if self.split == "all":
            return True
        u = _stable_hash_to_unit(obj_id, seed=self.split_seed)
        if self.split == "train":
            return u < self.train_ratio
        if self.split == "test":
            return u >= self.train_ratio
        raise ValueError(f"Unsupported split={self.split}. Use 'train'/'test'/'all'.")

    def _build_index(self):
        obj_dirs = sorted([d for d in os.listdir(self.json_root) if os.path.isdir(os.path.join(self.json_root, d))])
        for obj_id in obj_dirs:
            if not self._keep_obj(obj_id):
                continue

            jdir = os.path.join(self.json_root, obj_id)
            json_files = sorted(glob.glob(os.path.join(jdir, "*.json")))

            for jp in json_files:
                base = os.path.splitext(os.path.basename(jp))[0]  # 100045_0
                pp = os.path.join(self.pc_root, obj_id, base + ".txt")
                if not os.path.exists(pp):
                    continue
                self.samples.append((jp, pp, obj_id))

    def __len__(self) -> int:
        n = len(self.samples)
        return min(n, self.max_samples) if self.max_samples is not None else n

    def __getitem__(self, index: int):
        jp, pp, _obj_id = self.samples[index % len(self.samples)]

        question, answer, meta = load_instruction_and_response(jp)
        target_parts = self._determine_target_parts_from_meta(meta)

        coords, colors, part_names, inst_codes = self._load_point_txt(pp)

        if self.sample_points:
            coords, colors, part_names, inst_codes = self._sample_or_pad(
                coords, colors, part_names, inst_codes, self.num_points
            )

        normalized_coords = coords.T

        seg_mask, part_indices = self._build_segmentation_target(target_parts, part_names, inst_codes)
        self._check_segmentation(seg_mask)

        user_query = LONG_QUESTION_LIST.format(sent=question)
        if self.answer_as_json_str:
            model_response = json.dumps(answer, ensure_ascii=False)
        else:
            model_response = ANSWER_LIST.format(sent=answer)

        try:
            from ..model.llava import conversation as conversation_lib
        except Exception:
            from model.llava import conversation as conversation_lib

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], user_query)
        conv.append_message(conv.roles[1], model_response)
        conversation_text = conv.get_prompt()

        return (
            torch.from_numpy(normalized_coords).float(),
            torch.from_numpy(colors).float(),
            conversation_text,
            user_query,
            model_response,
            seg_mask,
            part_indices,
            jp,
        )

    @staticmethod
    def _determine_target_parts_from_meta(meta: Dict) -> List[str]:
        pc_map = meta.get("point_cloud", {})
        return [pc_map[k] for k in pc_map.keys() if k != "base"]

    @staticmethod
    def _load_point_txt(file_path: str):
        coords, colors, part_names, inst_codes = [], [], [], []

        with open(file_path, "r") as f:
            for line in f:
                t = line.strip().split()
                if len(t) < 8:
                    continue

                part = t[1]
                x, y, z = float(t[2]), float(t[3]), float(t[4])
                r, g, b = float(t[5]), float(t[6]), float(t[7])

                tail = tuple(int(float(v)) for v in t[8:]) if len(t) > 8 else tuple()

                coords.append([x, y, z])
                colors.append([r, g, b])
                part_names.append(part)
                inst_codes.append(tail)

        coords = np.asarray(coords, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        part_names = np.asarray(part_names, dtype=np.str_)
        inst_codes = np.asarray(inst_codes, dtype=object)

        return coords, colors, part_names, inst_codes

    @staticmethod
    def _sample_or_pad(coords, colors, part_names, inst_codes, n: int):
        N = coords.shape[0]
        if N == 0:
            raise RuntimeError("Empty point cloud.")

        if N >= n:
            choice = np.random.choice(N, n, replace=False)
        else:
            extra = np.random.choice(N, n - N, replace=True)
            choice = np.concatenate([np.arange(N), extra], axis=0)

        return coords[choice], colors[choice], part_names[choice], inst_codes[choice]

    @staticmethod
    def _build_segmentation_target(target_parts: List[str],
                                point_part_names: np.ndarray,
                                point_inst_codes: np.ndarray):
        from collections import defaultdict, Counter

        sem_to_counter = defaultdict(Counter)
        for sem, code in zip(point_part_names.tolist(), point_inst_codes.tolist()):
            if isinstance(code, list) or isinstance(code, np.ndarray):
                code = tuple(int(x) for x in code)
            elif not isinstance(code, tuple):
                code = tuple()
            sem_to_counter[sem][code] += 1

        sem_to_codes_sorted = {}
        for sem, c in sem_to_counter.items():
            sem_to_codes_sorted[sem] = sorted(c.keys(), key=_onehot_link_idx)

        sem_used = defaultdict(int)

        masks = []
        indices = []
        for sem_name in target_parts:
            if sem_name not in PART2IDX:
                raise ValueError(f"Unrecognized part name: {sem_name}")
            indices.append(PART2IDX[sem_name])

            codes_sorted = sem_to_codes_sorted.get(sem_name, [])
            use_i = sem_used[sem_name]
            sem_used[sem_name] += 1

            if len(codes_sorted) == 0 or codes_sorted[0] == tuple():
                mask = (point_part_names == sem_name).astype(np.float32)
            else:
                if use_i < len(codes_sorted):
                    chosen_code = codes_sorted[use_i]
                    inst_eq = np.array([tuple(c) == chosen_code for c in point_inst_codes.tolist()], dtype=bool)
                    mask = ((point_part_names == sem_name) & inst_eq).astype(np.float32)
                else:
                    mask = (point_part_names == sem_name).astype(np.float32)

            masks.append(mask)

        seg_mask = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32)  # [P,N]
        part_indices = torch.tensor(indices, dtype=torch.long)
        return seg_mask, part_indices
    
    @staticmethod
    def _check_segmentation(seg_mask: torch.Tensor, eps: float = 1e-6):
        per_point_sum = seg_mask.sum(dim=0)

        has_overlap = (per_point_sum > 1.0 + eps).any().item()
        has_hole    = (per_point_sum < 1.0 - eps).any().item()

        if has_overlap or has_hole:
            overlap_cnt = int((per_point_sum > 1.0 + eps).sum().item())
            hole_cnt    = int((per_point_sum < 1.0 - eps).sum().item())
            raise ValueError(
                f"[SegCheck Failed] overlap_points={overlap_cnt}, hole_points={hole_cnt}. "
                f"Need disjoint masks and full coverage."
            )


def collate_fn_back(
    batch,
    tokenizer=None,
    conv_type="llava_v1",
    use_mm_start_end=True,
    local_rank=-1,
    inference_mode=False,
    point_token_len: int = 1135, 
):
    try:
        from model.llava.mm_utils import tokenizer_point_token
        from model.llava.constants import IGNORE_INDEX
        from model.llava import conversation as conversation_lib
    except Exception:
        from ..model.llava.mm_utils import tokenizer_point_token
        from ..model.llava.constants import IGNORE_INDEX
        from ..model.llava import conversation as conversation_lib

    point_list, rgb_list = [], []
    conversation_list, questions_list, response_list = [], [], []
    segment_label_list, logist_label_list = [], []
    json_path = None

    for (points, rgb, conversations, questions, response, segment_label, logist_label, json_path_) in batch:
        point_list.append(points.to(torch.float32))
        rgb_list.append(rgb.to(torch.float32))
        conversation_list.append(conversations)
        questions_list.append(questions)
        response_list.append(response)
        segment_label_list.append(segment_label)
        logist_label_list.append(logist_label)
        json_path = json_path_

    replace_token_core = DEFAULT_POINT_PATCH_TOKEN * int(point_token_len)
    replace_token = DEFAULT_PT_START_TOKEN + replace_token_core + DEFAULT_PT_END_TOKEN

    if inference_mode:
        if use_mm_start_end:
            for i in range(len(questions_list)):
                questions_list[i] = questions_list[i].replace(DEFAULT_POINT_TOKEN, replace_token)
        return {
            "points": torch.stack(point_list, dim=0),
            "colors": torch.stack(rgb_list, dim=0),
            "questions": questions_list,
            "responses": response_list,
            "segment_label": segment_label_list,
            "logist_label": logist_label_list,
            "json_path": json_path,
        }

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            conversation_list[i] = conversation_list[i].replace(DEFAULT_POINT_TOKEN, replace_token)

    input_ids = [tokenizer_point_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    targets = input_ids.clone()

    conv = conversation_lib.default_conversation.copy()
    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou, tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    truncate_len = tokenizer.model_max_length - point_token_len

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]

    return {
        "points": torch.stack(point_list, dim=0),
        "colors": torch.stack(rgb_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "segment_label": segment_label_list,
        "logist_label": logist_label_list,
        "json_path": json_path,
    }