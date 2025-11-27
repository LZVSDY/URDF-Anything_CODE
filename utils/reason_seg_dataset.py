

import numpy as np
import torch
import torch.nn.functional as F

import os
try:
    from ..model.llava import conversation as conversation_lib
except:
    import os
    import sys
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_point_token
    from model.llava.constants import IGNORE_INDEX
import transformers
import glob
import json
import random
from torch.utils.data import DataLoader
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

def read_path_list(file_path: str) -> list[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise FileNotFoundError(f"Failed to read path list from {file_path}: {e}")

def load_instruction_and_response(json_file: str):
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            with open(json_file, "r", encoding=encoding) as f:
                content = json.load(f)
            return content["question"], content["answer"]
        except (UnicodeDecodeError, KeyError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Unable to parse JSON file: {json_file}")


def pc_normalize(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


class URDFReasoningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str = None,
        split: str = "train",
        task_mode: str = "all_parameters",
        max_samples: int | None = None,
    ):
        self.data_root = data_root
        self.split = split
        self.task_mode = task_mode
        self.max_samples = max_samples

        list_dir = os.path.join(self.data_root, "train_test_txt")
        json_index_file = os.path.join(list_dir, f"json_{split}_{task_mode}.txt")
        point_index_file = os.path.join(list_dir, f"point_{split}_{task_mode}.txt")

        self.json_files = read_path_list(json_index_file)
        self.point_files = read_path_list(point_index_file)

        assert len(self.json_files) == len(self.point_files), \
            f"Mismatch between JSON ({len(self.json_files)}) and point ({len(self.point_files)}) file counts."

        self._total_items = len(self.json_files)

    def __len__(self) -> int:
        return self.max_samples if self.max_samples is not None else self._total_items

    def __getitem__(self, index: int):

        idx = index % self._total_items
        json_path = self.json_files[idx]
        point_path = self.point_files[idx]

        target_parts = self._determine_target_parts(json_path)

        coords, colors, seg_matrix = self._load_point_data(point_path)
        normalized_coords = pc_normalize(coords).T 

        seg_mask, part_indices = self._build_segmentation_target(target_parts, seg_matrix)

        question, answer = load_instruction_and_response(json_path)
        user_query = LONG_QUESTION_LIST.format(sent=question)
        model_response = ANSWER_LIST.format(sent=answer)

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
            json_path,
        )

    def _determine_target_parts(self, json_path: str) -> list[str]:
        with open(json_path, "r") as f:
            meta = json.load(f)
        return [
            meta["point_cloud"][key]
            for key in meta["point_cloud"]
            if key != "base"
        ]

    def _load_point_data(self, file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()

        parsed = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) < 9:
                continue
            values = [float(x) for x in tokens[2:]]
            parsed.append(values)

        data = np.array(parsed, dtype=np.float32)
        return data[:, :3], data[:, 3:6], data[:, 6:]

    def _build_segmentation_target(self, part_names: list[str], seg_tensor: np.ndarray):
        masks = []
        indices = []
        for name in part_names:
            if name not in PART_CATEGORIES:
                raise ValueError(f"Unrecognized part name: {name}")
            idx = PART_CATEGORIES.index(name)
            masks.append(seg_tensor[:, idx])
            indices.append(idx)
        return torch.tensor(np.stack(masks), dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, inference_mode=False
):
    point_list = []
    conversation_list = []
    questions_list = []
    response_list = []
    offset_list = [0]
    cnt = 0
    segment_label_list = []
    logist_label_list =[]
    rgb_list = []
    json_path = None
    token_lengths = []

    for (points, rgb, conversations,questions,response,segment_label,logist_label,json_path_) in batch:
        point_list.append(points.to(torch.float32))
        conversation_list.append(conversations)
        questions_list.append(questions)
        response_list.append(response)
        cnt += len(conversations)
        offset_list.append(cnt)
        segment_label_list.append(segment_label)
        logist_label_list.append(logist_label)
        rgb_list.append(rgb.to(torch.float32))
        json_path = json_path_

    if inference_mode:
        if use_mm_start_end:
            # replace <image> token
            for i in range(len(questions_list)):
                replace_token = (
                        DEFAULT_PT_START_TOKEN + replace_token + DEFAULT_PT_END_TOKEN
                    )
                questions_list[i] = questions_list[i].replace(
                        DEFAULT_POINT_TOKEN, replace_token
                    )
        return {
            "points": torch.stack(point_list, dim=0),
            "rgb": torch.stack(rgb_list, dim=0),
            "questions": questions_list,
            "responses": response_list,
            "segment_label": segment_label_list,
            "logist_label": logist_label_list,
            "json_path": json_path
        }

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = (
                    DEFAULT_PT_START_TOKEN + replace_token + DEFAULT_PT_END_TOKEN
                )
            
            
            conversation_list[i] = conversation_list[i].replace(
                    DEFAULT_POINT_TOKEN, replace_token
                )
    # import pdb;pdb.set_trace()
    input_ids = [
        tokenizer_point_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
                ]
   
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    token_lengths.append(input_ids.shape[1])
    # input_ids = torch.stack(input_ids, dim=0)


    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()


    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou, tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len
    truncate_len = tokenizer.model_max_length - 1135

    if input_ids.shape[1] > truncate_len:
        input_ids = input_ids[:, :truncate_len]
        targets = targets[:, :truncate_len]
        attention_masks = attention_masks[:, :truncate_len]
    return {
            "points":torch.stack(point_list, dim=0),
            "rgb":torch.stack(rgb_list, dim=0),
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            "segment_label":segment_label_list,
            "logist_label":logist_label_list,
            "json_path":json_path
        }

