import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers import BitsAndBytesConfig, CLIPVisionModel
import sys
import os
import types

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from utils.loss import dice_loss
from .Uni3D.models.uni3d import create_uni3d
from .seg_decoder.decoder import PointCrossAttentionDecoder, PartSegmentationEmbHead

class LisaMetaModel:
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)
        self.config = config
        self.backbone3d_path = kwargs.get("backbone3d_path", None)
        self.initialize_lisa_modules(self.config)

    def build_backbone3d(self):
        if 'uni3d-b' in self.backbone3d_path:
            pc_model = 'eva02_base_patch14_448'
            pc_feat_dim = 768
        if 'uni3d-g' in self.backbone3d_path:
            pc_model = 'eva_giant_patch14_560'
            pc_feat_dim = 1408

        args = {
            "pc_model": pc_model,
            "pc_feat_dim": pc_feat_dim,
            "group_size": 64,
            "num_group": 512,
            "pc_encoder_dim": 512,
            "embed_dim": 1024,
            "patch_dropout": 0,
        }
        self.backbone3d_args = types.SimpleNamespace(**args)
        self.backbone3d = create_uni3d(self.backbone3d_args)
        if self.backbone3d_path is not None:
            sd = torch.load(self.backbone3d_path, map_location="cpu")['module']
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            self.backbone3d.load_state_dict(sd)

    def initialize_lisa_modules(self, config):
        self.build_backbone3d()
        for name, param in self.backbone3d.named_parameters():
            param.requires_grad = False


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.seg_hidden_dim = kwargs.pop("seg_hidden_dim")
        self.context_fusion = kwargs.pop("context_fusion")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        in_dim, out_dim = config.hidden_size, self.seg_hidden_dim
        text_fc = [
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.seg_emb_head = PartSegmentationEmbHead(
            embed_dim=self.model.backbone3d_args.pc_feat_dim,
            mlp=[out_dim, out_dim],
        )

        query_dim = out_dim if not self.context_fusion else out_dim * 2

        self.seg_decoder = PointCrossAttentionDecoder(
            query_dim=query_dim,
            point_feat_dim=out_dim,
            hidden_dim=out_dim, 
            num_heads=8,
            num_layers=3,
            mlp_ratio=2,
        )
        self.post_init()

    def get_visual_embs(self, points):
        xyz = points[:, :, :3].contiguous()
        color = points[:, :, 3:].contiguous()
        final_feat, centers, intermediates = self.model.backbone3d.point_encoder(
            xyz, color, return_intermediate=True,
        )
        H4, H8, H12 = intermediates
        pc_feat = self.seg_emb_head(xyz, centers, H4, H8, H12)
        return pc_feat, xyz

    def forward(self, **kwargs):
        if kwargs.get("labels", None) is None:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        points: torch.FloatTensor = None,
        colors: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_masks: torch.Tensor = None,
        segment_label: List[torch.FloatTensor] = None,
        logist_label: List[torch.FloatTensor] = None,
        seg_type_ids: List = None,
        return_lm_out: bool = False,
        **kwargs,
    ):
        if points.shape[-1] not in [3, 6]:
            points = points.permute(0, 2, 1)
        points = torch.cat([points, colors], dim=-1) 

        point_embeddings, xyz = self.get_visual_embs(points) 
        batch_size = point_embeddings.shape[0]

        output = super().forward(
            points=points,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )

        full_h = output.hidden_states[-1]
        full_h = self.text_hidden_fcs[0](full_h)

        B, T_full, _ = full_h.shape
        L_total = input_ids.shape[1]
        prefix_len = T_full - L_total

        seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)
        seg_token_mask = torch.cat([
            torch.zeros((B, prefix_len), dtype=torch.bool, device=input_ids.device),
            seg_token_mask,
            torch.zeros((B, 1), dtype=torch.bool, device=input_ids.device),
        ], dim=1)
        assert seg_token_mask.shape[1] == full_h.shape[1]
        last_hidden_state = full_h

        pred_embeddings = []
        for i in range(batch_size):
            mask_i = seg_token_mask[i]
            if mask_i.any():
                emb_i = last_hidden_state[i][mask_i]
                if self.context_fusion:
                    cat_emb = last_hidden_state[i][:-1, :][mask_i[1:]]
                    emb_i = torch.cat([cat_emb, emb_i], dim=-1)
            else:
                d = last_hidden_state.shape[-1]
                if self.context_fusion:
                    d *= 2
                emb_i = torch.zeros(0, d, device=last_hidden_state.device,
                                    dtype=last_hidden_state.dtype)
            pred_embeddings.append(emb_i)

        pred_segment = []
        seg_loss = 0.0
        num_valid_samples = 0

        for i in range(batch_size):
            if pred_embeddings[i].numel() == 0:
                continue

            h_seg = pred_embeddings[i]  
            point_feat = point_embeddings[i] 
            xyz_i = xyz[i]

            logits = self.seg_decoder(h_seg, point_feat, xyz=xyz_i)
            pred_mask = torch.sigmoid(logits)

            K_pred = pred_mask.shape[0]
            K_gt = segment_label[i].shape[0]

            if K_pred < K_gt:
                pad = torch.zeros(K_gt - K_pred, pred_mask.shape[1], 
                                device=pred_mask.device)
                logits_padded = torch.cat([logits, torch.full_like(pad, -10.0)], dim=0)
                gt_mask = segment_label[i]
            else:
                logits_padded = logits[:K_gt]
                gt_mask = segment_label[i]

            bce = F.binary_cross_entropy_with_logits(logits_padded, gt_mask)
            dice = dice_loss(torch.sigmoid(logits_padded), gt_mask, gt_mask.shape[0])

            seg_loss += self.bce_loss_weight * bce
            seg_loss += self.dice_loss_weight * dice

            pred_segment.append(pred_mask)
            num_valid_samples += 1

        if num_valid_samples > 0:
            seg_loss = seg_loss / num_valid_samples
        else:
            seg_loss = torch.tensor(0.0, device=points.device)

        ce_loss = output.loss
        total_loss = ce_loss + seg_loss

        if return_lm_out:
            lm_out = self.lm_head(output.hidden_states[-1])
            return total_loss, seg_loss, pred_segment, segment_label, lm_out
        return total_loss, seg_loss, pred_segment, segment_label


    @torch.no_grad()
    def evaluate(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        seg_type_ids=None,
        tokenizer=None,
        return_probs: bool = True,
    ):
        device = input_ids.device

        if points.shape[-1] not in (3, 6):
            points = points.permute(0, 2, 1).contiguous()
        if points.shape[-1] == 3:
            if colors is None:
                colors_in = torch.full_like(points[..., :3], 0.4, device=device)
            else:
                if colors.shape[-1] != 3:
                    colors = colors.permute(0, 2, 1).contiguous()
                colors_in = colors.to(device)
            points = torch.cat([points, colors_in], dim=-1)

        gen_out = self.generate(
            points=points,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
        output_ids = gen_out.sequences 

        attn = torch.ones_like(output_ids, device=device)
        fw_out = self(
            points=points,
            input_ids=output_ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        full_h = fw_out.hidden_states[-1]         
        full_h = self.text_hidden_fcs[0](full_h)    

        B, T_full, _ = full_h.shape
        L_total = output_ids.shape[1]
        prefix_len = T_full - L_total

        seg_token_mask = (output_ids[:, 1:] == self.seg_token_idx)
        seg_token_mask = torch.cat([
            torch.zeros((B, prefix_len), dtype=torch.bool, device=device),
            seg_token_mask,
            torch.zeros((B, 1), dtype=torch.bool, device=device),
        ], dim=1)

        point_embeddings, xyz = self.get_visual_embs(points)  

        pred_masks = []
        for i in range(B):
            mask_i = seg_token_mask[i] 
            if not mask_i.any():
                pred_masks.append(
                    torch.zeros((0, point_embeddings.shape[1]), device=device)
                )
                continue

            seg_pos = torch.nonzero(mask_i, as_tuple=False).squeeze(1) 
            h_seg = full_h[i, seg_pos, :] 

            if self.context_fusion:
                prev_pos = torch.clamp(seg_pos - 1, min=0)
                h_prev = full_h[i, prev_pos, :]
                h_seg = torch.cat([h_prev, h_seg], dim=-1) 

            logits = self.seg_decoder(h_seg, point_embeddings[i], xyz=xyz[i])
            probs = torch.sigmoid(logits) if return_probs else logits
            pred_masks.append(probs)

        return output_ids, pred_masks

    def _debug_palette_10(self):
        return [
            (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
            (255, 255,   0), (255,   0, 255), (  0, 255, 255),
            (255, 128,   0), (128,   0, 255), (  0, 255, 128),
            (255,   0, 128),
        ]


    @staticmethod
    def _write_ply_xyzrgb(path, xyz, rgb_uint8):
        import numpy as np
        xyz = np.asarray(xyz)
        rgb = np.asarray(rgb_uint8)
        assert xyz.ndim == 2 and xyz.shape[1] == 3
        assert rgb.ndim == 2 and rgb.shape[1] == 3
        assert xyz.shape[0] == rgb.shape[0]
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")

    @staticmethod
    def _topk_parts(mask_kn, kmax=10):
        K = int(mask_kn.shape[0])
        if K <= kmax:
            return torch.arange(K, device=mask_kn.device)
        area = mask_kn.sum(dim=1)
        _, idx = torch.topk(area, k=kmax, largest=True, sorted=True)
        return idx

    def _mask_to_labels(self, mask_kn, threshold=0.5, kmax=10):
        assert mask_kn.ndim == 2
        idx = self._topk_parts(mask_kn, kmax=kmax)
        m = mask_kn[idx]
        conf, lab = torch.max(m, dim=0)
        lab = lab.clone()
        lab[conf < threshold] = -1
        return lab, idx

    def _labels_to_rgb(self, labels_1d, palette, bg=(30, 30, 30)):
        import numpy as np
        labels = labels_1d.detach().cpu().numpy().astype(np.int32)
        rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
        rgb[:] = np.array(bg, dtype=np.uint8)
        for cid in range(len(palette)):
            rgb[labels == cid] = np.array(palette[cid], dtype=np.uint8)
        return rgb

    def _dump_seg_debug_ply(self, xyz_bn3, gt_kn, pred_kn, tag):
        import os
        os.makedirs(self.debug_ply_dir, exist_ok=True)
        palette = self._debug_palette_10()
        gt_lab, _ = self._mask_to_labels(gt_kn, threshold=0.5, kmax=10)
        pred_lab, _ = self._mask_to_labels(pred_kn, threshold=0.5, kmax=10)
        xyz = xyz_bn3.detach().cpu().numpy()
        gt_rgb = self._labels_to_rgb(gt_lab, palette)
        pred_rgb = self._labels_to_rgb(pred_lab, palette)
        self._write_ply_xyzrgb(os.path.join(self.debug_ply_dir, f"{tag}_gt.ply"), xyz, gt_rgb)
        self._write_ply_xyzrgb(os.path.join(self.debug_ply_dir, f"{tag}_pred.ply"), xyz, pred_rgb)
        diff = (gt_lab != pred_lab)
        diff_idx = torch.where(diff)[0]
        if diff_idx.numel() > 0:
            xyz_d = xyz[diff_idx.detach().cpu().numpy()]
            gt_rgb_d = gt_rgb[diff_idx.detach().cpu().numpy()]
            self._write_ply_xyzrgb(os.path.join(self.debug_ply_dir, f"{tag}_diff.ply"), xyz_d, gt_rgb_d)
        else:
            self._write_ply_xyzrgb(os.path.join(self.debug_ply_dir, f"{tag}_diff.ply"),
                                   xyz[:0], xyz[:0].astype("uint8"))