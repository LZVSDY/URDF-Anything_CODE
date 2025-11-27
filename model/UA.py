from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
import sys
import os
import types
# from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
#                          DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from utils.loss import dice_loss
from .Uni3D.models.uni3d import create_uni3d
from .seg_decoder.decoder import PointCrossAttentionDecoder
from utils.pointnet_util import PointNetFeaturePropagation


class PartSegmentationEmbHead(nn.Module):
    def __init__(self, embed_dim=512, mlp=[512, 512]):
        super().__init__()
        in_channel = 3 * embed_dim
        self.propagation = PointNetFeaturePropagation(in_channel, mlp)

    def forward(self, xyz, centers, H4, H8, H12):
        B, N, _ = xyz.shape
        G = centers.shape[1]
        fused = torch.cat([H4, H8, H12], dim=-1)
        
        fused = fused.permute(0, 2, 1) 
        centers = centers.permute(0, 2, 1)
        xyz = xyz.permute(0, 2, 1)

        point_features = self.propagation(xyz, centers, None, fused)
        point_features = point_features.permute(0, 2, 1)
        return point_features



class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # if not hasattr(self.config, "train_mask_decoder"):
        #     self.config.train_mask_decoder = kwargs["train_mask_decoder"]
        #     self.config.out_dim = kwargs["out_dim"]
        #     self.vision_pretrained = kwargs.get("vision_pretrained", None)
        # else:
        #     self.vision_pretrained = kwargs.get("vision_pretrained", None)
        #     self.initialize_lisa_modules(self.config)
        self.backbone3d_path = kwargs.get("backbone3d_path", None)
        self.initialize_lisa_modules(self.config)

    def build_backbone3d(self):
        args = {"pc_model": 'eva02_base_patch14_448',
            "pc_feat_dim": 768,
            "group_size": 32,
            "num_group": 512,
            "pc_encoder_dim": 512,
            "embed_dim": 1024,
            "patch_dropout": 0
        }
        self.backbone3d_args = types.SimpleNamespace(**args)
        self.backbone3d = create_uni3d(self.backbone3d_args)
        if self.backbone3d_path is not None:
            sd = torch.load(self.backbone3d_path, map_location="cpu")['module']
            distributed = False
            if not distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}           
            self.backbone3d.load_state_dict(sd)


    def initialize_lisa_modules(self, config):
        self.build_backbone3d()
        for name, param in self.backbone3d.named_parameters(): 
            param.requires_grad = False



class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
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
    def __init__(
        self,
        config,
        **kwargs,
    ):
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
        self.seg_emb_head = PartSegmentationEmbHead(embed_dim=self.model.backbone3d_args.pc_feat_dim, mlp=[out_dim, out_dim])
        query_dim = out_dim if not self.context_fusion else out_dim * 2
        self.seg_decoder = PointCrossAttentionDecoder(query_dim=query_dim, point_feat_dim=out_dim)

        self.post_init()

    def get_visual_embs(self, points):
        xyz = points[:, :, :3].contiguous()
        color = points[:, :, 3:].contiguous()        
        final_feat, centers, intermediates = self.model.backbone3d.point_encoder(xyz, color, return_intermediate=True)
        H4, H8, H12 = intermediates
        pc_feat = self.seg_emb_head(xyz, centers, H4, H8, H12)
        return pc_feat

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
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
        colors = torch.full_like(points[..., :3], 0.4).to(points.device) if colors is None else colors
        points = torch.cat([points, colors], dim=-1)
        point_embeddings = self.get_visual_embs(points)
        batch_size = point_embeddings.shape[0]

        output = super().forward(
            points=points,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )

        seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)
        seg_token_mask = torch.cat([torch.zeros((batch_size, 1135)).bool().cuda(), seg_token_mask, torch.zeros((batch_size, 1)).bool().cuda()],dim=1, )
        last_hidden_state = self.text_hidden_fcs[0](output.hidden_states[-1])
        assert seg_token_mask.shape[-1] == last_hidden_state.shape[-2]

        pred_embeddings = []
        for i in range(batch_size):
            mask_i = seg_token_mask[i]  # [L]
            if mask_i.any():
                emb_i = last_hidden_state[i][mask_i]
                if self.context_fusion:
                    cat_emb = last_hidden_state[i][:-1, :][mask_i[1:]]
                    emb_i = torch.cat([cat_emb, emb_i], dim=-1)
            else:
                emb_i = torch.zeros(0, last_hidden_state.shape[-1], device=last_hidden_state.device, dtype=last_hidden_state.dtype)
                if self.context_fusion:
                    emb_i = torch.zeros(0, last_hidden_state.shape[-1]*2, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
            pred_embeddings.append(emb_i)

        pred_segment = []
        seg_loss = 0.0
        num_valid_samples = 0

        for i in range(batch_size):
            if pred_embeddings[i].numel() == 0:
                continue

            h_seg = pred_embeddings[i]
            point_feat = point_embeddings[i]
            logits = self.seg_decoder(h_seg, point_feat)   
            pred_mask = torch.sigmoid(logits)

            gt_mask = segment_label[i][:pred_mask.shape[0]]

            # Loss
            seg_loss += self.bce_loss_weight * F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
            seg_loss += self.dice_loss_weight * dice_loss(pred_mask, gt_mask, pred_mask.shape[-2])
            pred_segment.append(pred_mask)
            num_valid_samples += 1

        if num_valid_samples > 0:
            seg_loss = seg_loss / num_valid_samples
        else:
            seg_loss = torch.tensor(0.0, device=points.device)

        # --- 5. Total loss ---
        ce_loss = output.loss
        total_loss = ce_loss + seg_loss

        if return_lm_out:
            lm_out = self.lm_head(output.hidden_states[-1])
            return total_loss, seg_loss, pred_segment, segment_label, lm_out

        return total_loss, seg_loss, pred_segment, segment_label
