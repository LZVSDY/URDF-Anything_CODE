
import os
import json
import copy
import argparse
import logging
from functools import partial
from typing import Dict, List, Optional
import logging

from transformers import logging as hf_logging
import torch
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

from model.UA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_point_token
from model.llava.constants import IGNORE_INDEX
from model.llava import conversation as conversation_lib
from utils.reason_seg_dataset import URDFReasoningDataset, collate_fn
from model.llava.constants import POINT_TOKEN_INDEX
from tqdm import tqdm
import numpy as np
import warnings
from pydantic.warnings import PydanticDeprecatedSince20


class LISADataModule(pl.LightningDataModule):
    def __init__(self, model_args, data_args, training_args, tokenizer):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.predict_type = data_args.predict_type

    def setup(self, stage=None):
        self.train_dataset = URDFReasoningDataset(task_mode=self.predict_type, split="train", max_samples=self.data_args.max_samples)
        self.val_dataset = URDFReasoningDataset(task_mode=self.predict_type, split="test", max_samples=self.data_args.max_samples)
        self.test_dataset = self.val_dataset
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

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
            ),
            pin_memory=True,
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
            ),
            pin_memory=True,
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
                inference_mode=True
            ),
        )


class LISALightningModule(pl.LightningModule):
    def __init__(self, model_args, data_args, training_args, tokenizer, load_ckpt_path=None):
        super().__init__()
        if load_ckpt_path is not None:
            self.__init_from_ckpt__(model_args, data_args, training_args, tokenizer, load_ckpt_path)
            return
        self.save_hyperparameters()
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

        # Model
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
            context_fusion=model_args.context_fusion
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

        # Vision tower
        self.model.get_model().initialize_vision_modules(model_args=model_args)
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()

        # Config
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



    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        total_loss, seg_loss, _, _ = self.model(**batch)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_llm_loss", total_loss-seg_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
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
    context_fusion: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
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
    model_max_length: int = field( default=2048,metadata={"help":"Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    auto_resume:bool=field(default=True)
    resume:str = ""
    debug:bool=field(default=False)
    train_mask_decoder:bool=field(default=False)
    deepspeed:bool=field(default=False)
    warmup_ratio: float = 0.3
    do_eval: bool = field(default=False)
    load_ckpt_path: str = field(default=None)



def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    pl.seed_everything(training_args.seed)

    # Tokenizer
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
        monitor=f"train_total_loss",
        dirpath=os.path.join(training_args.output_dir, "checkpoints"),
        filename="{epoch}-{train_total_loss:.4f}",
        save_top_k=1,
        every_n_epochs=1,
        mode = "min",
        save_last=True,
        save_weights_only=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    strategy = "deepspeed"

    trainer = Trainer(
        default_root_dir=training_args.output_dir,
        max_epochs=training_args.num_train_epochs,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        precision="bf16",
        strategy=strategy,
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