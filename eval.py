import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, HfArgumentParser

from train_lightning import ModelArguments, DataArguments, TrainingArguments
from train_lightning import LISALightningModule, LISADataModule


def _load_trainable_ckpt_into_pl_module(pl_module: LISALightningModule, ckpt_path: str, merge_lora: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    sd_model = {}
    for k, v in sd.items():
        if k.startswith("model."):
            sd_model[k[len("model."):]] = v

    for k, v in sd_model.items():
        if "lora_B" in k and hasattr(v, "shape") and len(v.shape) == 2:
            ckpt_r = v.shape[1]
            cur_r = getattr(pl_module.training_args, "lora_r", None)
            print(f"[ckpt] detected LoRA rank = {ckpt_r}, current args lora_r = {cur_r}")
            if cur_r is not None:
                assert int(cur_r) == int(ckpt_r), f"LoRA rank 不一致：ckpt r={ckpt_r} 但当前模型 r={cur_r}"
            break

    cur_sd = pl_module.model.state_dict()
    for key in ["lm_head.weight",
        "base_model.model.lm_head.weight",
        "model.lm_head.weight",
        "base_model.model.model.embed_tokens.weight",
        "model.embed_tokens.weight"]:
        if key in sd_model and key in cur_sd:
            a = sd_model[key]
            b = cur_sd[key]
            if hasattr(a, "shape") and hasattr(b, "shape") and a.shape != b.shape:
                print(f"[patch] {key} shape mismatch: ckpt {tuple(a.shape)} -> model {tuple(b.shape)}")
                new_w = b.clone()
                n0 = min(a.shape[0], b.shape[0])
                new_w[:n0] = a[:n0]
                sd_model[key] = new_w

    pl_module.model.load_state_dict(sd_model, strict=False)

    if merge_lora and getattr(pl_module.training_args, "lora_enable", False):
        try:
            print("[load] merging LoRA weights...")
            pl_module.model = pl_module.model.merge_and_unload()
            print("[load] LoRA merged.")
        except Exception as e:
            print("[WARN] merge_and_unload failed, keep peft model. err =", repr(e))


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

    pl_module = LISALightningModule(model_args, data_args, training_args, tokenizer, load_ckpt_path=None)

    _load_trainable_ckpt_into_pl_module(pl_module, training_args.load_ckpt_path, merge_lora=True)

    sd = pl_module.model.state_dict()
    keys = [k for k in sd.keys() if "seg_emb_head.propagation" in k and ("running_mean" in k or "running_var" in k)]
    datamodule = LISADataModule(model_args, data_args, training_args, tokenizer)

    precision = "bf16" if training_args.bf16 else ("16-mixed" if training_args.fp16 else "32-true")
    
    trainer = Trainer(
        devices="auto",
        accelerator="gpu",
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.test(pl_module, datamodule=datamodule)


if __name__ == "__main__":
    main()