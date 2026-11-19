"""The five frozen vision-language backbones evaluated in the paper (Sec. 4.3).

All models are loaded from their released weights through ``open_clip``. DermLIP needs one
extra step: its image tower is the PanDerm ViT (:mod:`declip.panderm_vit`), which is not a
stock ``open_clip`` architecture, so the tower is swapped in before the checkpoint is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass

import open_clip
import torch
from open_clip.factory import _get_hf_config, load_checkpoint
from open_clip.model import CustomTextCLIP
from open_clip.pretrained import download_pretrained_from_hf
from open_clip.transform import PreprocessCfg, image_transform_v2, merge_preprocess_dict

from declip.panderm_vit import PanDermVisionTransformer

log = logging.getLogger(__name__)

HF_HUB_PREFIX = "hf-hub:"


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    display_name: str
    model: str
    pretrained: str | None = None
    panderm_vision: bool = False


BACKBONES: dict[str, BackboneSpec] = {
    "dermlip": BackboneSpec(
        "dermlip",
        "DermLIP",
        "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        panderm_vision=True,
    ),
    "make": BackboneSpec("make", "MAKE", "hf-hub:xieji-x/MAKE"),
    "biomedclip": BackboneSpec(
        "biomedclip",
        "BiomedCLIP",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ),
    "pmc-clip": BackboneSpec("pmc-clip", "PMC-CLIP", "hf-hub:ryanyip7777/pmc_vit_l_14"),
    # OpenAI weights were trained with QuickGELU; plain "ViT-L-14" would load them with GELU.
    "openai-clip": BackboneSpec(
        "openai-clip", "OpenAI-CLIP", "ViT-L-14-quickgelu", pretrained="openai"
    ),
}


@dataclass
class Backbone:
    spec: BackboneSpec
    model: torch.nn.Module
    preprocess: object
    tokenizer: object
    device: torch.device
    precision: str = "amp"
    _weights_digest: str | None = None

    @property
    def autocast(self):
        if self.precision == "amp" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type=self.device.type, enabled=False)

    @property
    def image_size(self) -> tuple[int, int]:
        size = getattr(self.model.visual, "image_size", None)
        if size is None:
            return (224, 224)
        return tuple(size) if not isinstance(size, int) else (size, size)

    def weights_digest(self) -> str:
        """MD5 over per-parameter sums; identifies the exact checkpoint in cache keys."""
        if self._weights_digest is None:
            parts = []
            with torch.no_grad():
                for name, p in sorted(self.model.named_parameters(), key=lambda kv: kv[0]):
                    parts.append(f"{name}:{p.detach().float().sum().item():.8e}")
            self._weights_digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        return self._weights_digest

    def cache_key(self) -> dict:
        return {
            "model": self.spec.model,
            "pretrained": self.spec.pretrained,
            "precision": self.precision,
            "weights": self.weights_digest(),
        }


def load_backbone(name: str, device: str = "cuda", precision: str = "amp") -> Backbone:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone '{name}'; choose from {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    dev = torch.device(device)
    if spec.panderm_vision:
        model, preprocess = _load_dermlip(spec.model, dev)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model, pretrained=spec.pretrained, precision="fp32", device=dev
        )
    model.eval()
    tokenizer = open_clip.get_tokenizer(spec.model)
    log.info("loaded backbone %s (%s)", spec.display_name, spec.model)
    return Backbone(spec, model, preprocess, tokenizer, dev, precision)


def _load_dermlip(model_name: str, device: torch.device):
    model_id = model_name[len(HF_HUB_PREFIX) :]
    config = _get_hf_config(model_id)
    model_cfg = copy.deepcopy(config["model_cfg"])
    vision_cfg = model_cfg["vision_cfg"]
    vision_cfg.pop("pretrain_path", None)
    model = CustomTextCLIP(**model_cfg)
    model.visual = PanDermVisionTransformer(
        image_size=vision_cfg["image_size"],
        patch_size=vision_cfg["patch_size"],
        embed_dim=vision_cfg["width"],
        depth=vision_cfg["layers"],
        num_heads=vision_cfg["width"] // 64,
        output_dim=model_cfg["embed_dim"],
    )
    model.to(device)
    checkpoint_path = download_pretrained_from_hf(model_id)
    incompatible = load_checkpoint(model, checkpoint_path, strict=False)
    if incompatible is not None and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"DermLIP checkpoint does not match the model: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    preprocess_cfg = merge_preprocess_dict(PreprocessCfg().__dict__, config["preprocess_cfg"])
    preprocess_cfg["size"] = vision_cfg["image_size"]
    preprocess = image_transform_v2(PreprocessCfg(**preprocess_cfg), is_train=False)
    return model, preprocess
