from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
from torch import optim

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from model import (
    Transformer,
    Informer,
    Reformer,
    Flowformer,
    Flashformer,
    iTransformer,
    iInformer,
    iReformer,
    iFlowformer,
    iFlashformer,
)
from . import csg_model


class Exp_Long_Term_Forecast_Ext(Exp_Long_Term_Forecast):
    """Extension point that adds iTransformer_CSG and weight decay support."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "Transformer": Transformer,
            "Informer": Informer,
            "Reformer": Reformer,
            "Flowformer": Flowformer,
            "Flashformer": Flashformer,
            "iTransformer": iTransformer,
            "iInformer": iInformer,
            "iReformer": iReformer,
            "iFlowformer": iFlowformer,
            "iFlashformer": iFlashformer,
            "iTransformer_CSG": csg_model,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        weight_decay = float(getattr(self.args, "weight_decay", 0.0))
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=weight_decay,
        )

    def _compact_tag(self, extra_tag: str) -> str:
        if not extra_tag:
            return ""
        safe = "".join(ch if ch.isalnum() else "_" for ch in extra_tag).strip("_")
        return safe[:18]

    def make_setting(self, extra_tag: str = "") -> str:
        signature = (
            f"{self.args.model_id}|{self.args.model}|{self.args.data}|{self.args.features}|"
            f"sl={self.args.seq_len}|ll={self.args.label_len}|pl={self.args.pred_len}|"
            f"dm={self.args.d_model}|nh={self.args.n_heads}|el={self.args.e_layers}|"
            f"dl={self.args.d_layers}|df={self.args.d_ff}|fc={self.args.factor}|"
            f"eb={self.args.embed}|des={self.args.des}|cls={self.args.class_strategy}|"
            f"bs={getattr(self.args, 'batch_size', 'na')}|lr={self.args.learning_rate}|"
            f"wd={getattr(self.args, 'weight_decay', 0.0)}|"
            f"csg={getattr(self.args, 'csg_reduction', 'na')}|tag={extra_tag}"
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        parts = [
            str(self.args.model_id),
            str(self.args.model),
            f"pl{self.args.pred_len}",
            f"sl{self.args.seq_len}",
            f"bs{getattr(self.args, 'batch_size', 'na')}",
        ]
        compact_tag = self._compact_tag(extra_tag)
        if compact_tag:
            parts.append(compact_tag)
        parts.append(digest)
        return "_".join(parts)
