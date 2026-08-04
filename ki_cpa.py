import numpy as np
import torch
import torch.nn as nn
from scvi import settings
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Encoder, DecoderSCVI
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl
from torchmetrics.functional import accuracy, pearson_corrcoef, r2_score
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cpa
from cpa._utils import PerturbationNetwork, KineticsPerturbationNetwork

class KineticsCPA(cpa.CPA):
    def __init__(
        self,
        adata,
        drug_to_target: torch.Tensor,
        n_targets: int,
        target_emb_dim: int = 64,
        kinetics_hidden: int = 128,
        rank: int = 1,
        combine_mode: str = "replace",
        use_rdkit_embeddings: bool = True,
        pretrained_target_embeddings: torch.Tensor = None,  # <<< 方案A：预训练值当初始值再微调
        freeze_target_embedding: bool = False,               # <<< 仅方案A用：是否冻结
        pretrained_target_features: torch.Tensor = None,     # <<< 方案B（推荐）：冻结ESM2特征+可训练投影层
        **kwargs,
    ):
        super().__init__(adata, use_rdkit_embeddings=use_rdkit_embeddings, **kwargs)
 
        original_pert_network: PerturbationNetwork = self.module.pert_network
 
        if not original_pert_network.use_rdkit:
            raise ValueError(
                "KI-CPA 需要 drug fingerprint 作为 efficacy 的输入特征之一，"
                "请在 setup_anndata 时指定 smiles_key 并传入 use_rdkit_embeddings=True"
            )
 
        drug_embeddings_for_kinetics = original_pert_network.pert_embedding
 
        device = next(self.module.parameters()).device
        self.module.pert_network = KineticsPerturbationNetwork(
            original_pert_network=original_pert_network,
            drug_embeddings_for_kinetics=drug_embeddings_for_kinetics,
            drug_to_target=drug_to_target,
            n_targets=n_targets,
            n_latent=self.module.n_latent,
            target_emb_dim=target_emb_dim,
            n_hidden=kinetics_hidden,
            rank=rank,
            combine_mode=combine_mode,
            pretrained_target_embeddings=pretrained_target_embeddings,  # <<<
            freeze_target_embedding=freeze_target_embedding,            # <<<
            pretrained_target_features=pretrained_target_features,      # <<<
        ).to(device)
 
        self._model_summary_string = (
            "Kinetics-Informed Compositional Perturbation Autoencoder (KI-CPA)"
        )
 
        self.init_params_ = self._get_init_params(locals())