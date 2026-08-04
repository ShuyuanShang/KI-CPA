from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial

from scvi.nn import FCLayers
from torch.distributions import Normal
from typing import Optional


class _REGISTRY_KEYS:
    X_KEY: str = "X"
    X_CTRL_KEY: str = None
    BATCH_KEY: str = None
    CATEGORY_KEY: str = "cpa_category"
    PERTURBATION_KEY: str = None
    PERTURBATION_DOSAGE_KEY: str = None
    PERTURBATIONS: str = "perts"
    PERTURBATIONS_DOSAGES: str = "perts_doses"
    SIZE_FACTOR_KEY: str = "size_factor"
    CAT_COV_KEYS: List[str] = []
    MAX_COMB_LENGTH: int = 2
    CONTROL_KEY: str = None
    DEG_MASK: str = None
    DEG_MASK_R2: str = None
    PADDING_IDX: int = 0


CPA_REGISTRY_KEYS = _REGISTRY_KEYS()


class VanillaEncoder(nn.Module):
    def __init__(
            self,
            n_input,
            n_output,
            n_hidden,
            n_layers,
            n_cat_list,
            use_layer_norm=True,
            use_batch_norm=False,
            output_activation: str = 'linear',
            dropout_rate: float = 0.1,
            activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.n_output = n_output
        self.output_activation = output_activation

        self.network = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_layer_norm=use_layer_norm,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
        )
        self.z = nn.Linear(n_hidden, n_output)

    def forward(self, inputs, *cat_list):
        if self.output_activation == 'linear':
            z = self.z(self.network(inputs, *cat_list))
        elif self.output_activation == 'relu':
            z = F.relu(self.z(self.network(inputs, *cat_list)))
        else:
            raise ValueError(f'Unknown output activation: {self.output_activation}')
        return z


class GeneralizedSigmoid(nn.Module):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, n_drugs, non_linearity='sigmoid'):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm.
        """
        super(GeneralizedSigmoid, self).__init__()
        self.non_linearity = non_linearity
        self.n_drugs = n_drugs

        self.beta = torch.nn.Parameter(
            torch.ones(1, n_drugs),
            requires_grad=True
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(1, n_drugs),
            requires_grad=True
        )

        self.vmap = None

    def forward(self, x, y):
        """
            Parameters
            ----------
            x: (batch_size, max_comb_len)
            y: (batch_size, max_comb_len)
        """
        y = y.long()
        if self.non_linearity == 'logsigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            bias = self.bias[0][y]
            beta = self.beta[0][y]
            c0 = bias.sigmoid()
            return (x * beta + bias).sigmoid() - c0
        else:
            return x

    def one_drug(self, x, i):
        if self.non_linearity == 'logsigm':
            c0 = self.bias[0][i].sigmoid()
            return (torch.log1p(x) * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        elif self.non_linearity == 'sigm':
            c0 = self.bias[0][i].sigmoid()
            return (x * self.beta[0][i] + self.bias[0][i]).sigmoid() - c0
        else:
            return x


class PerturbationNetwork(nn.Module):
    def __init__(self,
                 n_perts,
                 n_latent,
                 doser_type='logsigm',
                 n_hidden=None,
                 n_layers=None,
                 dropout_rate: float = 0.0,
                 drug_embeddings=None,):
        super().__init__()
        self.n_latent = n_latent
        
        if drug_embeddings is not None:
            self.pert_embedding = drug_embeddings
            self.pert_transformation = nn.Linear(drug_embeddings.embedding_dim, n_latent)
            self.use_rdkit = True
        else:
            self.use_rdkit = False
            self.pert_embedding = nn.Embedding(n_perts, n_latent, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
            
        self.doser_type = doser_type
        if self.doser_type == 'mlp':
            self.dosers = nn.ModuleList()
            for _ in range(n_perts):
                self.dosers.append(
                    FCLayers(
                        n_in=1,
                        n_out=1,
                        n_hidden=n_hidden,
                        n_layers=n_layers,
                        use_batch_norm=False,
                        use_layer_norm=True,
                        dropout_rate=dropout_rate
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(n_perts, non_linearity=self.doser_type)

    def forward(self, perts, dosages):
        """
            perts: (batch_size, max_comb_len)
            dosages: (batch_size, max_comb_len)
        """
        bs, max_comb_len = perts.shape
        perts = perts.long()

        # Handle different doser types
        if self.doser_type == 'mlp':
            # For MLP dosers, we need to call each layer individually
            scaled_dosages = torch.zeros_like(dosages)
            for i, doser in enumerate(self.dosers):
                mask = (perts == i)
                if mask.any():
                    scaled_dosages[mask] = doser(dosages[mask].unsqueeze(-1)).squeeze(-1)
        else:
            # For GeneralizedSigmoid
            scaled_dosages = self.dosers(dosages, perts)  # (batch_size, max_comb_len)

        drug_embeddings = self.pert_embedding(perts)  # (batch_size, max_comb_len, n_drug_emb_dim)

        if self.use_rdkit:
            drug_embeddings = self.pert_transformation(drug_embeddings.view(bs * max_comb_len, -1)).view(bs, max_comb_len, -1)

        z_drugs = torch.einsum('bm,bme->bme', [scaled_dosages, drug_embeddings])  # (batch_size, n_latent)

        z_drugs = torch.einsum('bmn,bm->bmn', z_drugs, (perts != CPA_REGISTRY_KEYS.PADDING_IDX).int()).sum(dim=1)  # mask single perts

        return z_drugs # (batch_size, n_latent)

class FocalLoss(nn.Module):
    """ 
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none')

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if len(y_true) == 0:
            return torch.tensor(0.)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(y_pred, dim=-1)
        ce = self.nll_loss(log_p, y_true)

        # get true class column from each row
        all_rows = torch.arange(len(y_pred))
        log_pt = log_p[all_rows, y_true]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class KineticsHead(nn.Module):

    def __init__(self, drug_fp_dim: int, target_emb_dim: int, n_hidden: int = 128, rank: int = 1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(drug_fp_dim + target_emb_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.log_kd_head = nn.Linear(n_hidden, 1)
        self.efficacy_head = nn.Linear(n_hidden, rank)

        nn.init.zeros_(self.efficacy_head.bias)
        nn.init.normal_(self.efficacy_head.weight, std=0.1)
        nn.init.zeros_(self.log_kd_head.bias) 

    def forward(self, drug_fp: torch.Tensor, target_emb: torch.Tensor):
        h = self.shared(torch.cat([drug_fp, target_emb], dim=-1))
        log_kd = self.log_kd_head(h).squeeze(-1)                  # (N,)
        efficacy = torch.tanh(self.efficacy_head(h))  # (N,rank) in [-1, 1]
        return log_kd, efficacy

class KineticsPerturbationNetwork(nn.Module):
 
    def __init__(
        self,
        original_pert_network: PerturbationNetwork,
        drug_embeddings_for_kinetics: nn.Embedding,
        drug_to_target: torch.Tensor,
        n_targets: int,
        n_latent: int,
        target_emb_dim: int = 64,
        n_hidden: int = 128,
        eps: float = 1e-6,
        rank: int = 1,
        combine_mode: str = "add",   # "replace" or "add"
        pretrained_target_embeddings: Optional[torch.Tensor] = None,
        freeze_target_embedding: bool = False,
        pretrained_target_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert combine_mode in ("replace", "add")
        self.n_latent = n_latent
        self.eps = eps
        self.rank = rank
        self.combine_mode = combine_mode
 
        self.original_pert_network = original_pert_network
 
        self.kinetics_drug_fp = drug_embeddings_for_kinetics
        self.kinetics_drug_fp.weight.requires_grad = False
        drug_fp_dim = drug_embeddings_for_kinetics.embedding_dim
 
        self.register_buffer("drug_to_target", drug_to_target.long())
 
        # ---- target_embedding 三选一 ----
        # 互斥：pretrained_target_embeddings 和 pretrained_target_features 只能传一个
        assert not (pretrained_target_embeddings is not None and pretrained_target_features is not None), (
            "pretrained_target_embeddings 和 pretrained_target_features 只能二选一，不能同时传"
        )
 
        if pretrained_target_features is not None:
            assert pretrained_target_features.shape[0] == n_targets, (
                f"pretrained_target_features第0维({pretrained_target_features.shape[0]})"
                f"和n_targets({n_targets})不匹配"
            )
            raw_dim = pretrained_target_features.shape[1]
            self.target_raw_features = nn.Embedding.from_pretrained(
                pretrained_target_features.clone().float(), freeze=True
            )
            self.target_feature_norm = nn.LayerNorm(raw_dim)
            self.target_projector = nn.Linear(raw_dim, target_emb_dim)
            self._target_mode = "frozen_projector"

        elif pretrained_target_embeddings is not None:
            assert pretrained_target_embeddings.shape == (n_targets, target_emb_dim), (
                f"预训练target embedding形状{tuple(pretrained_target_embeddings.shape)}"
                f"和期望的(n_targets={n_targets}, target_emb_dim={target_emb_dim})不匹配，"
                "检查n_targets/target_emb_dim是不是和构建预训练矩阵时用的一致"
            )
            self.target_embedding = nn.Embedding.from_pretrained(
                pretrained_target_embeddings.clone().float(), freeze=freeze_target_embedding
            )
            self._target_mode = "finetune_embedding"
        else:
            self.target_embedding = nn.Embedding(n_targets, target_emb_dim)
            self._target_mode = "random"
 
        self.kinetics_head = KineticsHead(drug_fp_dim, target_emb_dim, n_hidden=n_hidden, rank=rank)
 
        self.direction_head = nn.Linear(target_emb_dim, rank * n_latent, bias=False)
 
    def _get_target_embedding(self, target_idx: torch.Tensor) -> torch.Tensor:
        """统一入口：不管三种target_embedding模式里哪一种，都从这里取出(target_emb_dim,)的表示"""
        if self._target_mode == "frozen_projector":
            raw = self.target_raw_features(target_idx)
            raw = self.target_feature_norm(raw)
            return self.target_projector(raw)
        else:
            return self.target_embedding(target_idx)
 
    def _kinetics_forward(self, perts: torch.Tensor, dosages: torch.Tensor):
 
        bs, max_comb_len = perts.shape
        perts_long = perts.long()
        N = bs * max_comb_len
 
        target_idx = self.drug_to_target[perts_long]                    # (bs, max_comb_len)
        target_emb = self._get_target_embedding(target_idx)             # (bs, max_comb_len, target_emb_dim)
        drug_fp = self.kinetics_drug_fp(perts_long)                     # (bs, max_comb_len, fp_dim)
 
        flat_fp = drug_fp.view(N, -1)
        flat_target_emb = target_emb.view(N, -1)
        flat_dosages = dosages.reshape(N)
 
        log_kd, efficacy = self.kinetics_head(flat_fp, flat_target_emb)  # log_kd: (N,), efficacy: (N, rank)
 
        kd = torch.exp(log_kd)                                            # (N,)
        theta = flat_dosages / (kd + flat_dosages + self.eps)             # (N,) —— Kd 仍是标量，占据率对所有 r 个方向共享
        delta_a = efficacy * theta.unsqueeze(-1)                          # (N, rank)
 
        direction = self.direction_head(flat_target_emb)                  # (N, rank * n_latent)
        direction = direction.view(N, self.rank, self.n_latent)
        direction = F.normalize(direction, dim=-1)                        
 
        z_flat = torch.einsum('nr,nrl->nl', delta_a, direction)           # (N, n_latent) —— r 个方向按 delta_a 加权求和
 
        z_pert_kinetics = z_flat.view(bs, max_comb_len, self.n_latent)
        mask = (perts_long != CPA_REGISTRY_KEYS.PADDING_IDX).int()
        z_pert_kinetics = torch.einsum('bmn,bm->bmn', z_pert_kinetics, mask).sum(dim=1)  # (bs, n_latent)
 
        delta_a = delta_a.view(bs, max_comb_len, self.rank)
        theta = theta.view(bs, max_comb_len)
        efficacy = efficacy.view(bs, max_comb_len, self.rank)
        kd = kd.view(bs, max_comb_len)
 
        return z_pert_kinetics, delta_a, theta, efficacy, kd
 
    def forward(self, perts: torch.Tensor, dosages: torch.Tensor):
        z_pert_kinetics, *_ = self._kinetics_forward(perts, dosages)
 
        if self.combine_mode == "replace":
            return z_pert_kinetics
        else:  # "add"
            z_drug_original = self.original_pert_network(perts, dosages)
            return z_drug_original + z_pert_kinetics
 
    @property
    def pert_embedding(self):
        return self.original_pert_network.pert_embedding
 
    @property
    def dosers(self):
        return self.original_pert_network.dosers
 
    @torch.no_grad()
    def diagnostics(self, perts: torch.Tensor, dosages: torch.Tensor):
        z_pert_kinetics, delta_a, theta, efficacy, kd = self._kinetics_forward(perts, dosages)
        return {
            "kd_mean": kd.mean().item(),
            "theta_mean": theta.mean().item(),
            "theta_std": theta.std().item(),
            "efficacy_mean": efficacy.mean().item(),
            "efficacy_std": efficacy.std().item(),
            # 每个 rank 分量分别看，方便判断是不是某几个方向被学"废"了（efficacy std 接近 0）
            "efficacy_std_per_rank": efficacy.std(dim=(0, 1)).tolist(),
            "delta_a_std": delta_a.std().item(),
            "z_pert_kinetics_norm": z_pert_kinetics.norm(dim=-1).mean().item(),
        }