"""
  # 方案B（推荐）：冻结ESM2特征 + 可训练线性投影层
  python train_new.py --input_file /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad --rank 1 --combine_mode add \
      --target_embedding_mode frozen_projector \
      --pretrained_target_npz target_embeddings_esm2_raw.npz

  # 方案A：预训练值当初始值，逐target微调
  python train_new.py --input_file /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad --rank 1 --combine_mode add \
      --target_embedding_mode finetune_embedding \
      --pretrained_target_npz target_embeddings_esm2_pca64.npz

  # 方案A的对照：预训练值完全冻结不微调
  python train_kicpa_loo.py --input_file /root/autodl-tmp/data/adata_loo.h5ad --rank 1 --combine_mode replace \
      --target_embedding_mode finetune_embedding \
      --pretrained_target_npz target_embeddings_esm2_pca64.npz --freeze_target_embedding
"""

import argparse

import anndata
import torch

from ki_cpa import KineticsCPA
from target_utils import build_pretrained_target_matrix, build_pretrained_target_features

CONTROL_PERT_NAME = 'DMSO'


def build_drug_to_target_mapping(adata, pert_encoder: dict, control_group: str,
                                  ood_drug_names: set = None,
                                  target_col: str = "TTD Target ID"):
    obs = adata.obs
    drug_to_target_name = (
        obs.loc[obs["perturbation"] != control_group, ["perturbation", target_col]]
        .drop_duplicates()
        .set_index("perturbation")[target_col]
        .to_dict()
    )
    check = obs.loc[obs["perturbation"] != control_group].groupby(
        "perturbation", observed=True
    )[target_col].nunique()
    assert (check <= 1).all(), (
        f"以下药物对应了多个 TTD Target ID: {check[check > 1].index.tolist()}"
    )

    unique_targets = sorted(set(drug_to_target_name.values()))
    target_name_to_idx = {t: i + 1 for i, t in enumerate(unique_targets)}
    n_targets = len(unique_targets) + 1

    drug_to_target = torch.zeros(len(pert_encoder), dtype=torch.long)
    missing = []
    for drug_name, drug_idx in pert_encoder.items():
        if drug_name in ("<PAD>", control_group):
            continue
        target_name = drug_to_target_name.get(drug_name)
        if target_name is not None:
            drug_to_target[drug_idx] = target_name_to_idx[target_name]
        else:
            missing.append(drug_name)

    if missing:
        print(f"警告: {len(missing)} 个药物找不到对应target，指向dummy(0): "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    # 硬校验：held-out(ood_test)药物必须有已知target，否则LOO实验的前提被破坏
    # ——它们的target_embedding会退化成随机初始化，而不是"未见药物但target已知"
    if ood_drug_names is not None:
        missing_ood = set(missing) & set(ood_drug_names)
        assert not missing_ood, (
            f"以下held-out(ood_test)药物找不到target，会导致LOO实验前提失效，"
            f"必须先排查setup_anndata的类别处理逻辑: {missing_ood}"
        )

    return drug_to_target, n_targets, target_name_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="prepare_loo_data.py的输出h5ad")
    parser.add_argument("--save_path", default=None,
                         help="不指定时会自动拼成 cpa_kicpa_loo_r{rank}_{combine_mode}")
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=1,
                         help="target-shared方向的秩，1即原始设计")
    parser.add_argument("--combine_mode", choices=["replace", "add"], default="replace",
                         help="replace: 只用kinetics支路做drug conditioning；"
                              "add: kinetics支路与原始z_drug相加（原始设计）")
    parser.add_argument("--target_embedding_mode", choices=["random", "finetune_embedding", "frozen_projector"],
                         default="random",
                         help="random(默认): 完全随机初始化；"
                              "finetune_embedding(方案A): 预训练值当初始值，逐target微调；"
                              "frozen_projector(方案B，推荐): 冻结ESM2原始特征+可训练共享投影层")
    parser.add_argument("--pretrained_target_npz", default=None,
                         help="build_target_embeddings_esm2.py产出的.npz路径，"
                              "target_embedding_mode不是random时必须提供")
    parser.add_argument("--freeze_target_embedding", action="store_true",
                         help="仅finetune_embedding模式下生效：冻结target_embedding，不参与梯度更新。"
                              "不加此flag则以预训练值为起点继续微调（默认，通常更推荐）")
    args = parser.parse_args()

    if args.target_embedding_mode != "random":
        assert args.pretrained_target_npz is not None, (
            f"--target_embedding_mode {args.target_embedding_mode} 需要同时提供 --pretrained_target_npz"
        )

    save_path = args.save_path or f"cpa_kicpa_loo_r{args.rank}_{args.combine_mode}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"rank={args.rank}, combine_mode={args.combine_mode}, save_path={save_path}")
    print(f"target_embedding模式: {args.target_embedding_mode}"
          + (f" (freeze={args.freeze_target_embedding})" if args.target_embedding_mode == "finetune_embedding" else ""))

    adata_processed = anndata.read_h5ad(args.input_file)
    print(f"数据形状: {adata_processed.shape}")
    print(f"split分布:\n{adata_processed.obs['split'].value_counts()}")

    KineticsCPA.setup_anndata(
        adata_processed,
        perturbation_key='perturbation',
        control_group=CONTROL_PERT_NAME,
        dosage_key='dose_val',
        categorical_covariate_keys=['cell_type'],
        is_count_data=False,
        max_comb_len=1,
        smiles_key='SMILES',
    )

    ood_drug_names = set(
        adata_processed.obs.loc[adata_processed.obs['split'] == 'ood_test', 'perturbation']
        .unique().tolist()
    )

    drug_to_target, n_targets, target_name_to_idx = build_drug_to_target_mapping(
        adata_processed,
        pert_encoder=KineticsCPA.pert_encoder,
        control_group=CONTROL_PERT_NAME,
        ood_drug_names=ood_drug_names,
        target_col="TTD Target ID",
    )
    print(f"n_targets(含dummy) = {n_targets}")

    # ---- 预训练target embedding（可选，三选一） ----
    target_emb_dim = 64  # 和下面ae_hparams/KineticsCPA构造保持一致，单独提出来方便两处引用
    pretrained_target_embeddings = None
    pretrained_target_features = None

    if args.target_embedding_mode == "finetune_embedding":
        pretrained_target_embeddings = build_pretrained_target_matrix(
            npz_path=args.pretrained_target_npz,
            target_name_to_idx=target_name_to_idx,
            n_targets=n_targets,
            target_emb_dim=target_emb_dim,
            seed=434,
        )
    elif args.target_embedding_mode == "frozen_projector":
        pretrained_target_features = build_pretrained_target_features(
            npz_path=args.pretrained_target_npz,
            target_name_to_idx=target_name_to_idx,
            n_targets=n_targets,
        )

    # ---- 与 train_data.py 完全一致的超参数，保证公平对比 ----
    ae_hparams = {
        "n_latent": 128, "recon_loss": "gauss", "doser_type": "mlp",
        "n_hidden_encoder": 256, "n_layers_encoder": 2,
        "n_hidden_decoder": 256, "n_layers_decoder": 2,
        "use_batch_norm_encoder": True, "use_layer_norm_encoder": True,
        "use_batch_norm_decoder": True, "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.2, "dropout_rate_decoder": 0.2,
        "variational": False, "seed": 434,
    }
    trainer_params = {
        "n_epochs_kl_warmup": None, "n_epochs_pretrain_ae": 50,
        "n_epochs_adv_warmup": 100, "n_epochs_mixup_warmup": 10,
        "mixup_alpha": 0.2, "adv_steps": 3, "n_hidden_adv": 32, "n_layers_adv": 2,
        "use_batch_norm_adv": True, "use_layer_norm_adv": True, "dropout_rate_adv": 0.4,
        "reg_adv": 10.0, "pen_adv": 10.0, "lr": 0.001, "wd": 1e-6,
        "adv_lr": 0.001, "adv_wd": 1e-6, "adv_loss": "cce",
        "doser_lr": 0.001, "doser_wd": 1e-6, "do_clip_grad": True,
        "gradient_clip_value": 0.5, "step_size_lr": 20,
    }

    model = KineticsCPA(
        adata=adata_processed,
        drug_to_target=drug_to_target,
        n_targets=n_targets,
        target_emb_dim=target_emb_dim,
        kinetics_hidden=128,
        rank=args.rank,
        combine_mode=args.combine_mode,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood_test",
        use_rdkit_embeddings=True,
        pretrained_target_embeddings=pretrained_target_embeddings,
        freeze_target_embedding=args.freeze_target_embedding,
        pretrained_target_features=pretrained_target_features,
        **ae_hparams,
    )

    model.train(
        max_epochs=args.max_epochs,
        use_gpu=True,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path=save_path,
    )

    print(f"\n训练完成，模型已保存至: {save_path}")
    print("评估时用 final_eval_kicpa.py / final_eval_loo.py，obsm key 记得用 'KineticsCPA_pred'，"
          "且评估应只针对 split=='ood_test' 的行，才是真正的leave-one-drug-out测试")


if __name__ == "__main__":
    main()