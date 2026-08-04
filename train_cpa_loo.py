"""
train_baseline_loo.py
====================
用与 train_kicpa_loo.py 完全相同的数据文件、完全相同的split、完全相同的
超参数，训练官方 baseline CPA（不含kinetics支路）。这是保证对比公平的
关键——不能拿旧的、用随机细胞切分训练出的baseline checkpoint来比较。

用法：
  python train_cpa_loo.py --input_file /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad --save_path cpa_baseline_loo
"""

import argparse

import anndata
import cpa
import torch

CONTROL_PERT_NAME = 'DMSO'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="prepare_loo_data.py的输出h5ad（与KI-CPA共用同一份）")
    parser.add_argument("--save_path", default="cpa_baseline_loo")
    parser.add_argument("--max_epochs", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    adata_processed = anndata.read_h5ad(args.input_file)
    print(f"数据形状: {adata_processed.shape}")
    print(f"split分布:\n{adata_processed.obs['split'].value_counts()}")

    cpa.CPA.setup_anndata(
        adata_processed,
        perturbation_key='perturbation',
        control_group=CONTROL_PERT_NAME,
        dosage_key='dose_val',
        categorical_covariate_keys=['cell_type'],
        is_count_data=False,
        max_comb_len=1,
        smiles_key='SMILES',
    )

    # ---- 与 train_kicpa_loo.py 完全一致的超参数 ----
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

    model = cpa.CPA(
        adata=adata_processed,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood_test",
        use_rdkit_embeddings=True,
        **ae_hparams,
    )

    model.train(
        max_epochs=args.max_epochs,
        use_gpu=True,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path=args.save_path,
    )

    print(f"\n训练完成，模型已保存至: {args.save_path}")
    print("评估时obsm key是 'CPA_pred'（不是'KineticsCPA_pred'），"
          "同样只看 split=='ood_test' 的行")


if __name__ == "__main__":
    main()