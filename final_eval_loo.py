"""
  python final_eval_loo.py \
      --model_type kicpa \
      --model_dir cpa_kicpa_loo_r1_add \
      --data_path /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad \
      --holdout_csv /root/autodl-tmp/data/adata_loo_cross_target_similar_holdout_drugs.csv \
      --output_csv kicpa_loo_eval.csv

  python final_eval_loo.py \
      --model_type baseline \
      --model_dir cpa_baseline_loo \
      --data_path /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad \
      --holdout_csv /root/autodl-tmp/data/adata_loo_cross_target_similar_holdout_drugs.csv\
      --output_csv baseline_loo_eval.csv
"""

import argparse
import pickle
from collections import defaultdict

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

import cpa
from cpa._utils import CPA_REGISTRY_KEYS
from ki_cpa import KineticsCPA

# ---- Monkey patch 修复 PyTorch 2.6 weights_only 问题 ----
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=pickle, **kwargs):
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)
torch.load = patched_torch_load

CONTROL_PERT_NAME = 'DMSO'
MIN_CELLS_FOR_EVAL = 2
MIN_CELLS_CTRL = 3
N_TOP_DEGS_LIST = [10, 20, 50, None]
SMILES_KEY = 'SMILES'
TARGET_COL = 'TTD Target ID'


def build_ctrl_pool_and_deg_dict(adata, ood_drugs):
    """
    ctrl_pool: {cell_type: AnnData} —— 从train+valid里pool出的DMSO细胞，
               作为held-out药物评估时的对照基准(LFC计算 + DEG计算共用)
    deg_dict:  {cell_type: {drug: [gene_names排序]}} —— 每个held-out药物
               在每个细胞系下的DEG列表，用held-out药物在ood_test里的真实
               表达 vs ctrl_pool计算，不涉及模型预测值
    """
    ref_mask = adata.obs['split'].isin(['train', 'valid']) & (adata.obs['perturbation'] == CONTROL_PERT_NAME)
    ctrl_pool = {}
    for cell_type in adata.obs['cell_type'].unique():
        sub = adata[ref_mask & (adata.obs['cell_type'] == cell_type)]
        if sub.n_obs >= MIN_CELLS_CTRL:
            ctrl_pool[cell_type] = sub.copy()

    deg_dict = defaultdict(dict)
    ood_adata = adata[adata.obs['split'] == 'ood_test']
    for cell_type in ood_adata.obs['cell_type'].unique():
        if cell_type not in ctrl_pool:
            continue
        ct_ood = ood_adata[ood_adata.obs['cell_type'] == cell_type]
        for drug in ct_ood.obs['perturbation'].unique():
            if drug not in ood_drugs:
                continue
            drug_cells = ct_ood[ct_ood.obs['perturbation'] == drug]
            if drug_cells.n_obs < MIN_CELLS_FOR_EVAL:
                continue

            combo = anndata.concat(
                [drug_cells, ctrl_pool[cell_type]], label='grp', keys=['treated', 'ctrl']
            )
            combo.obs['grp'] = combo.obs['grp'].astype('category')
            try:
                sc.tl.rank_genes_groups(combo, groupby='grp', groups=['treated'],
                                         reference='ctrl', method='wilcoxon')
                df_deg = sc.get.rank_genes_groups_df(combo, group='treated')
                deg_dict[cell_type][drug] = df_deg['names'].tolist()
            except Exception as e:
                print(f"  警告: {cell_type}/{drug} 计算DEG失败: {e}")

    return ctrl_pool, deg_dict


def compute_per_drug_kinetics_diagnostics(model, adata, ood_drugs):
    """
    对每个held-out药物单独调用一次 model.module.pert_network.diagnostics()，
    黑盒复用已有方法，得到per-drug而非全局的kinetics诊断量。
    仅当 model 是 KineticsCPA 时调用。
    """
    rows = []
    ood_mask = adata.obs['split'] == 'ood_test'
    for drug in ood_drugs:
        drug_mask = ood_mask & (adata.obs['perturbation'] == drug)
        indices = np.where(drug_mask)[0]
        if len(indices) == 0:
            continue
        loader = model._make_data_loader(adata=adata, indices=indices, batch_size=max(1024, len(indices)))
        batch = next(iter(loader))
        perts = batch[CPA_REGISTRY_KEYS.PERTURBATIONS].to(model.module.device)
        perts_doses = batch[CPA_REGISTRY_KEYS.PERTURBATIONS_DOSAGES].to(model.module.device)
        diag = model.module.pert_network.diagnostics(perts, perts_doses)
        row = {"drug": drug, "n_samples_diag": len(indices)}
        for k, v in diag.items():
            if k == "efficacy_std_per_rank":
                for r_idx, val in enumerate(v):
                    row[f"diag_efficacy_std_rank{r_idx}"] = val
            else:
                row[f"diag_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_structure_similarity_correlation(df, holdout_df, n_top_deg_for_analysis=20):
    """
    仅当holdout_df里有avg_similarity_to_siblings列时才运行（即用
    prepare_loo_data_structure_diverse.py生成的数据）。
    计算每个药物的r2_mean_lfc_deg（在n_top_deg_for_analysis这个子集下）
    和avg_similarity_to_siblings之间的Spearman相关系数，打印结果。
    """
    if "avg_similarity_to_siblings" not in holdout_df.columns:
        print("\n（holdout_csv没有avg_similarity_to_siblings列，跳过结构相似度相关性分析，"
              "如果想看这个诊断，用prepare_loo_data_structure_diverse.py重新生成数据）")
        return

    sub = df[df["n_top_deg"] == n_top_deg_for_analysis][["perturbation", "r2_mean_lfc_deg"]]
    merged = sub.merge(
        holdout_df[["drug", "avg_similarity_to_siblings"]].dropna(subset=["avg_similarity_to_siblings"]),
        left_on="perturbation", right_on="drug", how="inner"
    )
    if len(merged) < 4:
        print(f"\n可用于结构相似度相关性分析的药物数太少({len(merged)})，跳过")
        return

    try:
        from scipy.stats import spearmanr
        rho, p_value = spearmanr(merged["avg_similarity_to_siblings"], merged["r2_mean_lfc_deg"])
        print(f"\n=== 结构相似度 vs 预测质量 相关性分析 (n={len(merged)}) ===")
        print(f"Spearman相关系数(avg_similarity_to_siblings vs r2_mean_lfc_deg): "
              f"rho={rho:.3f}, p={p_value:.3f}")
        if p_value < 0.05 and rho > 0.3:
            print("  -> 显著正相关：held-out药物和兄弟药物结构越像，R²越高。"
                  "暗示模型对结构差异大的药物预测更差，可能依赖了结构相似性这个shortcut，"
                  "target桥梁没有完全弥补这个gap")
        elif p_value < 0.05 and rho < -0.3:
            print("  -> 显著负相关：反直觉，值得进一步检查数据/流程是否有问题")
        else:
            print("  -> 无显著相关：held-out药物的预测质量和它跟兄弟药物的结构相似度关系不大，"
                  "更支持模型是在利用target信息做泛化，而不是单纯抄结构相近药物的答案")
        print(merged.sort_values("avg_similarity_to_siblings").to_string(index=False))
    except ImportError:
        print("\n未安装scipy，跳过结构相似度相关性分析（pip install scipy）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["kicpa", "baseline"], required=True)
    parser.add_argument("--model_dir", required=True, help="model.train()时save_path指定的目录")
    parser.add_argument("--data_path", required=True, help="prepare_loo_data.py(或structure_diverse版本)的输出h5ad")
    parser.add_argument("--holdout_csv", required=True, help="对应的*_holdout_drugs.csv")
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}, model_type={args.model_type}")

    # ---- 1. 加载数据 ----
    print(f"加载数据: {args.data_path}")
    adata = anndata.read_h5ad(args.data_path)
    print(f"数据形状: {adata.shape}")
    print(f"split分布:\n{adata.obs['split'].value_counts()}")
    assert set(adata.obs['split'].unique()) <= {'train', 'valid', 'ood_test'}, (
        "split列取值不是预期的train/valid/ood_test，确认是不是传错了旧版数据文件"
    )

    holdout_df = pd.read_csv(args.holdout_csv)
    ood_drugs = set(holdout_df['drug'])
    print(f"held-out药物数: {len(ood_drugs)}")
    actual_ood_drugs = set(adata.obs.loc[adata.obs['split'] == 'ood_test', 'perturbation'].unique())
    if ood_drugs != actual_ood_drugs:
        print(f"⚠️ holdout_csv记录的药物集合与adata里实际ood_test的药物集合不完全一致！"
              f"csv独有:{ood_drugs - actual_ood_drugs}, adata独有:{actual_ood_drugs - ood_drugs}")

    # ---- 2. setup_anndata + 加载模型 ----
    setup_kwargs = dict(
        perturbation_key='perturbation',
        control_group=CONTROL_PERT_NAME,
        dosage_key='dose_val',
        categorical_covariate_keys=['cell_type'],
        is_count_data=False,
        max_comb_len=1,
        smiles_key=SMILES_KEY,
    )

    if args.model_type == "kicpa":
        KineticsCPA.setup_anndata(adata, **setup_kwargs)
        model = KineticsCPA.load(dir_path=args.model_dir, adata=adata, use_gpu="cuda")
        assert type(model.module.pert_network).__name__ == "KineticsPerturbationNetwork", (
            "加载后的pert_network不是KineticsPerturbationNetwork，load()没有正确重建KI-CPA结构"
        )
        pn = model.module.pert_network
        print(f"加载模型: rank={pn.rank}, combine_mode={pn.combine_mode}")
        target_mode = getattr(pn, "_target_mode", "random")
        if target_mode == "frozen_projector":
            print("target_embedding模式: frozen_projector（ESM2原始特征冻结，共享线性投影层可训练）")
        elif target_mode == "finetune_embedding":
            frozen = not pn.target_embedding.weight.requires_grad
            print(f"target_embedding模式: finetune_embedding（是否冻结: {frozen}）")
        else:
            print("target_embedding模式: random（原始随机初始化设计）")
    else:
        cpa.CPA.setup_anndata(adata, **setup_kwargs)
        model = cpa.CPA.load(dir_path=args.model_dir, adata=adata, use_gpu="cuda")

    PRED_KEY = f"{model.__class__.__name__}_pred"
    print(f"模型类型: {type(model).__name__}, 预测结果obsm key: {PRED_KEY}")

    # ---- 3.【仅kicpa】per-drug kinetics诊断 ----
    if args.model_type == "kicpa":
        print("\n=== 按药物计算kinetics诊断（非全局聚合） ===")
        diag_df = compute_per_drug_kinetics_diagnostics(model, adata, ood_drugs)
        print(diag_df.to_string(index=False))
        diag_out = args.output_csv.rsplit(".", 1)[0] + "_kinetics_diag.csv"
        diag_df.to_csv(diag_out, index=False)
        print(f"per-drug kinetics诊断已保存至: {diag_out}")

    # ---- 4. 预测 ----
    print("\n运行预测...")
    model.predict(adata, batch_size=1024)
    print(f"预测完成，adata.obsm['{PRED_KEY}'] 已生成")

    # ---- 5. 构建DEG字典和ctrl_pool（用真实数据，不涉及模型预测） ----
    print("\n计算held-out药物的DEG（真实表达 vs train+valid里pool的DMSO）...")
    ctrl_pool, deg_dict = build_ctrl_pool_and_deg_dict(adata, ood_drugs)
    print(f"完成DEG计算，覆盖 {len(deg_dict)} 个细胞系")
    assert len(deg_dict) > 0, "DEG计算结果为空，检查ctrl_pool是否为空"

    # ---- 6. 在 ood_test 上评估 ----
    print("\n开始在 ood_test（held-out药物）上评估...")
    ood_adata = adata[adata.obs['split'] == 'ood_test'].copy()
    print(f"ood_test 细胞数: {ood_adata.n_obs}")

    results = defaultdict(list)
    for cell_type in tqdm(ood_adata.obs['cell_type'].unique()):
        if cell_type not in ctrl_pool or cell_type not in deg_dict:
            continue
        ct_ood = ood_adata[ood_adata.obs['cell_type'] == cell_type]
        ctrl_cells = ctrl_pool[cell_type]
        x_ctrl_full = np.asarray(ctrl_cells.X.todense()) if hasattr(ctrl_cells.X, 'todense') else np.asarray(ctrl_cells.X)

        for drug in ct_ood.obs['perturbation'].unique():
            if drug not in deg_dict[cell_type]:
                continue
            treated_cells = ct_ood[ct_ood.obs['perturbation'] == drug]
            if treated_cells.n_obs < MIN_CELLS_FOR_EVAL:
                continue

            x_true = np.asarray(treated_cells.X.todense()) if hasattr(treated_cells.X, 'todense') else np.asarray(treated_cells.X)
            x_pred = np.asarray(treated_cells.obsm[PRED_KEY])
            x_ctrl = x_ctrl_full
            deg_list = deg_dict[cell_type][drug]

            for n_top_deg in N_TOP_DEGS_LIST:
                if n_top_deg is not None:
                    degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
                    n_top_label = n_top_deg
                else:
                    degs = np.arange(adata.n_vars)
                    n_top_label = 'all'
                if len(degs) < 2:
                    continue

                x_true_deg = x_true[:, degs]
                x_pred_deg = x_pred[:, degs]
                x_ctrl_deg = x_ctrl[:, degs]

                true_mean = x_true_deg.mean(0)
                pred_mean = x_pred_deg.mean(0)
                ctrl_mean = x_ctrl_deg.mean(0)

                results['cell_type'].append(cell_type)
                results['perturbation'].append(drug)
                results['n_top_deg'].append(n_top_label)
                results['n_treated_cells'].append(treated_cells.n_obs)
                results['n_ctrl_cells'].append(ctrl_cells.n_obs)
                results['r2_mean_deg'].append(r2_score(true_mean, pred_mean))
                results['r2_var_deg'].append(r2_score(x_true_deg.var(0), x_pred_deg.var(0)))
                results['r2_mean_lfc_deg'].append(r2_score(true_mean - ctrl_mean, pred_mean - ctrl_mean))
                results['r2_var_lfc_deg'].append(r2_score(
                    x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0)
                ))

    df = pd.DataFrame(results)
    if df.empty:
        raise SystemExit("评估结果为空！检查ctrl_pool/deg_dict是否覆盖了ood_test里的细胞系和药物")

    # ---- 7. merge holdout元数据 ----
    merge_cols = ['drug', 'target', 'n_sibling_drugs_left', 'n_samples_held_out']
    # 兼容prepare_loo_data_structure_diverse.py产出的额外列（有就带上，没有就跳过）
    for optional_col in ['avg_similarity_to_siblings', 'is_structurally_distant']:
        if optional_col in holdout_df.columns:
            merge_cols.append(optional_col)

    df = df.merge(
        holdout_df[merge_cols],
        left_on='perturbation', right_on='drug', how='left'
    ).drop(columns=['drug'])

    df.to_csv(args.output_csv, index=False)

    print(f"\n=== {args.model_type} LOO评估结果 (ood_test held-out drugs) ===")
    print(df.groupby('n_top_deg')[['r2_mean_deg', 'r2_var_deg', 'r2_mean_lfc_deg', 'r2_var_lfc_deg']].mean())
    print(f"\n评估的 (cell_type, perturbation) 组合总数: {len(df[df['n_top_deg'] == 20])}")

    # ---- 8. 结构相似度 vs 预测质量 相关性分析（有structure_diverse数据时才跑） ----
    analyze_structure_similarity_correlation(df, holdout_df)

    print(f"\n结果已保存到: {args.output_csv}")
    print("\n下一步: 用 evaluate_by_drug.py 对kicpa和baseline两份输出分别按药物聚合，"
          "再对比两者在相同held-out药物上的差异；"
          "如果这次跑的是--pretrained_target_embeddings版本，建议再跑一遍不带预训练的版本"
          "(相同的held-out药物集合、相同seed)，对比两者的r2_mean_lfc_deg，"
          "量化预训练target embedding带来的增量")


if __name__ == "__main__":
    main()