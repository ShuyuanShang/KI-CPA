"""
  # 方案1: 结构差异优先（报告第6.1节，测试"结构不像但target对"的泛化能力）
  python prepare_loo_data.py \
      --input_file /root/autodl-tmp/data/single_target_drug_GSE70138.h5ad \
      --output_file /root/autodl-tmp/data/adata_loo_structure_diverse.h5ad \
      --selection_mode structure_diverse \
      --min_drugs_per_target 3 --seed 0

  # 方案2: 跨target结构相似优先（报告第6.2节，压力测试"会不会被结构骗"）
  python prepare_loo_data.py \
      --input_file /root/autodl-tmp/data/single_target_drug_GSE70138.h5ad \
      --output_file /root/autodl-tmp/data/adata_loo_cross_target_similar.h5ad \
      --selection_mode cross_target_similar \
      --min_drugs_per_target 3 --seed 0

  # 对照组：随机选（等价于旧版prepare_loo_data.py）
  python prepare_loo_data.py \
      --input_file /root/autodl-tmp/data/single_target_drug_GSE70138.h5ad \
      --output_file /root/autodl-tmp/data/adata_loo_random.h5ad \
      --selection_mode random \
      --min_drugs_per_target 3 --seed 0
"""

import argparse

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog('rdApp.*')
except ImportError as e:
    raise ImportError(
        "structure_diverse/cross_target_similar模式需要rdkit：pip install rdkit"
        "（如果在训练用的conda环境里跑，通常已经装过了）"
    ) from e

CONTROL_PERT_NAME = 'DMSO'
CORE_CELL_LINES = ['MCF7', 'A375', 'HA1E', 'HT29', 'PC3', 'YAPC', 'HELA']
VALID_PERT_TIME = 24.0
N_TOP_GENES = 2000
DOSE_SENTINEL = -666
TARGET_COL = "TTD Target ID"
# 来自《药物结构相似度与靶点关系统计分析报告》第5.2节：
# 不同target组相似度分布的第25分位数，作为"结构异常不像"的参考基准（structure_diverse模式用）
DEFAULT_DISTANT_THRESHOLD = 0.0723
# 同target组相似度分布的第90分位数，作为"结构异常相似"的参考基准（cross_target_similar模式用）
DEFAULT_CONFUSABLE_THRESHOLD = 0.3975


def compute_fingerprints(obs_df, drug_col, smiles_col, radius=2, n_bits=2048):
    """对每个药物(去重)计算Morgan指纹，返回 {drug_name: fp}，解析失败的跳过"""
    unique_drugs = obs_df[[drug_col, smiles_col]].drop_duplicates(subset=[drug_col])
    fps = {}
    invalid = []
    for _, row in unique_drugs.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            invalid.append(row[drug_col])
            continue
        fps[row[drug_col]] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    if invalid:
        print(f"警告: {len(invalid)} 个药物的SMILES无法被RDKit解析，"
              f"structure_diverse选药时会跳过这些药物: {invalid[:10]}{'...' if len(invalid) > 10 else ''}")
    return fps


def compute_max_cross_target_similarity(obs_df, drug_col, target_col, smiles_col,
                                         control_value=CONTROL_PERT_NAME,
                                         radius=2, n_bits=2048):
    """
    对每个(有target标注的)药物，计算它与"所有不同target的药物"里
    结构最相似的那一个，返回：
      {drug: {"max_cross_target_similarity": float,
              "most_similar_drug": str,
              "most_similar_target": str}}

    用于cross_target_similar选药模式——挑结构上被别的target"传染"得
    最厉害的药物做held-out，直接压力测试模型是不是在用结构相似性
    抄近路，而不是真的在用target信息做泛化。
    """
    df = obs_df[[drug_col, smiles_col, target_col]].drop_duplicates(subset=[drug_col])
    df = df[df[drug_col] != control_value].dropna(subset=[smiles_col, target_col])

    drugs = df[drug_col].tolist()
    targets = df[target_col].tolist()
    fps = []
    valid_idx = []
    invalid = []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            invalid.append(row[drug_col])
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        valid_idx.append(i)

    if invalid:
        print(f"警告: {len(invalid)} 个药物的SMILES无法解析，不参与cross_target相似度计算: "
              f"{invalid[:10]}{'...' if len(invalid) > 10 else ''}")

    valid_df = df.loc[valid_idx].reset_index(drop=True)
    n = len(fps)
    result = {}
    for i in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        same_target_mask = (valid_df[target_col].values == valid_df[target_col].iloc[i])
        sims_masked = sims.copy()
        sims_masked[same_target_mask] = -1.0  # 排除同target的药物(含自己)
        if (sims_masked > -1.0).any():
            j = int(np.argmax(sims_masked))
            result[valid_df[drug_col].iloc[i]] = {
                "max_cross_target_similarity": float(sims_masked[j]),
                "most_similar_drug": valid_df[drug_col].iloc[j],
                "most_similar_target": valid_df[target_col].iloc[j],
            }
    return result


def compute_avg_sibling_similarity(mapping, fps):
    """
    mapping: {target: [drug, ...]}（同prepare_loo_data.py里的写法）
    返回 {drug: avg_tanimoto_similarity_to_siblings}，指纹解析失败或
    没有可比较的兄弟药物时该药物不出现在返回值里
    """
    avg_sim = {}
    for target, drugs in mapping.items():
        valid_drugs = [d for d in drugs if d in fps]
        for d in valid_drugs:
            others = [o for o in valid_drugs if o != d]
            if not others:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fps[d], [fps[o] for o in others])
            avg_sim[d] = float(np.mean(sims))
    return avg_sim


def assign_split_loo(obs_df, target_col=TARGET_COL, drug_col='perturbation',
                      control_value=CONTROL_PERT_NAME, min_drugs_per_target=3,
                      valid_frac=0.1, min_cells_for_split=10, seed=434,
                      selection_mode='structure_diverse', smiles_col='SMILES',
                      fingerprint_radius=2, fingerprint_bits=2048,
                      distant_threshold=DEFAULT_DISTANT_THRESHOLD,
                      confusable_threshold=DEFAULT_CONFUSABLE_THRESHOLD):
    """
    返回 (split: pd.Series, holdout_df: pd.DataFrame)
    split 取值: 'train' / 'valid' / 'ood_test'

    selection_mode:
      'random': 每个target下随机选held-out药物（旧版prepare_loo_data.py的行为）
      'structure_diverse': 优先选与"兄弟药物"(同target下其他药物)结构差异最大的，
          测试模型是否真的在利用target信息做泛化，而不是靠结构相似性抄近路
      'cross_target_similar': 优先选与"某个不同target的药物"结构异常相似的，
          直接压力测试模型会不会"被结构骗"——预测出来的效应更像它的结构近邻，
          而不是它真实标注的target该有的效应
    """
    rng = np.random.default_rng(seed)

    # ---- 1. 找出每个target下的候选药物 ----
    sub = obs_df[[target_col, drug_col]].dropna(subset=[target_col])
    sub = sub[sub[drug_col] != control_value]
    mapping = sub.groupby(target_col, observed=True)[drug_col].unique().apply(
        lambda x: sorted(set(x))
    ).to_dict()

    eligible_targets = {t: drugs for t, drugs in mapping.items() if len(drugs) >= min_drugs_per_target}
    print(f"满足 n_drugs>={min_drugs_per_target} 的靶点数: {len(eligible_targets)}")

    if selection_mode not in ('random', 'structure_diverse', 'cross_target_similar'):
        raise ValueError(f"未知selection_mode: {selection_mode}")

    # ---- 2. 计算结构相似度（按选药模式各自需要的统计量） ----
    avg_sim = {}
    cross_target_info = {}
    if selection_mode == 'structure_diverse':
        print("计算Morgan指纹 + 兄弟药物平均相似度...")
        fps = compute_fingerprints(obs_df, drug_col, smiles_col, fingerprint_radius, fingerprint_bits)
        avg_sim = compute_avg_sibling_similarity(eligible_targets, fps)
    elif selection_mode == 'cross_target_similar':
        print("计算Morgan指纹 + 跨target最大相似度...")
        cross_target_info = compute_max_cross_target_similarity(
            obs_df, drug_col, target_col, smiles_col, control_value,
            radius=fingerprint_radius, n_bits=fingerprint_bits,
        )

    # ---- 3. 逐target选held-out药物 ----
    holdout = {}  # drug -> target
    holdout_meta = {}  # drug -> 附加信息dict（不同selection_mode字段不同）
    for target, drugs in eligible_targets.items():
        if selection_mode == 'random':
            chosen = rng.choice(drugs)
            holdout_meta[chosen] = {}

        elif selection_mode == 'structure_diverse':
            candidates = [d for d in drugs if d in avg_sim]
            if not candidates:
                print(f"警告: target {target} 下所有药物指纹解析失败，退化为随机选")
                chosen = rng.choice(drugs)
                holdout_meta[chosen] = {"avg_similarity_to_siblings": None}
            else:
                min_sim = min(avg_sim[d] for d in candidates)
                tied = [d for d in candidates if abs(avg_sim[d] - min_sim) < 1e-9]
                chosen = rng.choice(tied)
                holdout_meta[chosen] = {"avg_similarity_to_siblings": avg_sim[chosen]}

        else:  # cross_target_similar
            candidates = [d for d in drugs if d in cross_target_info]
            if not candidates:
                print(f"警告: target {target} 下所有药物指纹解析失败，退化为随机选")
                chosen = rng.choice(drugs)
                holdout_meta[chosen] = {
                    "max_cross_target_similarity": None,
                    "most_similar_cross_target_drug": None,
                    "most_similar_cross_target_target": None,
                }
            else:
                max_sim = max(cross_target_info[d]["max_cross_target_similarity"] for d in candidates)
                tied = [d for d in candidates
                        if abs(cross_target_info[d]["max_cross_target_similarity"] - max_sim) < 1e-9]
                chosen = rng.choice(tied)
                info = cross_target_info[chosen]
                holdout_meta[chosen] = {
                    "max_cross_target_similarity": info["max_cross_target_similarity"],
                    "most_similar_cross_target_drug": info["most_similar_drug"],
                    "most_similar_cross_target_target": info["most_similar_target"],
                }

        holdout[chosen] = target

    holdout_rows = []
    for drug, target in sorted(holdout.items()):
        n_sibling = len(mapping[target]) - 1
        n_samples = int((obs_df[drug_col] == drug).sum())
        row = {
            "drug": drug, "target": target, "n_sibling_drugs_left": n_sibling,
            "n_samples_held_out": n_samples,
        }
        row.update(holdout_meta[drug])
        if selection_mode == 'structure_diverse':
            sim = holdout_meta[drug].get("avg_similarity_to_siblings")
            row["is_structurally_distant"] = (sim is not None and sim < distant_threshold)
        elif selection_mode == 'cross_target_similar':
            sim = holdout_meta[drug].get("max_cross_target_similarity")
            row["is_structurally_confusable"] = (sim is not None and sim > confusable_threshold)
        holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows).sort_values("target").reset_index(drop=True)

    if selection_mode == 'structure_diverse':
        n_distant = holdout_df["is_structurally_distant"].sum()
        print(f"\nheld-out药物里，与兄弟药物平均相似度 < {distant_threshold}（'结构异常不像'）的有 "
              f"{n_distant}/{len(holdout_df)} 个")
        print(f"held-out药物平均相似度分布:\n{holdout_df['avg_similarity_to_siblings'].describe()}")
    elif selection_mode == 'cross_target_similar':
        n_confusable = holdout_df["is_structurally_confusable"].sum()
        print(f"\nheld-out药物里，与某个不同target药物相似度 > {confusable_threshold}（'结构异常相似'）的有 "
              f"{n_confusable}/{len(holdout_df)} 个")
        print(f"held-out药物跨target最大相似度分布:\n{holdout_df['max_cross_target_similarity'].describe()}")

    # ---- 4. 标记 ood_test ----
    is_holdout_drug = obs_df[drug_col].isin(holdout.keys())
    split = pd.Series('train', index=obs_df.index, dtype=object)
    split[is_holdout_drug] = 'ood_test'

    # ---- 5. 对剩下的行(非held-out药物 + control)做train/valid分层切分 ----
    remaining_mask = ~is_holdout_drug
    remaining_obs = obs_df.loc[remaining_mask]
    group_key = remaining_obs['cell_type'].astype(str) + '_' + remaining_obs[drug_col].astype(str)
    for cat, idx in remaining_obs.groupby(group_key).groups.items():
        idx = np.array(idx)
        if len(idx) < min_cells_for_split:
            continue
        rng.shuffle(idx)
        n_valid = max(1, int(len(idx) * valid_frac))
        split.loc[idx[:n_valid]] = 'valid'

    return split, holdout_df


def main():
    parser = argparse.ArgumentParser(description="生成leave-one-drug-out划分的h5ad（支持结构差异优先选held-out药物）")
    parser.add_argument("--input_file", default='/root/autodl-tmp/data/single_target_drug_GSE70138.h5ad')
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_drugs_per_target", type=int, default=3)
    parser.add_argument("--valid_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection_mode", choices=["random", "structure_diverse", "cross_target_similar"],
                         default="structure_diverse",
                         help="structure_diverse(默认): 每个target下优先选与兄弟药物结构最不像的做held-out；"
                              "cross_target_similar: 优先选与某个不同target药物结构异常相似的做held-out；"
                              "random: 旧版prepare_loo_data.py的随机选，留作对照实验")
    parser.add_argument("--fingerprint_radius", type=int, default=2)
    parser.add_argument("--fingerprint_bits", type=int, default=2048)
    parser.add_argument("--distant_threshold", type=float, default=DEFAULT_DISTANT_THRESHOLD,
                         help="仅structure_diverse模式：用于打标'is_structurally_distant'的参考阈值，"
                              "默认取自统计报告里不同target组相似度分布的25分位数")
    parser.add_argument("--confusable_threshold", type=float, default=DEFAULT_CONFUSABLE_THRESHOLD,
                         help="仅cross_target_similar模式：用于打标'is_structurally_confusable'的参考阈值，"
                              "默认取自统计报告里同target组相似度分布的90分位数")
    args = parser.parse_args()

    print(f"读取数据: {args.input_file}")
    adata = anndata.read_h5ad(args.input_file)
    adata_processed = adata.copy()

    # ---- 与train_data.py / prepare_loo_data.py完全一致的预处理 ----
    adata_processed.obs['perturbation'] = (
        adata_processed.obs['pert_iname'].astype(str)
        .str.replace('+/-', 'racemic', regex=False)
        .str.replace('+', 'plus', regex=False)
    ).astype('category')
    assert not adata_processed.obs['perturbation'].astype(str).str.contains(r'\+', regex=True).any()

    condition_series = adata_processed.obs['pert_type'].astype(str)
    adata_processed.obs['condition'] = condition_series.map(
        {'ctl_vehicle': 'control', 'trt_cp': 'treated'}
    ).astype('category')

    adata_processed.obs['dose_val'] = adata_processed.obs['pert_dose'].astype(float)
    dmso_mask = adata_processed.obs['perturbation'] == CONTROL_PERT_NAME
    adata_processed.obs.loc[dmso_mask, 'dose_val'] = 0.0
    assert (adata_processed.obs['dose_val'] == DOSE_SENTINEL).sum() == 0

    adata_processed.obs['cell_type'] = adata_processed.obs['cell_id'].astype(str)
    adata_processed = adata_processed[adata_processed.obs['cell_type'].isin(CORE_CELL_LINES)].copy()
    adata_processed = adata_processed[adata_processed.obs['pert_time'] == VALID_PERT_TIME].copy()

    x_min, x_max = adata_processed.X.min(), adata_processed.X.max()
    assert 0 <= x_min and x_max <= 20

    adata_processed = adata_processed[:, adata_processed.var['pr_is_bing'] == 1].copy()
    sc.pp.highly_variable_genes(adata_processed, n_top_genes=N_TOP_GENES, subset=True)

    adata_processed.obs['SMILES'] = adata_processed.obs['SMILES'].astype(object)
    adata_processed.obs.loc[
        adata_processed.obs['perturbation'] == CONTROL_PERT_NAME, 'SMILES'
    ] = 'CS(C)=O'

    print(f"预处理后数据形状: {adata_processed.shape}")
    print(f"选药策略: {args.selection_mode}")

    # ---- LOO 切分 ----
    split, holdout_df = assign_split_loo(
        adata_processed.obs,
        min_drugs_per_target=args.min_drugs_per_target,
        valid_frac=args.valid_frac,
        seed=args.seed,
        selection_mode=args.selection_mode,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        distant_threshold=args.distant_threshold,
        confusable_threshold=args.confusable_threshold,
    )
    adata_processed.obs['split'] = split.astype('category')

    print("\n" + "=" * 70)
    print(f"held-out 药物数: {len(holdout_df)}")
    print("=" * 70)
    print(holdout_df.to_string(index=False))

    if len(holdout_df) == 0:
        raise SystemExit(
            f"没有靶点满足 n_drugs>={args.min_drugs_per_target}，"
            "请调低 --min_drugs_per_target 或检查数据"
        )

    if holdout_df["n_samples_held_out"].min() < 20:
        print("\n⚠️ 部分held-out药物样本数<20，评估其R²时方差会较大")

    print(f"\nsplit列分布:\n{adata_processed.obs['split'].value_counts()}")

    # 断言：每个细胞系在valid里都要有DMSO control（沿用train_data.py同款检查）
    valid_ctrl_counts = adata_processed.obs[
        (adata_processed.obs['split'] == 'valid') &
        (adata_processed.obs['perturbation'] == CONTROL_PERT_NAME)
    ].groupby('cell_type', observed=True).size()
    assert (valid_ctrl_counts > 0).all(), (
        "有细胞系在valid split里没有control细胞！调整valid_frac或min_cells_for_split"
    )

    adata_processed.write(args.output_file)
    holdout_csv = args.output_file.rsplit('.', 1)[0] + '_holdout_drugs.csv'
    holdout_df.to_csv(holdout_csv, index=False)
    print(f"\n已保存: {args.output_file}")
    print(f"held-out药物明细(含{args.selection_mode}模式对应的诊断列): {holdout_csv}")
    print(
        "\n下一步：train_kicpa_loo.py / train_baseline_loo.py 不需要做任何改动，"
        f"直接加载 {args.output_file}，用法和之前完全一样。"
        "\n最终评估用final_eval_loo.py，会自动识别holdout_csv里的额外列做对应诊断"
        "（structure_diverse模式看avg_similarity_to_siblings相关性分析；"
        "cross_target_similar模式建议额外检查这些held-out药物的预测结果，"
        "是不是更像most_similar_cross_target_drug的效应而不是自己target该有的效应）"
        "\n建议做法：用相同的--seed，分别跑random/structure_diverse/cross_target_similar"
        "三种selection_mode，各自训练+评估后对比final_eval_loo.py输出的r2_mean_lfc_deg，"
        "系统性地检验模型对结构shortcut的依赖程度"
    )


if __name__ == "__main__":
    main()