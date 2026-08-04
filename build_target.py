"""
  python build_target.py \
    --input_csv target_sequences_from_ttd.csv \
    --output_file target_embeddings_esm2_pca64.npz \
    --esm_model facebook/esm2_t12_35M_UR50D \
    --reduction pca
"""

import argparse
import os
import time
 
import numpy as np
import pandas as pd
import requests
import torch


def fetch_sequence_from_uniprot(uniprot_id, session, max_retries=3, sleep_sec=0.3):
    """从UniProt REST API拉取FASTA序列（作为TTD文件里sequence缺失时的补充手段），失败返回None"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and resp.text.startswith(">"):
                lines = resp.text.strip().split("\n")
                return "".join(lines[1:])
            elif resp.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(sleep_sec * (attempt + 1))
    return None


def collect_sequences(df, target_col, sequence_col, uniprot_col):
    """
    优先用df里的sequence_col；为空的行如果有uniprot_id就尝试从UniProt补拉。
    返回 ({target_id: sequence}, missing_target_ids)
    """
    seqs = {}
    missing = []
    need_fetch = []
 
    for _, row in df.iterrows():
        target_id = row[target_col]
        seq = row.get(sequence_col, "")
        if isinstance(seq, str) and len(seq) > 0:
            seqs[target_id] = seq
        else:
            need_fetch.append(row)
 
    if need_fetch:
        print(f"{len(need_fetch)} 个target在CSV里没有序列，尝试从UniProt补拉...")
        session = requests.Session()
        for i, row in enumerate(need_fetch):
            target_id = row[target_col]
            uniprot_id = row.get(uniprot_col, "")
            if not isinstance(uniprot_id, str) or len(uniprot_id) == 0:
                missing.append(target_id)
                continue
            seq = fetch_sequence_from_uniprot(uniprot_id, session)
            if seq is None:
                missing.append(target_id)
            else:
                seqs[target_id] = seq
            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(need_fetch)}")
 
    if missing:
        print(f"警告: {len(missing)} 个target最终没能获取到序列，将在对齐矩阵时退化为随机初始化: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
    print(f"最终获取到序列的target数: {len(seqs)}/{len(df)}")
    return seqs, missing
 
 
def compute_esm2_embeddings(seqs: dict, model_name: str, batch_size: int = 8,
                             max_length: int = 1024, device=None, hf_endpoint=None):
    """
    seqs: {target_id: sequence}
    返回 {target_id: np.ndarray(hidden_dim,)}，用mean pooling（按attention_mask加权平均）
    """
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"使用HuggingFace镜像: {hf_endpoint}")
 
    from transformers import AutoTokenizer, AutoModel
 
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载ESM2模型: {model_name} (device={device})")

    local_model_path = "/root/autodl-tmp/Research_Notebook/codes/cpa/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path).to(device).eval()
 
    target_ids = list(seqs.keys())
    embeddings = {}
 
    with torch.no_grad():
        for start in range(0, len(target_ids), batch_size):
            batch_ids = target_ids[start:start + batch_size]
            batch_seqs = [seqs[t][:max_length] for t in batch_ids]  # 超长序列截断，避免显存爆炸
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length + 2).to(device)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
 
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).cpu().numpy()
 
            for tid, emb in zip(batch_ids, mean_pooled):
                embeddings[tid] = emb
 
            print(f"  ESM2编码进度: {min(start + batch_size, len(target_ids))}/{len(target_ids)}")
 
    return embeddings


def standardize_embeddings(embeddings: dict):
    """
    对所有target的embedding做按维度的z-score标准化：(x - mean) / std，
    用这批target自己的统计量(population mean/std)。
 
    这一步很重要：ESM2原始输出的每个维度，数值尺度差异可能很大，
    如果不标准化直接做PCA，方差大的维度会不成比例地主导主成分方向，
    这个"方差大"很多时候只是任意的量纲问题，不代表信息量更大；
    如果是--reduction none直接喂给模型的可训练投影层，不标准化会导致
    输入尺度和模型其他部分(比如drug_fp、原来nn.Embedding的默认初始化
    尺度)不匹配，训练一开始就可能不稳定。
 
    返回 (标准化后的{target_id: vector}, mean数组, std数组)
    """
    target_ids = list(embeddings.keys())
    matrix = np.stack([embeddings[t] for t in target_ids])  # (n_targets, raw_dim)
 
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_safe = np.where(std < 1e-8, 1.0, std)  # 避免除以接近0的标准差（比如某维度所有target都一样）
 
    standardized = (matrix - mean) / std_safe
    return {t: standardized[i] for i, t in enumerate(target_ids)}, mean, std_safe
 
 
def reduce_dim_pca(embeddings: dict, target_emb_dim: int):
    """对embedding字典做PCA降维到target_emb_dim，返回 (降维后的{target_id: vector}, 拟合好的PCA对象)"""
    from sklearn.decomposition import PCA
 
    target_ids = list(embeddings.keys())
    matrix = np.stack([embeddings[t] for t in target_ids])  # (n_targets, raw_dim)
 
    n_components = min(target_emb_dim, matrix.shape[0], matrix.shape[1])
    if n_components < target_emb_dim:
        print(f"警告: target数({matrix.shape[0]})或原始维度({matrix.shape[1]})小于"
              f"target_emb_dim({target_emb_dim})，PCA只能降到{n_components}维，会补0到{target_emb_dim}维")
 
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA降维: {matrix.shape[1]} -> {n_components} 维，保留了 {explained:.1%} 的方差")
 
    if n_components < target_emb_dim:
        pad = np.zeros((reduced.shape[0], target_emb_dim - n_components))
        reduced = np.concatenate([reduced, pad], axis=1)
 
    return {t: reduced[i] for i, t in enumerate(target_ids)}, pca
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True,
                         help="parse_ttd_target_file.py的输出CSV（target_id,uniprot_id,target_name,sequence）")
    parser.add_argument("--target_col", default="target_id")
    parser.add_argument("--sequence_col", default="sequence")
    parser.add_argument("--uniprot_col", default="uniprot_id")
    parser.add_argument("--output_file", required=True, help="输出.npz文件路径")
    parser.add_argument("--esm_model", default="facebook/esm2_t12_35M_UR50D",
                         help="ESM2模型名（HuggingFace），越大越准但越慢。"
                              "小: facebook/esm2_t6_8M_UR50D (320维,最快)；"
                              "中(推荐): facebook/esm2_t12_35M_UR50D (480维)；"
                              "大: facebook/esm2_t33_650M_UR50D (1280维,效果通常更好但慢很多)")
    parser.add_argument("--target_emb_dim", type=int, default=64,
                         help="仅在--reduction pca时生效：PCA降到的维度，需要和train脚本里"
                              "target_emb_dim参数保持一致（用于'方案A：预训练值当初始值再微调'）")
    parser.add_argument("--reduction", choices=["pca", "none"], default="pca",
                         help="pca(默认): 降维到--target_emb_dim，配合'方案A'（预训练值当初始值，"
                              "之后逐target独立微调）使用；"
                              "none: 不降维，直接保存ESM2原始输出(通常480/1280维)，"
                              "配合'方案B'（冻结ESM2特征+单独学一个跨target共享的线性投影层，"
                              "见_utils.py里的pretrained_target_features参数）使用，推荐这个")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024, help="蛋白序列最大长度，超长截断")
    args = parser.parse_args()
 
    df = pd.read_csv(args.input_csv)
    assert args.target_col in df.columns, f"CSV里找不到列 '{args.target_col}'"
    df = df.drop_duplicates(subset=[args.target_col])
    print(f"共 {len(df)} 个target")
 
    print("\n=== 第1步: 获取蛋白序列 ===")
    seqs, missing_targets = collect_sequences(df, args.target_col, args.sequence_col, args.uniprot_col)
    assert len(seqs) > 0, "一个序列都没获取到，检查CSV内容是否正确"
 
    print("\n=== 第2步: 计算ESM2 embedding ===")
    raw_embeddings = compute_esm2_embeddings(
        seqs, args.esm_model, batch_size=args.batch_size, max_length=args.max_length
    )
 
    print("\n=== 第3步: 按维度z-score标准化 ===")
    standardized_embeddings, feat_mean, feat_std = standardize_embeddings(raw_embeddings)
    print(f"标准化完成，原始尺度举例（前5维）: mean={feat_mean[:5]}, std={feat_std[:5]}")
 
    print("\n=== 第4步: 降维（可选） ===")
    if args.reduction == "pca":
        reduced_embeddings, pca = reduce_dim_pca(standardized_embeddings, args.target_emb_dim)
        saved_dim = args.target_emb_dim
        print("方案A：标准化+PCA降维后的向量将作为target_embedding的初始值，训练时逐target独立微调")
    else:
        reduced_embeddings = standardized_embeddings
        saved_dim = next(iter(standardized_embeddings.values())).shape[0]
        print(f"方案B：不做降维，保存标准化后的ESM2原始输出({saved_dim}维)，"
              "训练时会用一个跨target共享的可训练线性层(前面还有一层LayerNorm做二次校准)做投影，"
              "ESM2特征本身保持冻结")
 
    target_ids = list(reduced_embeddings.keys())
    matrix = np.stack([reduced_embeddings[t] for t in target_ids])
 
    np.savez(
        args.output_file,
        target_ids=np.array(target_ids, dtype=object),
        embeddings=matrix,
        missing_targets=np.array(missing_targets, dtype=object),
        esm_model=args.esm_model,
        target_emb_dim=saved_dim,
        reduction=args.reduction,
        feature_mean=feat_mean,  # 标准化用的统计量，存下来方便复现/调试，对齐时不需要用到
        feature_std=feat_std,
    )
    print(f"\n已保存: {args.output_file}")
    print(f"覆盖target数: {len(target_ids)}，缺失(将在训练时退化为随机初始化)的target数: {len(missing_targets)}")
    if args.reduction == "pca":
        print(
            "\n下一步（方案A）：在train_kicpa_loo.py里用build_pretrained_target_matrix()"
            "（见ki_cpa_pretrained_target_utils.py）把这份.npz和你数据里实际用到的"
            "target_name_to_idx对齐，传给pretrained_target_embeddings参数"
        )
    else:
        print(
            "\n下一步（方案B，推荐）：在train_kicpa_loo.py里用build_pretrained_target_features()"
            "（见ki_cpa_pretrained_target_utils.py）把这份.npz对齐，"
            "传给pretrained_target_features参数"
        )
 
 
if __name__ == "__main__":
    main()