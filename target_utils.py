"""
用法（方案B，推荐，在train_kicpa_loo.py里，build_drug_to_target_mapping()之后加）：

    from ki_cpa_pretrained_target_utils import build_pretrained_target_features

    drug_to_target, n_targets, target_name_to_idx = build_drug_to_target_mapping(...)

    pretrained_target_features = build_pretrained_target_features(
        npz_path="target_embeddings_esm2_raw.npz",
        target_name_to_idx=target_name_to_idx,
        n_targets=n_targets,
    )

    model = KineticsCPA(
        ...,
        pretrained_target_features=pretrained_target_features,  # 需要ki_cpa.py把这个参数
        ...,                                                      # 一路传给KineticsPerturbationNetwork
    )
"""

import numpy as np
import torch


def build_pretrained_target_matrix(npz_path, target_name_to_idx, n_targets, target_emb_dim,
                                    seed=434, dummy_init_std=0.01):
    """
    【方案A：预训练值当初始值，之后逐target独立微调】
    返回 torch.FloatTensor，形状 (n_targets, target_emb_dim)，
    传给 KineticsPerturbationNetwork(pretrained_target_embeddings=...)
    要求.npz是用 build_target_embeddings_esm2.py --reduction pca 生成的
    （即已经降维到target_emb_dim）
    """
    data = np.load(npz_path, allow_pickle=True)
    esm_target_ids = list(data["target_ids"])
    esm_embeddings = data["embeddings"]  # (n_esm_targets, saved_dim)
    saved_dim = esm_embeddings.shape[1]
    assert saved_dim == target_emb_dim, (
        f".npz里保存的维度({saved_dim})和当前target_emb_dim({target_emb_dim})不一致，"
        "要么重新跑build_target_embeddings_esm2.py --reduction pca时指定匹配的--target_emb_dim，"
        "要么改这里的target_emb_dim参数。如果这份.npz是用--reduction none生成的（方案B），"
        "应该用build_pretrained_target_features()而不是这个函数"
    )
    esm_lookup = {tid: esm_embeddings[i] for i, tid in enumerate(esm_target_ids)}

    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=dummy_init_std, size=(n_targets, target_emb_dim)).astype(np.float32)

    n_found, n_missing = 0, 0
    missing_list = []
    for target_name, idx in target_name_to_idx.items():
        if target_name in esm_lookup:
            matrix[idx] = esm_lookup[target_name]
            n_found += 1
        else:
            n_missing += 1
            missing_list.append(target_name)

    print(f"[方案A] 预训练target embedding对齐完成: {n_found}/{len(target_name_to_idx)} 个target命中，"
          f"{n_missing} 个退化为随机初始化")
    if missing_list:
        print(f"  缺失的target: {missing_list[:10]}{'...' if len(missing_list) > 10 else ''}")

    return torch.from_numpy(matrix)


def build_pretrained_target_features(npz_path, target_name_to_idx, n_targets):
    """
    【方案B（推荐）：ESM2原始特征永远冻结，单独学一个跨target共享的线性投影层】
    返回 torch.FloatTensor，形状 (n_targets, raw_dim)，raw_dim是ESM2原始输出维度
    （不做任何降维），传给 KineticsPerturbationNetwork(pretrained_target_features=...)
    要求.npz是用 build_target_embeddings_esm2.py --reduction none 生成的

    第0行(dummy/未知target)约定为全0向量——经过下游的nn.Linear投影层后，
    相当于只保留投影层的bias项，是所有未知target共享的、确定性的默认表示，
    不是随机噪声（这一点和方案A不同，方案A的dummy行是小随机值）
    """
    data = np.load(npz_path, allow_pickle=True)
    esm_target_ids = list(data["target_ids"])
    esm_embeddings = data["embeddings"]  # (n_esm_targets, raw_dim)
    raw_dim = esm_embeddings.shape[1]

    reduction = str(data["reduction"]) if "reduction" in data else "unknown"
    if reduction == "pca":
        print(f"警告: 这份.npz是用--reduction pca生成的(维度={raw_dim})，"
              f"用于方案B可能不是你想要的（方案B通常想要未降维的原始ESM2输出）。"
              "如果就是想用这个降维后的向量当固定特征也可以，只是不是推荐用法")

    esm_lookup = {tid: esm_embeddings[i] for i, tid in enumerate(esm_target_ids)}

    matrix = np.zeros((n_targets, raw_dim), dtype=np.float32)  # dummy行(idx=0)保持全0

    n_found, n_missing = 0, 0
    missing_list = []
    for target_name, idx in target_name_to_idx.items():
        if target_name in esm_lookup:
            matrix[idx] = esm_lookup[target_name]
            n_found += 1
        else:
            n_missing += 1
            missing_list.append(target_name)

    print(f"[方案B] 预训练target特征对齐完成: {n_found}/{len(target_name_to_idx)} 个target命中，"
          f"{n_missing} 个退化为全0向量(经投影层后等价于只用bias项)")
    if missing_list:
        print(f"  缺失的target: {missing_list[:10]}{'...' if len(missing_list) > 10 else ''}")

    return torch.from_numpy(matrix)