"""
  python parse_ttd.py \
      --input_file /root/autodl-tmp/data/P1-01-TTD_target_download.txt \
      --output_file target_sequences_from_ttd.csv
"""

import argparse

import pandas as pd


def parse_ttd_target_file(path):
    """
    返回 {target_id: {"uniprot_id":..., "target_name":..., "sequence":...}}
    只提取我们需要的三个字段，DRUGINFO等其他字段忽略
    """
    targets = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.lstrip().startswith("---"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            target_id, field = parts[0].strip(), parts[1].strip()
            value = "\t".join(parts[2:]).strip()

            # 过滤掉文件开头"Abbreviations:"那段说明性的表头
            # （表头行第一列是"TARGETID"/"UNIPROID"这些字段名本身，
            # 不是真正的target id，用"Txxxxx"这个格式校验来排除它们）
            if not (target_id.startswith("T") and target_id[1:].isdigit()):
                continue

            entry = targets.setdefault(target_id, {})
            if field == "SEQUENCE":
                entry["sequence"] = value
            elif field == "UNIPROID":
                entry["uniprot_id"] = value
            elif field == "TARGNAME":
                entry["target_name"] = value
            # DRUGINFO等其他字段忽略，用不到

    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="TTD官方下载的target文件，如P1-01-TTD_target_download.txt")
    parser.add_argument("--output_file", required=True, help="输出CSV，列为target_id,uniprot_id,target_name,sequence")
    args = parser.parse_args()

    print(f"解析: {args.input_file}")
    targets = parse_ttd_target_file(args.input_file)
    print(f"共解析到 {len(targets)} 个target条目")

    rows = []
    n_with_seq = 0
    for target_id, info in targets.items():
        has_seq = "sequence" in info and len(info["sequence"]) > 0
        if has_seq:
            n_with_seq += 1
        rows.append({
            "target_id": target_id,
            "uniprot_id": info.get("uniprot_id", ""),
            "target_name": info.get("target_name", ""),
            "sequence": info.get("sequence", ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_file, index=False)
    print(f"其中 {n_with_seq}/{len(targets)} 个target带有SEQUENCE字段")
    print(f"已保存: {args.output_file}")
    print("\n下一步：python build_target_embeddings_esm2.py --input_csv "
          f"{args.output_file} --output_file target_embeddings_esm2.npz")


if __name__ == "__main__":
    main()