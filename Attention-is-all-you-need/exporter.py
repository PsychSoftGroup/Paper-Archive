# -*- coding: utf-8 -*-
# 将你已训练的 Python 端工件导出为 C++ 端可直接读取的三件套：
# 1) config.bin（int32 小端；顺序：vocab_size, d_model, n_layers, d_ff, max_len, eos_id, unk_id）
# 2) vocab.txt（每行一个 token，行号即 id；从 vocab.json 构造）
# 3) weights.bin（float32 小端，row-major，按固定顺序拼接）
#
# 注意：本导出脚本匹配你当前“最初版本”的 Python 模型结构与命名：
# - Embedding: we.weight
# - 每层：
#   - blocks.{i}.ln1.weight / blocks.{i}.ln1.bias
#   - blocks.{i}.attn.W_q.weight / W_k.weight / W_v.weight
#   - blocks.{i}.ln2.weight / blocks.{i}.ln2.bias
#   - blocks.{i}.ffn.fc1.weight / fc1.bias / fc2.weight / fc2.bias
# - 末端：ln_f.weight / ln_f.bias / head.weight / head.bias
#
# 位置编码（正弦）在 C++ 端运行时重建（和 Python 公式一致），无需导出。
#
# 用法示例（在 ipynb 中新建 cell 执行）：
#   !python export_for_c_infer.py --run_dir runs_ipynb/test
#
import os
import json
import argparse
import struct
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="包含 vocab.json / config.json / transformer_state.pt 的目录")
    args = ap.parse_args()

    run_dir = args.run_dir
    vocab_json = os.path.join(run_dir, "vocab.json")
    config_json = os.path.join(run_dir, "config.json")
    state_pt   = os.path.join(run_dir, "transformer_state.pt")

    assert os.path.isfile(vocab_json), f"not found: {vocab_json}"
    assert os.path.isfile(config_json), f"not found: {config_json}"
    assert os.path.isfile(state_pt),   f"not found: {state_pt}"

    # 读取 config / vocab
    with open(config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(vocab_json, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)

    vocab_size = int(cfg["vocab_size"])
    d_model    = int(cfg["d_model"])
    n_layers   = int(cfg["n_layers"])
    d_ff       = int(cfg["d_ff"])
    max_len    = int(cfg["max_len"])

    # 找出 eos_id / unk_id
    eos_id = token_to_id.get("<|endoftext|>")
    unk_id = token_to_id.get("<UNK>")
    if eos_id is None or unk_id is None:
        raise ValueError("vocab.json 中未找到 <|endoftext|> 或 <UNK>")

    # 写 vocab.txt（按 id 排）
    id_to_token = [""] * vocab_size
    for tok, idx in token_to_id.items():
        idx = int(idx)
        if 0 <= idx < vocab_size:
            id_to_token[idx] = tok
    vocab_txt = os.path.join(run_dir, "vocab.txt")
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for tok in id_to_token:
            f.write(tok + "\n")

    # 写 config.bin（int32 小端）
    config_bin = os.path.join(run_dir, "config.bin")
    with open(config_bin, "wb") as f:
        for v in [vocab_size, d_model, n_layers, d_ff, max_len, eos_id, unk_id]:
            f.write(struct.pack("<i", int(v)))

    # 读取 state_dict
    state = torch.load(state_pt, map_location="cpu")
    # 写 weights.bin（float32 小端、row-major、按固定顺序）
    weights_bin = os.path.join(run_dir, "weights.bin")

    def write_tensor(arr: np.ndarray, fh):
        arr = np.asarray(arr, dtype=np.float32, order="C")
        fh.write(arr.tobytes(order="C"))

    with open(weights_bin, "wb") as f:
        # 1) we.weight [vocab_size, d_model]
        write_tensor(state["we.weight"].numpy(), f)

        # 2) 每层
        for i in range(n_layers):
            # ln1
            write_tensor(state[f"blocks.{i}.ln1.weight"].numpy(), f)  # gamma
            write_tensor(state[f"blocks.{i}.ln1.bias"].numpy(),   f)  # beta
            # attn
            write_tensor(state[f"blocks.{i}.attn.W_q.weight"].numpy(), f)  # [d_model, d_model] (out, in)
            write_tensor(state[f"blocks.{i}.attn.W_k.weight"].numpy(), f)
            write_tensor(state[f"blocks.{i}.attn.W_v.weight"].numpy(), f)
            # ln2
            write_tensor(state[f"blocks.{i}.ln2.weight"].numpy(), f)
            write_tensor(state[f"blocks.{i}.ln2.bias"].numpy(),   f)
            # ffn
            write_tensor(state[f"blocks.{i}.ffn.fc1.weight"].numpy(), f)  # [d_ff, d_model]
            write_tensor(state[f"blocks.{i}.ffn.fc1.bias"].numpy(),   f)  # [d_ff]
            write_tensor(state[f"blocks.{i}.ffn.fc2.weight"].numpy(), f)  # [d_model, d_ff]
            write_tensor(state[f"blocks.{i}.ffn.fc2.bias"].numpy(),   f)  # [d_model]

        # 3) ln_f
        write_tensor(state["ln_f.weight"].numpy(), f)
        write_tensor(state["ln_f.bias"].numpy(),   f)

        # 4) head
        write_tensor(state["head.weight"].numpy(), f)  # [vocab_size, d_model]
        write_tensor(state["head.bias"].numpy(),   f)  # [vocab_size]

    print("Exported to:")
    print(" -", config_bin)
    print(" -", vocab_txt)
    print(" -", weights_bin)


main()