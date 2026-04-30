import argparse
import importlib
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model.transformer import Seq2SeqTransformer
from tokenizer import BaseTokenizer
from utils import translate_sentence
import yaml


def load_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_tokenizer(cfg):
    mod, cls = cfg.get("tokenizer", "tokenizer.JiebaEnTokenizer").rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)()


def visualize_attention(attn_map, src_tokens, tgt_tokens, layer, head, save_path):
    attn_map = attn_map.detach().cpu().numpy()

    sns.set()
    plt.figure(figsize=(len(src_tokens) * 0.5, len(tgt_tokens) * 0.5))
    sns.heatmap(attn_map, xticklabels=src_tokens, yticklabels=tgt_tokens, cmap="viridis")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.title(f"Decoder Layer {layer}, Head {head} Attention")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved attention heatmap to {save_path}")


def main():
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--save_path", default="attn_map.png")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(cfg)
    with open("data/processed/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("data/processed/tgt_vocab.pkl", "rb") as f:
        tgt_vocab = pickle.load(f)
    tokenizer.set_vocab(src_vocab, tgt_vocab)

    src_ids = tokenizer.encode_src(args.src)
    tgt_ids = tokenizer.encode_tgt(args.tgt)
    src_ids = torch.tensor([src_ids], dtype=torch.long, device=device)  # (1, L_src)
    tgt_ids = torch.tensor([tgt_ids], dtype=torch.long, device=device)  # (1, L_tgt)

    mcfg = cfg["model"]
    model = Seq2SeqTransformer(
        num_encoder_layers=mcfg["enc_layers"],
        num_decoder_layers=mcfg["dec_layers"],
        emb_size=mcfg["emb_size"],
        nhead=mcfg["nhead"],
        src_vocab_size=tokenizer.src_vocab_size,
        tgt_vocab_size=tokenizer.tgt_vocab_size,
        dim_feedforward=mcfg["ffn_dim"],
        dropout=mcfg.get("dropout", 0.1),
        pad_id=tokenizer.pad_token_id,
    ).to(device)

    checkpoint = torch.load(args.pt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    with torch.no_grad():
        print("start")
        output = model(src_ids, tgt_ids, return_attn=True)

    if isinstance(output, tuple):
        logits, attns = output
    else:
        print("模型未返回 attention")
        return
    print("attns:", attns)
    dec_cross_attns = attns.get("cross", None)
    if dec_cross_attns is None:
        print("没有找到 decoder cross attention")
        return

    src_tokens = [tokenizer.src_id2tok.get(i, "<unk>") for i in src_ids[0].cpu().tolist()]
    tgt_tokens = [tokenizer.tgt_id2tok.get(i, "<unk>") for i in tgt_ids[0].cpu().tolist()]


    if args.layer < 0 or args.layer >= len(dec_cross_attns):
        raise ValueError(f"Invalid layer index {args.layer}, must be in [0, {len(dec_cross_attns)-1}]")
    attn_layer = dec_cross_attns[args.layer]
    attn_map = attn_layer[0, args.head]
    if args.head < 0 or args.head >= attn_layer.shape[1]:
        raise ValueError(f"Invalid head index {args.head}, must be in [0, {attn_layer.shape[1]-1}]")

    visualize_attention(attn_map, src_tokens, tgt_tokens, args.layer, args.head, args.save_path)

if __name__ == "__main__":
    main()