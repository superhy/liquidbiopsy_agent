#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from liquidbiopsy_agent.utils.storage import get_data_root

LOGGER = logging.getLogger("methylgpt_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal runnable demo for MethylGPTEncoder.")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="weights/methylgpt/pretrained.pt",
        help="Checkpoint path relative to LIQUID_BIOPSY_DATA_ROOT (or absolute path under that root).",
    )
    return parser.parse_args()


def build_demo_vocab(n_cpg_tokens: int = 1024):
    from liquidbiopsy_agent.multimodal.methylgpt_encoder import MinimalMethylVocab

    token_to_id = {"<pad>": 0}
    for i in range(1, n_cpg_tokens + 1):
        token_to_id[f"cg{i:07d}"] = i
    return MinimalMethylVocab(vocab=token_to_id, pad_token="<pad>", pad_value=-2.0)


def build_demo_inputs(batch_size: int, seq_len: int, vocab: MinimalMethylVocab):
    import torch

    n_tokens = len(vocab.vocab)
    gene_ids = torch.randint(low=1, high=n_tokens, size=(batch_size, seq_len), dtype=torch.long)
    values = torch.rand(batch_size, seq_len, dtype=torch.float32)

    pad_id = vocab.vocab[vocab.pad_token]
    gene_ids[0, -16:] = pad_id
    values[0, -16:] = vocab.pad_value
    gene_ids[1, -8:] = pad_id
    values[1, -8:] = vocab.pad_value
    return gene_ids, values


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    LOGGER.info("Using data root: %s", get_data_root())

    from liquidbiopsy_agent.multimodal.methylgpt_encoder import MethylGPTEncoder

    ckpt_path = args.ckpt_path
    vocab = build_demo_vocab(n_cpg_tokens=2048)
    model_hparams = {
        "d_model": 128,
        "nhead": 4,
        "nlayers": 2,
        "d_hid": 128,
        "dropout": 0.1,
        "do_mvc": True,
        "do_dab": False,
        "use_batch_labels": False,
        "num_batch_labels": None,
        "domain_spec_batchnorm": False,
        "n_input_bins": None,
        "ecs_threshold": 0.3,
        "explicit_zero_prob": False,
        "use_fast_transformer": False,
        "pre_norm": False,
        "cell_emb_style": "cls",
    }

    encoder = MethylGPTEncoder(
        ckpt_path=ckpt_path,
        vocab=vocab,
        model_hparams=model_hparams,
        device="auto",
        precision="auto",
    )

    gene_ids, values = build_demo_inputs(batch_size=2, seq_len=128, vocab=vocab)
    cell_emb = encoder.encode(gene_ids, values, mode="cell_emb")
    token_emb = encoder.encode(gene_ids, values, mode="token_emb")
    LOGGER.info("cell_emb shape: %s", tuple(cell_emb.shape))
    LOGGER.info("token_emb shape: %s", tuple(token_emb.shape))

    sample_list = [
        {"gene_ids": gene_ids[0], "values": values[0]},
        {"gene_ids": gene_ids[1], "values": values[1]},
    ]
    cell_emb_batch = encoder.encode_batch(sample_list, mode="cell_emb")
    LOGGER.info("encode_batch(cell_emb) shape: %s", tuple(cell_emb_batch.shape))


if __name__ == "__main__":
    main()
