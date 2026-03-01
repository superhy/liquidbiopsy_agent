from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch
from scgpt.model.model import TransformerModel
from liquidbiopsy_agent.utils.storage import get_data_root, resolve_data_path

EncodeMode = Literal["cell_emb", "token_emb"]

LOGGER = logging.getLogger(__name__)


class MinimalMethylVocab:
    """Minimal vocab compatible with MethylGPT/scGPT inference calls."""

    def __init__(self, vocab: Dict[str, int], pad_token: str = "<pad>", pad_value: float = -2.0):
        if pad_token not in vocab:
            raise ValueError(f"pad_token '{pad_token}' is missing in vocab mapping")
        self.vocab = dict(vocab)
        self.pad_token = pad_token
        self.pad_value = float(pad_value)


class MethylGPTEncoder:
    """Inference-only encoder for cfDNA methylome panel signals."""

    def __init__(
        self,
        ckpt_path: str | Path,
        vocab: Any,
        model_hparams: Mapping[str, Any],
        device: str = "auto",
        precision: str = "auto",
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ckpt_path = resolve_data_path(
            ckpt_path,
            path_kind="MethylGPT checkpoint",
            must_exist=False,
        )
        self.vocab = vocab
        self.model_hparams = dict(model_hparams)
        self.device = self._resolve_device(device)
        self.precision = precision

        self._validate_vocab(self.vocab)
        self.pad_id = int(self.vocab.vocab[self.vocab.pad_token])
        self.pad_value = float(self.vocab.pad_value)
        self._autocast_dtype = self._resolve_autocast_dtype(self.precision, self.device)

        self.model = self._build_model(self.model_hparams)
        self._load_checkpoint(self.ckpt_path)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info(
            "MethylGPTEncoder ready on device=%s precision=%s data_root=%s",
            self.device,
            self.precision,
            get_data_root(),
        )

    def _validate_vocab(self, vocab: Any) -> None:
        for attr in ("vocab", "pad_token", "pad_value"):
            if not hasattr(vocab, attr):
                raise TypeError(f"vocab must expose '{attr}'")
        if not isinstance(vocab.vocab, dict):
            raise TypeError("vocab.vocab must be a dict(token -> id)")
        if vocab.pad_token not in vocab.vocab:
            raise ValueError(f"pad_token '{vocab.pad_token}' missing in vocab.vocab")

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_autocast_dtype(self, precision: str, device: torch.device):
        precision = precision.lower()
        if precision not in {"auto", "fp16", "bf16", "fp32"}:
            raise ValueError("precision must be one of: auto, fp16, bf16, fp32")

        if device.type != "cuda":
            return None

        if precision == "fp32":
            return None
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            self.logger.warning("bf16 requested but unsupported on this GPU. Falling back to fp16.")
            return torch.float16

        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_model(self, hparams: Mapping[str, Any]) -> TransformerModel:
        required = ("d_model", "nhead", "nlayers", "dropout")
        missing = [k for k in required if k not in hparams]
        if missing:
            raise ValueError(f"Missing required model_hparams: {missing}")

        d_model = int(hparams["d_model"])
        nhead = int(hparams["nhead"])
        nlayers = int(hparams["nlayers"])
        dropout = float(hparams["dropout"])
        d_hid = int(hparams.get("d_hid", d_model))
        ntoken = len(self.vocab.vocab)

        model = TransformerModel(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=int(hparams.get("nlayers_cls", 3)),
            n_cls=int(hparams.get("n_cls", 1)),
            vocab=self.vocab.vocab,
            dropout=dropout,
            pad_token=self.vocab.pad_token,
            pad_value=self.vocab.pad_value,
            do_mvc=bool(hparams.get("do_mvc", True)),
            do_dab=bool(hparams.get("do_dab", False)),
            use_batch_labels=bool(hparams.get("use_batch_labels", False)),
            num_batch_labels=hparams.get("num_batch_labels", None),
            domain_spec_batchnorm=bool(hparams.get("domain_spec_batchnorm", False)),
            input_emb_style=str(hparams.get("input_emb_style", "continuous")),
            n_input_bins=hparams.get("n_input_bins", None),
            cell_emb_style=str(hparams.get("cell_emb_style", "cls")),
            mvc_decoder_style=str(hparams.get("mvc_decoder_style", "inner product")),
            ecs_threshold=hparams.get("ecs_threshold", 0.3),
            explicit_zero_prob=bool(hparams.get("explicit_zero_prob", False)),
            use_fast_transformer=bool(hparams.get("use_fast_transformer", False)),
            fast_transformer_backend=str(hparams.get("fast_transformer_backend", "flash")),
            pre_norm=bool(hparams.get("pre_norm", False)),
        )
        return model

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            self.logger.warning("Checkpoint not found: %s. Skip loading weights.", ckpt_path)
            return

        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        incompatible = self.model.load_state_dict(state, strict=False)
        self.logger.info(
            "Checkpoint loaded from %s (missing=%d unexpected=%d)",
            ckpt_path,
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )

    def _as_tensor(self, x: Any, *, dtype: torch.dtype, name: str) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype)
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=dtype)
        raise TypeError(f"{name} must be torch.Tensor, numpy.ndarray, list, or tuple")

    def _prepare_inputs(self, gene_ids: Any, values: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_ids_t = self._as_tensor(gene_ids, dtype=torch.long, name="gene_ids")
        values_t = self._as_tensor(values, dtype=torch.float32, name="values")

        if gene_ids_t.ndim == 1:
            gene_ids_t = gene_ids_t.unsqueeze(0)
        elif gene_ids_t.ndim != 2:
            raise ValueError(f"gene_ids must have shape (L,) or (B,L), got {tuple(gene_ids_t.shape)}")

        if values_t.ndim == 1:
            values_t = values_t.unsqueeze(0)
        elif values_t.ndim != 2:
            raise ValueError(f"values must have shape (L,) or (B,L), got {tuple(values_t.shape)}")

        if gene_ids_t.shape != values_t.shape:
            raise ValueError(
                f"gene_ids and values shape mismatch: {tuple(gene_ids_t.shape)} vs {tuple(values_t.shape)}"
            )

        non_padding = values_t.ne(self.pad_value)
        if non_padding.any():
            valid_values = values_t[non_padding]
            min_v = float(valid_values.min().item())
            max_v = float(valid_values.max().item())
            if min_v < 0.0 or max_v > 1.0:
                self.logger.warning(
                    "values out of expected [0,1] range (excluding pad_value=%s): min=%.4f max=%.4f",
                    self.pad_value,
                    min_v,
                    max_v,
                )
        return gene_ids_t, values_t

    def _autocast_ctx(self):
        if self.device.type == "cuda" and self._autocast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def encode(self, gene_ids: Any, values: Any, mode: EncodeMode = "cell_emb") -> torch.Tensor:
        if mode not in {"cell_emb", "token_emb"}:
            raise ValueError("mode must be 'cell_emb' or 'token_emb'")

        gene_ids_t, values_t = self._prepare_inputs(gene_ids, values)
        gene_ids_t = gene_ids_t.to(self.device, non_blocking=True)
        values_t = values_t.to(self.device, non_blocking=True)

        # Must follow scGPT padding mask construction.
        src_key_padding_mask = gene_ids_t.eq(self.vocab.vocab[self.vocab.pad_token])

        self.model.eval()
        with torch.no_grad():
            with self._autocast_ctx():
                if mode == "token_emb":
                    # Must call _encode for token-level embeddings.
                    h = self.model._encode(
                        gene_ids_t,
                        values_t,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=None,
                    )
                    return h.detach().cpu()

                # Must call model(...) and read out["cell_emb"] for sample embeddings.
                out = self.model(
                    gene_ids_t,
                    values_t,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                emb = out["cell_emb"]
                return emb.detach().cpu()

    def encode_batch(self, list_of_samples: Sequence[Any], mode: EncodeMode = "cell_emb") -> torch.Tensor:
        if not list_of_samples:
            raise ValueError("list_of_samples is empty")

        parsed: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx, sample in enumerate(list_of_samples):
            if isinstance(sample, Mapping):
                if "gene_ids" not in sample or "values" not in sample:
                    raise ValueError("Each mapping sample must contain 'gene_ids' and 'values'")
                gene_ids = sample["gene_ids"]
                values = sample["values"]
            elif isinstance(sample, (tuple, list)) and len(sample) == 2:
                gene_ids, values = sample
            else:
                raise TypeError(
                    f"Sample {idx} must be dict with keys gene_ids/values or tuple(gene_ids, values)"
                )

            gi, va = self._prepare_inputs(gene_ids, values)
            if gi.shape[0] != 1:
                raise ValueError(
                    f"encode_batch expects each sample to be single sequence; got batch shape {tuple(gi.shape)}"
                )
            parsed.append((gi.squeeze(0), va.squeeze(0)))

        max_len = max(int(g.shape[0]) for g, _ in parsed)
        batch_size = len(parsed)

        gene_ids_batch = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        values_batch = torch.full((batch_size, max_len), float(self.pad_value), dtype=torch.float32)
        for i, (g, v) in enumerate(parsed):
            length = int(g.shape[0])
            gene_ids_batch[i, :length] = g
            values_batch[i, :length] = v

        return self.encode(gene_ids_batch, values_batch, mode=mode)
