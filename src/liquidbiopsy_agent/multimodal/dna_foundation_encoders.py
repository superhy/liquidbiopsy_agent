from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from liquidbiopsy_agent.utils.storage import get_models_root, resolve_data_path

DEFAULT_MODEL_NTV2 = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DEFAULT_MODEL_DNABERT2 = "zhihan1996/DNABERT-2-117M"
DEFAULT_MODEL_HYENADNA = "LongSafari/hyenadna-small-32k-seqlen-hf"
DEFAULT_MODEL_CADUCEUS = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
DEFAULT_MODEL_EPIBERT = "super-dainiu/epiBERT-unibert"
DEFAULT_MODEL_ENFORMER = "EleutherAI/enformer-191k"

MODEL_KEY_TO_DEFAULT_SOURCE = {
    "ntv2": DEFAULT_MODEL_NTV2,
    "dnabert2": DEFAULT_MODEL_DNABERT2,
    "hyenadna": DEFAULT_MODEL_HYENADNA,
    "caduceus": DEFAULT_MODEL_CADUCEUS,
    "epibert": DEFAULT_MODEL_EPIBERT,
    "epcot": "",
    "enformer": DEFAULT_MODEL_ENFORMER,
}

MODEL_KEY_TO_LOCAL_SUBDIR = {
    "ntv2": "ntv2",
    "dnabert2": "dnabert2",
    "hyenadna": "hyenadna",
    "caduceus": "caduceus",
    "epibert": "epibert",
    "epcot": "epcot",
    "enformer": "enformer",
}


@dataclass(frozen=True)
class EncoderInputProfile:
    """Model-specific input policy for GSE243474 BED encoding."""

    default_window_size: int
    default_max_intervals_per_file: int
    default_batch_size: int
    peak_focus_mask: bool = False
    use_reverse_complement_augmentation: bool = False


def _resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def _normalize_dna_sequence(sequence: str) -> str:
    seq = sequence.upper()
    chars = []
    for ch in seq:
        if ch in {"A", "C", "G", "T", "N"}:
            chars.append(ch)
        else:
            chars.append("N")
    return "".join(chars)


def _reverse_complement(sequence: str) -> str:
    table = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})
    return sequence.translate(table)[::-1]


def _apply_peak_focus_mask(sequence: str, interval_span: tuple[int, int] | None, flank: int = 32) -> str:
    if interval_span is None:
        return sequence
    seq_len = len(sequence)
    start, end = interval_span
    keep_start = max(0, start - flank)
    keep_end = min(seq_len, end + flank)
    if keep_start >= keep_end:
        return "N" * seq_len
    return ("N" * keep_start) + sequence[keep_start:keep_end] + ("N" * (seq_len - keep_end))


def _resolve_model_source(
    model_key: str,
    model_name: str | None,
    model_root: str | Path | None,
) -> str:
    if model_name:
        model_candidate = Path(model_name).expanduser()
        if model_candidate.exists():
            return str(resolve_data_path(model_candidate, path_kind=f"{model_key} model path", must_exist=True))
        return model_name

    if model_root is not None:
        root = resolve_data_path(model_root, path_kind="foundation model root", must_exist=False)
    else:
        root = get_models_root(must_exist=False)

    subdir = MODEL_KEY_TO_LOCAL_SUBDIR.get(model_key, model_key)
    local_dir = root / subdir
    if local_dir.exists() and local_dir.is_dir():
        return str(local_dir)

    default_source = MODEL_KEY_TO_DEFAULT_SOURCE.get(model_key, "")
    if default_source:
        return default_source
    raise ValueError(f"No default source for model_key='{model_key}'. Please pass explicit model_name.")


def _infer_hidden_size_from_config(config) -> int | None:
    for attr in ("hidden_size", "d_model", "dim", "n_embd"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


class BaseDNAFoundationEncoder(ABC):
    @property
    @abstractmethod
    def model_key(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_profile(self) -> EncoderInputProfile:
        raise NotImplementedError

    def preprocess_sequence(self, sequence: str, interval_span: tuple[int, int] | None = None) -> str:
        del interval_span
        return _normalize_dna_sequence(sequence)

    def set_runtime_peak_focus_mask(self, enabled: bool) -> None:
        del enabled

    @abstractmethod
    def encode_sequences(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 32,
        interval_spans: Sequence[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class HuggingFaceDNAFoundationEncoder(BaseDNAFoundationEncoder):
    def __init__(
        self,
        model_key: str,
        model_name: str,
        input_profile: EncoderInputProfile,
        *,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        device: str = "auto",
        peak_focus_flank: int = 32,
        use_masked_lm_loader: bool = False,
        force_peak_focus_mask: bool = False,
    ) -> None:
        self._model_key = model_key
        self._model_name = model_name
        self._input_profile = input_profile
        self._peak_focus_flank = peak_focus_flank
        self._force_peak_focus_mask = force_peak_focus_mask
        self._device = _resolve_device(device)
        self._use_masked_lm_loader = use_masked_lm_loader

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        if use_masked_lm_loader:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )

        self.model.to(self._device)
        self.model.eval()
        emb = _infer_hidden_size_from_config(self.model.config)
        if emb is None:
            emb = self._infer_embedding_dim_by_forward()
        self._embedding_dim = int(emb)

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def input_profile(self) -> EncoderInputProfile:
        return self._input_profile

    def preprocess_sequence(self, sequence: str, interval_span: tuple[int, int] | None = None) -> str:
        seq = _normalize_dna_sequence(sequence)
        if self.input_profile.peak_focus_mask or self._force_peak_focus_mask:
            seq = _apply_peak_focus_mask(seq, interval_span, flank=self._peak_focus_flank)
        return seq

    def set_runtime_peak_focus_mask(self, enabled: bool) -> None:
        self._force_peak_focus_mask = bool(enabled)

    def _infer_embedding_dim_by_forward(self) -> int:
        test = self.encode_sequences(["ACGTACGT"], batch_size=1)
        if test.ndim != 2 or test.shape[1] <= 0:
            raise RuntimeError(f"Unable to infer embedding dim for model '{self.model_name}'.")
        return int(test.shape[1])

    def _get_last_hidden_state(self, outputs, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if isinstance(first, torch.Tensor) and first.ndim == 3:
                return first

        last = getattr(outputs, "last_hidden_state", None)
        if last is not None:
            return last

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]

        # For masked LM heads that may not expose last_hidden_state directly.
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None:
            base_out = base_model(**encoded, return_dict=True)
            base_last = getattr(base_out, "last_hidden_state", None)
            if base_last is not None:
                return base_last

        raise RuntimeError(
            f"Model '{self.model_name}' did not return hidden states usable for embedding extraction."
        )

    def _encode_preprocessed_sequences(self, sequences: Sequence[str], batch_size: int) -> np.ndarray:
        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        rows: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = list(sequences[i : i + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                if "attention_mask" not in encoded and "input_ids" in encoded:
                    encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                accepted_args = set(signature(self.model.forward).parameters.keys())
                model_inputs = {k: v for k, v in encoded.items() if k in accepted_args}
                try:
                    outputs = self.model(**model_inputs, return_dict=True, output_hidden_states=True)
                except TypeError:
                    outputs = self.model(**model_inputs)
                last_hidden = self._get_last_hidden_state(outputs, model_inputs)
                pooled = _mean_pool_last_hidden(last_hidden, encoded["attention_mask"])
                rows.append(pooled.detach().cpu().numpy())
        return np.vstack(rows).astype(np.float32)

    def encode_sequences(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 32,
        interval_spans: Sequence[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if interval_spans is not None and len(interval_spans) != len(sequences):
            raise ValueError("interval_spans length must match sequences length.")

        prepared: list[str] = []
        if interval_spans is None:
            for sequence in sequences:
                prepared.append(self.preprocess_sequence(sequence, interval_span=None))
        else:
            for sequence, span in zip(sequences, interval_spans):
                prepared.append(self.preprocess_sequence(sequence, interval_span=span))

        emb = self._encode_preprocessed_sequences(prepared, batch_size=batch_size)
        if not self.input_profile.use_reverse_complement_augmentation:
            return emb

        rc_sequences = [_reverse_complement(seq) for seq in prepared]
        rc_emb = self._encode_preprocessed_sequences(rc_sequences, batch_size=batch_size)
        return ((emb + rc_emb) * 0.5).astype(np.float32)


class EnformerPytorchEncoder(BaseDNAFoundationEncoder):
    def __init__(
        self,
        model_name: str,
        input_profile: EncoderInputProfile,
        *,
        local_files_only: bool = True,
        device: str = "auto",
        force_peak_focus_mask: bool = False,
        peak_focus_flank: int = 32,
    ) -> None:
        from enformer_pytorch import Enformer

        self._model_key = "enformer"
        self._model_name = model_name
        self._input_profile = input_profile
        self._device = _resolve_device(device)
        self._force_peak_focus_mask = force_peak_focus_mask
        self._peak_focus_flank = peak_focus_flank

        # The downloaded checkpoint from EleutherAI/enformer-191k may have minor
        # architecture drift with current enformer-pytorch versions.
        self.model = Enformer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self._device)
        self.model.eval()

        cfg_dim = getattr(self.model.config, "dim", None)
        self._embedding_dim = int(cfg_dim * 2) if isinstance(cfg_dim, int) and cfg_dim > 0 else 3072

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def input_profile(self) -> EncoderInputProfile:
        return self._input_profile

    def _to_one_hot(self, batch_sequences: Sequence[str]) -> torch.Tensor:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.zeros((len(batch_sequences), len(batch_sequences[0]), 4), dtype=np.float32)
        for i, seq in enumerate(batch_sequences):
            for j, ch in enumerate(seq):
                idx = mapping.get(ch, None)
                if idx is not None:
                    arr[i, j, idx] = 1.0
        return torch.from_numpy(arr)

    def preprocess_sequence(self, sequence: str, interval_span: tuple[int, int] | None = None) -> str:
        seq = _normalize_dna_sequence(sequence)
        if self.input_profile.peak_focus_mask or self._force_peak_focus_mask:
            seq = _apply_peak_focus_mask(seq, interval_span, flank=self._peak_focus_flank)
        target_len = self.input_profile.default_window_size
        if len(seq) < target_len:
            seq = seq + ("N" * (target_len - len(seq)))
        elif len(seq) > target_len:
            seq = seq[:target_len]
        return seq

    def set_runtime_peak_focus_mask(self, enabled: bool) -> None:
        self._force_peak_focus_mask = bool(enabled)

    def encode_sequences(
        self,
        sequences: Sequence[str],
        *,
        batch_size: int = 1,
        interval_spans: Sequence[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if not sequences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if interval_spans is not None and len(interval_spans) != len(sequences):
            raise ValueError("interval_spans length must match sequences length.")

        if interval_spans is None:
            prepared = [self.preprocess_sequence(s, interval_span=None) for s in sequences]
        else:
            prepared = [self.preprocess_sequence(s, interval_span=span) for s, span in zip(sequences, interval_spans)]
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(prepared), batch_size):
                batch = prepared[i : i + batch_size]
                x = self._to_one_hot(batch).to(self.device)
                out = self.model(x, return_only_embeddings=True)
                if out.ndim != 3:
                    raise RuntimeError("Enformer output must be [batch, tokens, dim].")
                pooled = out.mean(dim=1)
                rows.append(pooled.detach().cpu().numpy())
        return np.vstack(rows).astype(np.float32)


class NucleotideTransformerV2Encoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL_NTV2, **kwargs) -> None:
        super().__init__(
            model_key="ntv2",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=256,
                default_max_intervals_per_file=1024,
                default_batch_size=32,
            ),
            use_masked_lm_loader=True,
            **kwargs,
        )


class DNABERT2Encoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL_DNABERT2, **kwargs) -> None:
        super().__init__(
            model_key="dnabert2",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=512,
                default_max_intervals_per_file=1024,
                default_batch_size=32,
            ),
            **kwargs,
        )


class HyenaDNAEncoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL_HYENADNA, **kwargs) -> None:
        super().__init__(
            model_key="hyenadna",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=4096,
                default_max_intervals_per_file=512,
                default_batch_size=8,
            ),
            **kwargs,
        )


class CaduceusEncoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str = DEFAULT_MODEL_CADUCEUS, **kwargs) -> None:
        super().__init__(
            model_key="caduceus",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=4096,
                default_max_intervals_per_file=512,
                default_batch_size=8,
            ),
            **kwargs,
        )


class EpiBERTEncoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(
            model_key="epibert",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=2048,
                default_max_intervals_per_file=512,
                default_batch_size=8,
                peak_focus_mask=True,
                use_reverse_complement_augmentation=True,
            ),
            **kwargs,
        )


class EPCOTEncoder(HuggingFaceDNAFoundationEncoder):
    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(
            model_key="epcot",
            model_name=model_name,
            input_profile=EncoderInputProfile(
                default_window_size=2048,
                default_max_intervals_per_file=512,
                default_batch_size=8,
                peak_focus_mask=True,
                use_reverse_complement_augmentation=True,
            ),
            **kwargs,
        )


SUPPORTED_MODEL_KEYS = (
    "ntv2",
    "dnabert2",
    "hyenadna",
    "caduceus",
    "epibert",
    "epcot",
    "enformer",
)


def list_supported_model_keys() -> tuple[str, ...]:
    return SUPPORTED_MODEL_KEYS


def build_dna_foundation_encoder(
    *,
    model_key: str = "ntv2",
    model_name: str | None = None,
    model_root: str | Path | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    device: str = "auto",
    force_peak_focus_mask: bool = False,
) -> BaseDNAFoundationEncoder:
    kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "device": device,
        "force_peak_focus_mask": force_peak_focus_mask,
    }
    model_key_norm = model_key.lower()
    if model_key_norm not in SUPPORTED_MODEL_KEYS:
        raise ValueError(f"Unsupported model_key='{model_key}'. Supported: {SUPPORTED_MODEL_KEYS}")

    if model_key_norm == "ntv2":
        source = _resolve_model_source("ntv2", model_name, model_root)
        last_err: Exception | None = None
        for cls in (NucleotideTransformerV2Encoder, DNABERT2Encoder):
            try:
                if cls is NucleotideTransformerV2Encoder:
                    return cls(model_name=source, **kwargs)
                fallback_source = _resolve_model_source("dnabert2", None, model_root)
                return cls(model_name=fallback_source, **kwargs)
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Failed to load ntv2 and dnabert2 fallback. Last error: {last_err}")

    if model_key_norm == "dnabert2":
        source = _resolve_model_source("dnabert2", model_name, model_root)
        return DNABERT2Encoder(model_name=source, **kwargs)

    if model_key_norm == "hyenadna":
        source = _resolve_model_source("hyenadna", model_name, model_root)
        return HyenaDNAEncoder(model_name=source, **kwargs)

    if model_key_norm == "caduceus":
        source = _resolve_model_source("caduceus", model_name, model_root)
        try:
            return CaduceusEncoder(model_name=source, **kwargs)
        except Exception as exc:
            # Caduceus requires mamba_ssm/triton; on Windows this commonly fails.
            hyena_source = _resolve_model_source("hyenadna", None, model_root)
            warnings.warn(
                "Caduceus loading failed, fallback to HyenaDNA. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            return HyenaDNAEncoder(model_name=hyena_source, **kwargs)

    if model_key_norm == "epibert":
        source = _resolve_model_source("epibert", model_name, model_root)
        return EpiBERTEncoder(model_name=source, **kwargs)

    if model_key_norm == "epcot":
        source = _resolve_model_source("epcot", model_name, model_root)
        if Path(source).is_dir() and not (Path(source) / "config.json").exists():
            epibert_source = _resolve_model_source("epibert", None, model_root)
            warnings.warn(
                "EPCOT checkpoint is missing. Fallback to EpiBERT. "
                f"Checked path: {source}"
            )
            return EpiBERTEncoder(model_name=epibert_source, **kwargs)
        try:
            return EPCOTEncoder(model_name=source, **kwargs)
        except Exception as exc:
            epibert_source = _resolve_model_source("epibert", None, model_root)
            warnings.warn(
                "EPCOT loading failed, fallback to EpiBERT. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            return EpiBERTEncoder(model_name=epibert_source, **kwargs)

    source = _resolve_model_source("enformer", model_name, model_root)
    return EnformerPytorchEncoder(
        model_name=source,
        input_profile=EncoderInputProfile(
            default_window_size=196_608,
            default_max_intervals_per_file=64,
            default_batch_size=1,
            use_reverse_complement_augmentation=True,
        ),
        local_files_only=local_files_only,
        device=device,
        force_peak_focus_mask=force_peak_focus_mask,
    )
