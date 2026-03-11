from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from liquidbiopsy_agent.pipeline.nodes.cnv import bin_counts
from liquidbiopsy_agent.utils.storage import get_data_root, resolve_data_path

from .bed_embedding import _strip_bed_suffix, encode_bed_folder_to_embeddings
from .dna_foundation_encoders import SUPPORTED_MODEL_KEYS as SUPPORTED_DNA_FOUNDATION_MODEL_KEYS

_INTERVAL_SIGNALS = {"cfchip_seq", "cfmedip_seq", "medip_seq"}
_LPWGS_SIGNALS = {"lpwgs", "ulpwgs"}
_VARIANT_SIGNALS = {"ctdna_variant", "variant"}
_FOUNDATION_SEQUENCE_SIGNALS = _INTERVAL_SIGNALS | _LPWGS_SIGNALS
_SUPPORTED_SIGNALS = tuple(sorted(_INTERVAL_SIGNALS | _LPWGS_SIGNALS | _VARIANT_SIGNALS))
_FOUNDATION_ENCODERS = {str(k).lower() for k in SUPPORTED_DNA_FOUNDATION_MODEL_KEYS}


@dataclass(frozen=True)
class BloodSignalEncoderSpec:
    signal: str
    input_formats: tuple[str, ...]
    default_encoder: str
    notes: str


class BaseBloodSignalEncoder(ABC):
    """Base class for blood-signal encoders."""

    @property
    @abstractmethod
    def encoder_key(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_signals(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_input_formats(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError


class IntervalFoundationEncoder(BaseBloodSignalEncoder):
    """
    ONLY for interval-region signals represented as BED-like genomic regions.

    Applicable:
      - cfChIP-seq region/peak BED(.gz)
      - cfMeDIP-seq / MeDIP-seq region/peak BED(.gz)
      - LPWGS/ULPWGS BED(.gz) fragment intervals (sequence-window embedding view; exploratory use)

    Not applicable:
      - VCF variant lists

    Notes:
      - For LPWGS, this captures sequence context around fragments and is exploratory.
      - For LPWGS production copy-number semantics, use `lpwgs_cnv_profile`.
    """

    @property
    def encoder_key(self) -> str:
        return "dna_foundation"

    @property
    def supported_signals(self) -> tuple[str, ...]:
        return tuple(sorted(_FOUNDATION_SEQUENCE_SIGNALS))

    @property
    def supported_input_formats(self) -> tuple[str, ...]:
        return ("bed", "bed.gz", "narrowpeak", "broadpeak", "gappedpeak")

    def encode(
        self,
        *,
        signal: str,
        input_dir: str | Path,
        fasta_path: str | Path,
        output_dir: str | Path | None = None,
        model_key: str = "ntv2",
        model_name: str | None = None,
        model_root: str | Path | None = "models",
        peak_mode: str = "mask_from_raw",
        window_size: int | None = None,
        max_intervals_per_file: int | None = None,
        batch_size: int | None = None,
        seed: int = 42,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        device: str = "auto",
        verbose: bool = True,
    ) -> dict[str, Any]:
        signal_norm = _normalize_signal(signal)
        if signal_norm not in _FOUNDATION_SEQUENCE_SIGNALS:
            raise ValueError(
                f"Signal '{signal}' is not supported by interval encoder. "
                f"Supported: {sorted(_FOUNDATION_SEQUENCE_SIGNALS)}"
            )

        out_dir = (
            _default_output_dir(
                signal=signal_norm,
                encoder=f"dnafm-{model_key}",
                mode=peak_mode,
            )
            if output_dir is None
            else resolve_data_path(output_dir, path_kind="blood signal output dir", must_exist=False)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = encode_bed_folder_to_embeddings(
            input_dir=resolve_data_path(input_dir, path_kind="interval input dir", must_exist=True),
            fasta_path=resolve_data_path(fasta_path, path_kind="reference fasta", must_exist=True),
            output_dir=out_dir,
            model_key=model_key,
            model_name=model_name,
            model_root=model_root,
            window_size=window_size,
            max_intervals_per_file=max_intervals_per_file,
            batch_size=batch_size,
            seed=seed,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            device=device,
            verbose=verbose,
            peak_mode=peak_mode,
        )

        safe_signal = _safe_token(signal_norm)
        safe_encoder = _safe_token(model_key)
        safe_mode = _safe_token(peak_mode)
        out_pack = out_dir / f"blood_features__signal-{safe_signal}__encoder-{safe_encoder}__mode-{safe_mode}.pt"
        if summary.get("cfdna_features_pt"):
            src_pack = Path(summary["cfdna_features_pt"])
            payload = torch.load(src_pack, map_location="cpu")
            torch.save(payload, out_pack)

        summary["blood_feature_pack_pt"] = str(out_pack) if out_pack.exists() else None
        summary["signal"] = signal_norm
        summary["encoder"] = model_key
        return summary


class LpWgsCnvProfileEncoder(BaseBloodSignalEncoder):
    """
    ONLY for LPWGS/ULPWGS copy-number profile encoding.

    Applicable:
      - LPWGS BED(.gz) fragment coordinates (will be binned)
      - Precomputed CNV bin tables: *_bin_counts.parquet

    Not applicable:
      - Peak-centric enrichment signals (cfChIP/cfMeDIP/MeDIP)
      - VCF variant records

    Notes:
      - This is a deterministic profile encoder (not a pretrained foundation model),
        because a generally adopted pretrained LPWGS foundation encoder is not yet
        standard in production workflows.
    """

    @property
    def encoder_key(self) -> str:
        return "lpwgs_cnv_profile"

    @property
    def supported_signals(self) -> tuple[str, ...]:
        return tuple(sorted(_LPWGS_SIGNALS))

    @property
    def supported_input_formats(self) -> tuple[str, ...]:
        return ("bed", "bed.gz", "cnv_parquet")

    def encode(
        self,
        *,
        signal: str,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        input_format: str = "bed.gz",
        bin_size: int = 1_000_000,
        target_bins: int = 256,
        verbose: bool = True,
    ) -> dict[str, Any]:
        signal_norm = _normalize_signal(signal)
        if signal_norm not in _LPWGS_SIGNALS:
            raise ValueError(
                f"Signal '{signal}' is not supported by LPWGS encoder. "
                f"Supported: {sorted(_LPWGS_SIGNALS)}"
            )
        if target_bins <= 8:
            raise ValueError("target_bins must be > 8")
        if bin_size <= 0:
            raise ValueError("bin_size must be > 0")

        src_dir = resolve_data_path(input_dir, path_kind="LPWGS input dir", must_exist=True)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"LPWGS input dir is not a folder: {src_dir}")
        out_dir = (
            _default_output_dir(
                signal=signal_norm,
                encoder=self.encoder_key,
                mode=f"bin{int(bin_size)}_tb{int(target_bins)}",
            )
            if output_dir is None
            else resolve_data_path(output_dir, path_kind="LPWGS output dir", must_exist=False)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        in_fmt = str(input_format).strip().lower()
        if in_fmt == "cnv_parquet":
            files = sorted(p for p in src_dir.glob("*_bin_counts.parquet") if p.is_file())
        else:
            files = sorted(
                p
                for p in src_dir.iterdir()
                if p.is_file() and not p.name.startswith("._") and re.search(r"\.bed(\.gz)?$", p.name, flags=re.I)
            )

        if not files:
            raise ValueError(f"No valid input files found in: {src_dir}")

        feature_map: dict[str, torch.Tensor] = {}
        rows: list[dict[str, Any]] = []
        for path in files:
            sid = _sample_id_from_path(path)
            if in_fmt == "cnv_parquet":
                df = pd.read_parquet(path)
            else:
                df = bin_counts(path, bin_size=bin_size)
            vec = _encode_cnv_dataframe(df, target_bins=target_bins)
            feature_map[sid] = torch.from_numpy(vec.copy())
            rows.append(
                {
                    "sample_id": sid,
                    "source_file": path.name,
                    "input_format": in_fmt,
                    "bin_size": int(bin_size),
                    "target_bins": int(target_bins),
                    "feature_dim": int(vec.shape[0]),
                    "n_cnv_rows": int(df.shape[0]),
                }
            )
            if verbose:
                print(f"[LPWGS] {sid}: feature_dim={vec.shape[0]} n_cnv_rows={df.shape[0]}")

        safe_signal = _safe_token(signal_norm)
        safe_mode = _safe_token(f"bin{int(bin_size)}_tb{int(target_bins)}")
        pack_pt = out_dir / f"blood_features__signal-{safe_signal}__encoder-{self.encoder_key}__mode-{safe_mode}.pt"
        meta_csv = out_dir / "metadata.csv"
        sample_index_json = out_dir / "sample_id_to_row.json"

        torch.save(feature_map, pack_pt)
        meta_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
        meta_df.to_csv(meta_csv, index=False)
        sample_index = {str(sid): int(i) for i, sid in enumerate(meta_df["sample_id"].tolist())}
        sample_index_json.write_text(json.dumps(sample_index, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "signal": signal_norm,
            "encoder": self.encoder_key,
            "input_dir": str(src_dir),
            "output_dir": str(out_dir),
            "files_processed": len(files),
            "feature_dim": int(next(iter(feature_map.values())).shape[0]),
            "blood_feature_pack_pt": str(pack_pt),
            "metadata_csv": str(meta_csv),
            "sample_id_to_row_json": str(sample_index_json),
        }


class VariantSignatureEncoder(BaseBloodSignalEncoder):
    """
    ONLY for ctDNA variant-call tables (VCF-like records).

    Applicable:
      - VCF text files (*.vcf, *.vcf.gz)

    Not applicable:
      - BED peak signals
      - LPWGS copy-number bin signals

    Notes:
      - This is a compact baseline signature encoder from VCF fields.
      - TODO(superhy): integrate DeepSEA/SpliceAI/CADD-style effect encoders when
        dedicated model assets are prepared in local models/ and benchmark labels exist.
    """

    @property
    def encoder_key(self) -> str:
        return "vcf_signature"

    @property
    def supported_signals(self) -> tuple[str, ...]:
        return tuple(sorted(_VARIANT_SIGNALS))

    @property
    def supported_input_formats(self) -> tuple[str, ...]:
        return ("vcf", "vcf.gz")

    def encode(
        self,
        *,
        signal: str,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        signal_norm = _normalize_signal(signal)
        if signal_norm not in _VARIANT_SIGNALS:
            raise ValueError(
                f"Signal '{signal}' is not supported by variant encoder. "
                f"Supported: {sorted(_VARIANT_SIGNALS)}"
            )

        src_dir = resolve_data_path(input_dir, path_kind="variant input dir", must_exist=True)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Variant input dir is not a folder: {src_dir}")
        out_dir = (
            _default_output_dir(signal=signal_norm, encoder=self.encoder_key, mode="baseline")
            if output_dir is None
            else resolve_data_path(output_dir, path_kind="variant output dir", must_exist=False)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            p
            for p in src_dir.iterdir()
            if p.is_file() and not p.name.startswith("._") and re.search(r"\.vcf(\.gz)?$", p.name, flags=re.I)
        )
        if not files:
            raise ValueError(f"No *.vcf or *.vcf.gz files found in: {src_dir}")

        feature_map: dict[str, torch.Tensor] = {}
        rows: list[dict[str, Any]] = []
        for path in files:
            sid = _sample_id_from_path(path)
            vec, stat = _encode_vcf_file(path)
            feature_map[sid] = torch.from_numpy(vec.copy())
            rows.append({"sample_id": sid, "source_file": path.name, **stat})
            if verbose:
                print(f"[VCF] {sid}: n_var={stat['n_variants']} feature_dim={vec.shape[0]}")

        safe_signal = _safe_token(signal_norm)
        pack_pt = out_dir / f"blood_features__signal-{safe_signal}__encoder-{self.encoder_key}__mode-baseline.pt"
        meta_csv = out_dir / "metadata.csv"
        sample_index_json = out_dir / "sample_id_to_row.json"
        torch.save(feature_map, pack_pt)
        meta_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
        meta_df.to_csv(meta_csv, index=False)
        sample_index = {str(sid): int(i) for i, sid in enumerate(meta_df["sample_id"].tolist())}
        sample_index_json.write_text(json.dumps(sample_index, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "signal": signal_norm,
            "encoder": self.encoder_key,
            "input_dir": str(src_dir),
            "output_dir": str(out_dir),
            "files_processed": len(files),
            "feature_dim": int(next(iter(feature_map.values())).shape[0]),
            "blood_feature_pack_pt": str(pack_pt),
            "metadata_csv": str(meta_csv),
            "sample_id_to_row_json": str(sample_index_json),
        }


def list_supported_blood_signal_specs() -> dict[str, BloodSignalEncoderSpec]:
    return {
        "cfchip_seq": BloodSignalEncoderSpec(
            signal="cfchip_seq",
            input_formats=("bed", "bed.gz", "narrowpeak", "broadpeak", "gappedpeak"),
            default_encoder="ntv2",
            notes="Peak/interval region sequence encoding via DNA foundation models.",
        ),
        "cfmedip_seq": BloodSignalEncoderSpec(
            signal="cfmedip_seq",
            input_formats=("bed", "bed.gz", "narrowpeak", "broadpeak", "gappedpeak"),
            default_encoder="epibert",
            notes="Methylation-enrichment regions; sequence interval embedding.",
        ),
        "medip_seq": BloodSignalEncoderSpec(
            signal="medip_seq",
            input_formats=("bed", "bed.gz", "narrowpeak", "broadpeak", "gappedpeak"),
            default_encoder="epibert",
            notes="MeDIP interval regions; sequence interval embedding.",
        ),
        "lpwgs": BloodSignalEncoderSpec(
            signal="lpwgs",
            input_formats=("bed", "bed.gz", "cnv_parquet"),
            default_encoder="lpwgs_cnv_profile",
            notes=(
                "Default uses LPWGS CNV profile encoder for correctness. "
                "Foundation sequence encoders are optional exploratory alternatives."
            ),
        ),
        "ulpwgs": BloodSignalEncoderSpec(
            signal="ulpwgs",
            input_formats=("bed", "bed.gz", "cnv_parquet"),
            default_encoder="lpwgs_cnv_profile",
            notes=(
                "Default uses LPWGS CNV profile encoder for correctness. "
                "Foundation sequence encoders are optional exploratory alternatives."
            ),
        ),
        "ctdna_variant": BloodSignalEncoderSpec(
            signal="ctdna_variant",
            input_formats=("vcf", "vcf.gz"),
            default_encoder="vcf_signature",
            notes="Compact variant signature vector from VCF records.",
        ),
        "variant": BloodSignalEncoderSpec(
            signal="variant",
            input_formats=("vcf", "vcf.gz"),
            default_encoder="vcf_signature",
            notes="Alias for ctDNA variant signal encoding.",
        ),
    }


def encode_blood_signal_dataset(
    *,
    signal: str,
    input_dir: str | Path,
    input_format: str,
    output_dir: str | Path | None = None,
    encoder: str | None = None,
    fasta_path: str | Path | None = None,
    model_name: str | None = None,
    model_root: str | Path | None = "models",
    peak_mode: str = "mask_from_raw",
    window_size: int | None = None,
    max_intervals_per_file: int | None = None,
    batch_size: int | None = None,
    bin_size: int = 1_000_000,
    target_bins: int = 256,
    seed: int = 42,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    device: str = "auto",
    verbose: bool = True,
) -> dict[str, Any]:
    # TODO(superhy): once the data disk is mounted, run end-to-end checks on:
    #   - GSE243474 cfChIP/cfMeDIP/LPWGS cohorts
    #   - local model loading from /Volumes/US202/liquid-agent-data/models
    #   - output package consistency for cross-modal training ingestion.
    signal_norm = _normalize_signal(signal)
    in_fmt = str(input_format).strip().lower()
    spec_map = list_supported_blood_signal_specs()
    if signal_norm not in spec_map:
        raise ValueError(f"Unsupported signal '{signal}'. Supported: {sorted(spec_map.keys())}")
    spec = spec_map[signal_norm]
    if in_fmt not in spec.input_formats:
        raise ValueError(
            f"Input format '{input_format}' is not supported for signal '{signal_norm}'. "
            f"Supported: {list(spec.input_formats)}"
        )

    selected_encoder = (encoder or spec.default_encoder).strip().lower()
    if signal_norm in _INTERVAL_SIGNALS:
        if fasta_path is None:
            raise ValueError("fasta_path is required for interval-sequence encoding.")
        interval_encoder = IntervalFoundationEncoder()
        return interval_encoder.encode(
            signal=signal_norm,
            input_dir=input_dir,
            fasta_path=fasta_path,
            output_dir=output_dir,
            model_key=selected_encoder,
            model_name=model_name,
            model_root=model_root,
            peak_mode=peak_mode,
            window_size=window_size,
            max_intervals_per_file=max_intervals_per_file,
            batch_size=batch_size,
            seed=seed,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            device=device,
            verbose=verbose,
        )

    if signal_norm in _LPWGS_SIGNALS:
        # Correctness-first policy:
        # - Default/recommended: lpwgs_cnv_profile (CNV semantics preserved).
        # - Optional exploratory branch: DNA foundation encoders on LPWGS BED intervals.
        if selected_encoder == "lpwgs_cnv_profile":
            lpwgs_encoder = LpWgsCnvProfileEncoder()
            return lpwgs_encoder.encode(
                signal=signal_norm,
                input_dir=input_dir,
                output_dir=output_dir,
                input_format=in_fmt,
                bin_size=bin_size,
                target_bins=target_bins,
                verbose=verbose,
            )

        if selected_encoder in _FOUNDATION_ENCODERS:
            if in_fmt not in {"bed", "bed.gz"}:
                raise ValueError(
                    f"LPWGS foundation encoding expects BED-like inputs, got '{in_fmt}'. "
                    "Use --input_format bed.gz (or bed), or switch encoder to lpwgs_cnv_profile for cnv_parquet."
                )
            if fasta_path is None:
                raise ValueError("fasta_path is required for LPWGS foundation encoding.")
            if verbose:
                print(
                    "[WARN] LPWGS + foundation encoder is exploratory sequence-context encoding. "
                    "For CNV-focused production use, prefer --encoder lpwgs_cnv_profile."
                )
            interval_encoder = IntervalFoundationEncoder()
            return interval_encoder.encode(
                signal=signal_norm,
                input_dir=input_dir,
                fasta_path=fasta_path,
                output_dir=output_dir,
                model_key=selected_encoder,
                model_name=model_name,
                model_root=model_root,
                peak_mode=peak_mode,
                window_size=window_size,
                max_intervals_per_file=max_intervals_per_file,
                batch_size=batch_size,
                seed=seed,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                device=device,
                verbose=verbose,
            )

        raise ValueError(
            f"Signal '{signal_norm}' encoder '{selected_encoder}' is unsupported. "
            f"Use one of foundation encoders {sorted(_FOUNDATION_ENCODERS)} or 'lpwgs_cnv_profile'."
        )

    if signal_norm in _VARIANT_SIGNALS:
        if selected_encoder != "vcf_signature":
            raise ValueError(
                f"Signal '{signal_norm}' currently supports encoder 'vcf_signature' only, "
                f"got '{selected_encoder}'."
            )
        var_encoder = VariantSignatureEncoder()
        return var_encoder.encode(
            signal=signal_norm,
            input_dir=input_dir,
            output_dir=output_dir,
            verbose=verbose,
        )

    raise RuntimeError(f"Unsupported signal: {signal_norm}")


def _default_output_dir(*, signal: str, encoder: str, mode: str) -> Path:
    data_root = get_data_root()
    return (data_root / "GSE243474" / "features" / signal / encoder / mode).resolve()


def _normalize_signal(signal: str) -> str:
    key = str(signal).strip().lower()
    if key not in _SUPPORTED_SIGNALS:
        raise ValueError(f"Unsupported blood signal '{signal}'. Supported: {_SUPPORTED_SIGNALS}")
    return key


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def _sample_id_from_path(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".vcf.gz"):
        return name[:-7]
    if name.lower().endswith(".vcf"):
        return name[:-4]
    return _strip_bed_suffix(name)


def _stable_downsample_vector(values: np.ndarray, target_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((target_bins,), dtype=np.float32)
    if values.size == target_bins:
        return values.astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_bins, dtype=np.float32)
    out = np.interp(dst_x, src_x, values.astype(np.float32, copy=False))
    return out.astype(np.float32, copy=False)


def _encode_cnv_dataframe(df: pd.DataFrame, *, target_bins: int) -> np.ndarray:
    if df.empty or "count" not in df.columns:
        base = np.zeros((6,), dtype=np.float32)
        contour = np.zeros((target_bins,), dtype=np.float32)
        return np.concatenate([base, contour], axis=0)

    work = df.copy()
    sort_cols = [c for c in ("chrom", "start", "end") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
    counts = work["count"].to_numpy(dtype=np.float32)
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    mad = float(np.median(np.abs(counts - np.median(counts))))
    p90 = float(np.percentile(counts, 90))
    p10 = float(np.percentile(counts, 10))
    cv = float(std / (mean + 1e-6))
    z = (counts - np.median(counts)) / (mad + 1e-6)
    amp_frac = float((z > 3.0).mean()) if z.size else 0.0
    del_frac = float((z < -3.0).mean()) if z.size else 0.0
    norm = counts / (mean + 1e-6)
    contour = _stable_downsample_vector(norm.astype(np.float32, copy=False), target_bins=target_bins)
    base = np.asarray([mean, std, cv, p10, p90, amp_frac - del_frac], dtype=np.float32)
    return np.concatenate([base, contour], axis=0).astype(np.float32, copy=False)


def _parse_info_field(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in info.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out[part] = "1"
    return out


def _iter_vcf_rows(path: Path) -> Iterable[tuple[str, str, str, str, str]]:
    if str(path).lower().endswith(".gz"):
        import gzip

        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            yield parts[0], parts[3], parts[4], parts[5], parts[7]


def _encode_vcf_file(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    n_var = 0
    n_snv = 0
    n_ins = 0
    n_del = 0
    n_other = 0
    n_transition = 0
    n_transversion = 0
    c_to_t = 0
    quals: list[float] = []
    afs: list[float] = []

    transition_pairs = {
        ("A", "G"),
        ("G", "A"),
        ("C", "T"),
        ("T", "C"),
    }
    for _, ref, alt, qual, info in _iter_vcf_rows(path):
        alt_alleles = [a for a in alt.split(",") if a and a != "."]
        if not alt_alleles:
            continue
        info_map = _parse_info_field(info)
        if "AF" in info_map:
            try:
                af_values = [float(x) for x in str(info_map["AF"]).split(",") if x]
                afs.extend(af_values)
            except ValueError:
                pass
        if qual not in {".", ""}:
            try:
                quals.append(float(qual))
            except ValueError:
                pass

        for alt_allele in alt_alleles:
            n_var += 1
            if len(ref) == 1 and len(alt_allele) == 1:
                n_snv += 1
                pair = (ref.upper(), alt_allele.upper())
                if pair in transition_pairs:
                    n_transition += 1
                else:
                    n_transversion += 1
                if pair == ("C", "T"):
                    c_to_t += 1
            elif len(ref) < len(alt_allele):
                n_ins += 1
            elif len(ref) > len(alt_allele):
                n_del += 1
            else:
                n_other += 1

    if n_var == 0:
        vec = np.zeros((10,), dtype=np.float32)
    else:
        ti_tv = float(n_transition / max(1, n_transversion))
        c_to_t_frac = float(c_to_t / max(1, n_snv))
        af_mean = float(np.mean(afs)) if afs else 0.0
        af_std = float(np.std(afs)) if afs else 0.0
        qual_mean = float(np.mean(quals)) if quals else 0.0
        qual_std = float(np.std(quals)) if quals else 0.0
        vec = np.asarray(
            [
                float(n_var),
                float(n_snv),
                float(n_ins),
                float(n_del),
                float(n_other),
                ti_tv,
                c_to_t_frac,
                af_mean,
                af_std,
                qual_mean + qual_std,
            ],
            dtype=np.float32,
        )

    stat = {
        "n_variants": int(n_var),
        "n_snv": int(n_snv),
        "n_ins": int(n_ins),
        "n_del": int(n_del),
        "n_other": int(n_other),
    }
    return vec, stat
