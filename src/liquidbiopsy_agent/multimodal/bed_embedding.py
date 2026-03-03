from __future__ import annotations

import gzip
import random
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

from .dna_foundation_encoders import BaseDNAFoundationEncoder, build_dna_foundation_encoder


def _strip_bed_suffix(file_name: str) -> str:
    if file_name.lower().endswith(".bed.gz"):
        return file_name[:-7]
    if file_name.lower().endswith(".bed"):
        return file_name[:-4]
    return Path(file_name).stem


def _extract_gsm_and_alias(file_name: str) -> tuple[str, str]:
    """Extract GSM id and a likely sample alias from a BED filename."""
    base = _strip_bed_suffix(file_name)
    m = re.match(r"^(GSM\d+)", base, flags=re.IGNORECASE)
    gsm = m.group(1) if m else ""
    body = re.sub(r"(?i)^GSM\d+[_-]?", "", base)

    alias_patterns = [
        re.compile(r"(BC\d+[A-Za-z0-9]*)", re.IGNORECASE),
        re.compile(r"(mPC\d+)", re.IGNORECASE),
        re.compile(r"(advPC[_-]?\d+)", re.IGNORECASE),
    ]
    for p in alias_patterns:
        mm = p.search(body)
        if mm:
            return gsm, mm.group(1)

    first_seg = re.split(r"[_-]+", body)[0] if body else ""
    return gsm, first_seg


def _open_text_auto(path: Path):
    if path.suffix.lower() == ".gz" or str(path).lower().endswith(".bed.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _is_narrow_peak_file(file_name: str) -> bool:
    return bool(re.search(r"(narrowpeak|sorted_peaks)", file_name, flags=re.IGNORECASE))


def _collect_bed_files(input_path: Path, peak_mode: str) -> tuple[list[Path], str]:
    all_bed = sorted([p for p in input_path.iterdir() if p.is_file() and re.search(r"\.bed(\.gz)?$", p.name, re.I)])
    if not all_bed:
        return [], "none"

    narrow = [p for p in all_bed if _is_narrow_peak_file(p.name)]
    raw = [p for p in all_bed if not _is_narrow_peak_file(p.name)]

    if peak_mode == "direct_narrow_peak":
        if narrow:
            return narrow, "narrow_peak"
        return all_bed, "all_bed_fallback"

    if raw:
        return raw, "raw_bed"
    return all_bed, "all_bed_fallback"


def _reservoir_sample_intervals(
    bed_path: Path,
    max_intervals: int,
    seed: int,
) -> tuple[list[tuple[str, int, int]], int]:
    """Reservoir-sample BED intervals and return sampled + total valid count."""
    sampled: list[tuple[str, int, int]] = []
    valid_count = 0
    rng = random.Random(seed)

    with _open_text_auto(bed_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("track") or s.startswith("browser"):
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0].strip()
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if not chrom or end <= start:
                continue

            valid_count += 1
            item = (chrom, start, end)
            if len(sampled) < max_intervals:
                sampled.append(item)
            else:
                j = rng.randint(0, valid_count - 1)
                if j < max_intervals:
                    sampled[j] = item

    return sampled, valid_count


def _build_chrom_resolver(fasta_obj: Fasta):
    names = set(str(k) for k in fasta_obj.keys())
    cache: dict[str, str | None] = {}

    def resolve(chrom: str) -> str | None:
        if chrom in cache:
            return cache[chrom]
        c = chrom
        out = None
        if c in names:
            out = c
        elif c.startswith("chr") and c[3:] in names:
            out = c[3:]
        elif (not c.startswith("chr")) and f"chr{c}" in names:
            out = f"chr{c}"
        elif c == "MT" and "chrM" in names:
            out = "chrM"
        elif c == "chrM" and "MT" in names:
            out = "MT"
        cache[chrom] = out
        return out

    return resolve


def _fetch_fixed_window_sequence(
    fasta_obj: Fasta,
    chrom_resolver,
    chrom: str,
    start: int,
    end: int,
    window_size: int,
) -> tuple[str, tuple[int, int]] | None:
    """
    Fetch fixed-length sequence around interval center.

    Boundary strategy:
      - Clamp coordinates to chromosome bounds
      - Pad with 'N' to keep fixed window length
    """
    resolved = chrom_resolver(chrom)
    if resolved is None:
        return None

    center = (start + end) // 2
    half = window_size // 2
    raw_w_start = center - half
    w_start = raw_w_start
    w_end = w_start + window_size

    chrom_len = len(fasta_obj[resolved])
    left_pad = max(0, -w_start)
    right_pad = max(0, w_end - chrom_len)
    w_start = max(0, w_start)
    w_end = min(chrom_len, w_end)

    raw_seq = fasta_obj[resolved][w_start:w_end]
    if hasattr(raw_seq, "seq"):
        seq = str(raw_seq.seq).upper()
    else:
        seq = str(raw_seq).upper()
    seq = re.sub(r"[^ACGTN]", "N", seq)
    seq = ("N" * left_pad) + seq + ("N" * right_pad)
    if len(seq) < window_size:
        seq += "N" * (window_size - len(seq))
    elif len(seq) > window_size:
        seq = seq[:window_size]
    rel_start = max(0, start - raw_w_start)
    rel_end = min(window_size, end - raw_w_start)
    if rel_end < rel_start:
        rel_end = rel_start
    return seq, (rel_start, rel_end)


def encode_bed_folder_to_embeddings(
    input_dir: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    model_key: str = "ntv2",
    model_name: str | None = None,
    model_root: str | Path | None = None,
    window_size: int | None = None,
    max_intervals_per_file: int | None = None,
    batch_size: int | None = None,
    seed: int = 42,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    device: str = "auto",
    verbose: bool = True,
    preview_samples: int = 3,
    preview_dims: int = 8,
    peak_mode: str = "mask_from_raw",
) -> dict[str, Any]:
    """
    Encode each BED/BED.GZ file in a folder into one fixed-size embedding vector.

    Returns a summary dict and writes:
      - embeddings.parquet (if available) and embeddings.csv
      - metadata.csv
    """
    input_path = Path(input_dir)
    fasta_file = Path(fasta_path)
    if output_dir is None:
        output_path = input_path / "embeddings"
    else:
        output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found or invalid: {input_path}")
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    peak_mode_norm = str(peak_mode).strip().lower()
    if peak_mode_norm not in {"direct_narrow_peak", "mask_from_raw"}:
        raise ValueError("peak_mode must be one of: direct_narrow_peak, mask_from_raw")
    output_path.mkdir(parents=True, exist_ok=True)

    encoder: BaseDNAFoundationEncoder = build_dna_foundation_encoder(
        model_key=model_key,
        model_name=model_name,
        model_root=model_root,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=device,
        force_peak_focus_mask=(peak_mode_norm == "mask_from_raw"),
    )
    effective_window_size = window_size if window_size and window_size > 0 else encoder.input_profile.default_window_size
    effective_max_intervals = (
        max_intervals_per_file
        if max_intervals_per_file and max_intervals_per_file > 0
        else encoder.input_profile.default_max_intervals_per_file
    )
    effective_batch_size = (
        batch_size
        if batch_size and batch_size > 0
        else encoder.input_profile.default_batch_size
    )
    embedding_dim = encoder.embedding_dim

    if verbose:
        print(f"[INFO] Input dir: {input_path}")
        print(f"[INFO] FASTA: {fasta_file}")
        print(f"[INFO] Output dir: {output_path}")
        print(f"[INFO] Device: {encoder.device}")
        print(f"[INFO] Requested model key: {model_key}")
        print(f"[INFO] Model key: {encoder.model_key}")
        print(f"[INFO] Loaded model: {encoder.model_name}")
        print(f"[INFO] Embedding dim: {embedding_dim}")
        print(f"[INFO] Window size: {effective_window_size}")
        print(f"[INFO] Max intervals/file: {effective_max_intervals}")
        print(f"[INFO] Batch size: {effective_batch_size}")

    fasta_obj = Fasta(str(fasta_file), as_raw=True, sequence_always_upper=True)
    chrom_resolver = _build_chrom_resolver(fasta_obj)

    files, selected_input_kind = _collect_bed_files(input_path, peak_mode=peak_mode_norm)
    if not files:
        raise ValueError(f"No .bed/.bed.gz files found in: {input_path}")
    if peak_mode_norm == "mask_from_raw":
        encoder.set_runtime_peak_focus_mask(True)
    if verbose:
        print(f"[INFO] Peak mode: {peak_mode_norm}")
        print(f"[INFO] Input kind selected: {selected_input_kind}")
        print(f"[INFO] Files selected: {len(files)}")

    emb_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    iterator = tqdm(files, desc="Encoding BED files") if verbose else files

    for bed_path in iterator:
        file_name = bed_path.name
        gsm, alias = _extract_gsm_and_alias(file_name)
        sample_id = _strip_bed_suffix(file_name)
        status = "ok"
        error_msg = ""
        num_intervals_used = 0
        num_intervals_seen = 0

        try:
            file_seed = seed + int(zlib.crc32(file_name.encode("utf-8")) & 0xFFFFFFFF)
            sampled_intervals, total_valid = _reservoir_sample_intervals(
                bed_path=bed_path,
                max_intervals=effective_max_intervals,
                seed=file_seed,
            )
            num_intervals_seen = int(total_valid)

            seqs: list[str] = []
            interval_spans: list[tuple[int, int]] = []
            for chrom, start, end in sampled_intervals:
                fetched = _fetch_fixed_window_sequence(
                    fasta_obj=fasta_obj,
                    chrom_resolver=chrom_resolver,
                    chrom=chrom,
                    start=start,
                    end=end,
                    window_size=effective_window_size,
                )
                if fetched:
                    seq, span = fetched
                    seqs.append(seq)
                    interval_spans.append(span)

            if not seqs:
                status = "no_valid_intervals"
                error_msg = "No usable intervals after sampling/FASTA lookup."
                emb_vec = np.full((embedding_dim,), np.nan, dtype=np.float32)
            else:
                interval_embeddings = encoder.encode_sequences(
                    seqs,
                    batch_size=effective_batch_size,
                    interval_spans=interval_spans,
                )
                emb_vec = interval_embeddings.mean(axis=0).astype(np.float32)
                num_intervals_used = int(interval_embeddings.shape[0])

        except Exception as exc:
            status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            emb_vec = np.full((embedding_dim,), np.nan, dtype=np.float32)
            if verbose:
                print(f"[WARN] {file_name}: {error_msg}")

        meta_rows.append(
            {
                "sample_id": sample_id,
                "file_name": file_name,
                "GSM": gsm,
                "alias": alias,
                "num_intervals_used": num_intervals_used,
                "num_intervals_seen": num_intervals_seen,
                "window_size_used": effective_window_size,
                "batch_size_used": effective_batch_size,
                "peak_mode": peak_mode_norm,
                "input_kind_selected": selected_input_kind,
                "requested_model_key": model_key,
                "resolved_model_key": encoder.model_key,
                "status": status,
                "error_msg": error_msg,
            }
        )
        row = {"sample_id": sample_id}
        for i, v in enumerate(emb_vec):
            row[f"emb_{i:04d}"] = float(v) if np.isfinite(v) else np.nan
        emb_rows.append(row)
        if preview_samples > 0 and len(preview_rows) < preview_samples:
            head_dims = max(1, int(preview_dims))
            emb_head = emb_vec[:head_dims].astype(np.float32, copy=False)
            preview_rows.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "status": status,
                    "num_intervals_used": num_intervals_used,
                    "embedding_dim": int(emb_vec.shape[0]),
                    "embedding_head": emb_head,
                }
            )

    emb_df = pd.DataFrame(emb_rows)
    meta_df = pd.DataFrame(meta_rows)

    emb_parquet = output_path / "embeddings.parquet"
    emb_csv = output_path / "embeddings.csv"
    meta_csv = output_path / "metadata.csv"

    parquet_ok = True
    try:
        emb_df.to_parquet(emb_parquet, index=False)
    except Exception as exc:
        parquet_ok = False
        if verbose:
            print(f"[WARN] Failed to write parquet ({exc}), fallback to CSV only.")
    emb_df.to_csv(emb_csv, index=False)
    meta_df.to_csv(meta_csv, index=False)

    status_counts = meta_df["status"].value_counts(dropna=False).to_dict()
    summary = {
        "input_dir": str(input_path),
        "fasta_path": str(fasta_file),
        "output_dir": str(output_path),
        "model_name": encoder.model_name,
        "requested_model_key": model_key,
        "model_key": encoder.model_key,
        "embedding_dim": embedding_dim,
        "window_size": effective_window_size,
        "max_intervals_per_file": effective_max_intervals,
        "batch_size": effective_batch_size,
        "peak_mode": peak_mode_norm,
        "input_kind_selected": selected_input_kind,
        "files_processed": len(files),
        "status_counts": status_counts,
        "embeddings_parquet": str(emb_parquet) if parquet_ok else None,
        "embeddings_csv": str(emb_csv),
        "metadata_csv": str(meta_csv),
        "preview_samples_printed": len(preview_rows),
    }

    if verbose:
        if preview_rows:
            print("[PREVIEW] First encoded samples (truncated):")
            for idx, item in enumerate(preview_rows, start=1):
                head_str = np.array2string(
                    item["embedding_head"],
                    precision=4,
                    separator=", ",
                    max_line_width=240,
                )
                print(
                    f"[PREVIEW] #{idx} sample_id={item['sample_id']} "
                    f"status={item['status']} intervals={item['num_intervals_used']} "
                    f"emb_size={item['embedding_dim']} head={head_str}"
                )
        print("[DONE] Encoding complete.")
        print(f"[DONE] files_processed: {summary['files_processed']}")
        print(f"[DONE] status_counts: {summary['status_counts']}")
        if summary["embeddings_parquet"]:
            print(f"[DONE] embeddings_parquet: {summary['embeddings_parquet']}")
        print(f"[DONE] embeddings_csv: {summary['embeddings_csv']}")
        print(f"[DONE] metadata_csv: {summary['metadata_csv']}")

    return summary
