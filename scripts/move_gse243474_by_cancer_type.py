from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl  # noqa: F401
import pandas as pd


def _normalize_text(value: Any) -> str:
    """Normalize free text for robust matching."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[\s_-]+", "", text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _normalize_col_name(value: Any) -> str:
    """Normalize column names for fuzzy column matching."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_matching_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find a column by exact/fuzzy normalized name matching."""
    if not candidates:
        return None

    col_norm_map = {col: _normalize_col_name(col) for col in columns}
    cand_norms = [_normalize_col_name(c) for c in candidates if _normalize_col_name(c)]

    for cand in cand_norms:
        for col, col_norm in col_norm_map.items():
            if col_norm == cand:
                return col

    for cand in cand_norms:
        for col, col_norm in col_norm_map.items():
            if cand in col_norm or col_norm in cand:
                return col

    return None


def _is_table_s1_sheet(sheet_name: str) -> bool:
    """Heuristic for locating the 'Table S1' sheet."""
    n = re.sub(r"[^a-z0-9]+", "", sheet_name.lower())
    return n in {"tables1", "table01s1", "tables01"} or ("table" in n and "s1" in n)


def _load_metadata_sheet(
    metadata_xlsx: str | Path,
    study_col_candidates: list[str],
    cancer_col_candidates: list[str],
) -> tuple[pd.DataFrame, str, str, str]:
    """Load metadata from the best sheet containing required columns."""
    xlsx_path = Path(metadata_xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_names = xls.sheet_names

    preferred_sheets = [s for s in sheet_names if _is_table_s1_sheet(s)]
    other_sheets = [s for s in sheet_names if s not in preferred_sheets]
    search_order = preferred_sheets + other_sheets

    def detect_header_row(sheet_name: str, max_scan_rows: int = 50) -> int | None:
        """Detect which row contains actual table headers."""
        raw = pd.read_excel(
            xlsx_path,
            sheet_name=sheet_name,
            engine="openpyxl",
            header=None,
            nrows=max_scan_rows,
            dtype=object,
        )
        if raw.empty:
            return None

        for ridx in range(len(raw)):
            row_vals = [str(v).strip() for v in raw.iloc[ridx].tolist() if pd.notna(v)]
            if not row_vals:
                continue
            study_col = _find_matching_column(row_vals, study_col_candidates)
            cancer_col = _find_matching_column(row_vals, cancer_col_candidates)
            if study_col and cancer_col:
                return ridx
        return None

    for sheet in search_order:
        try:
            header_row = detect_header_row(sheet)
            if header_row is None:
                continue
            df = pd.read_excel(
                xlsx_path,
                sheet_name=sheet,
                engine="openpyxl",
                header=header_row,
                dtype=object,
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue

        columns = [str(c) for c in df.columns]
        study_col = _find_matching_column(columns, study_col_candidates)
        cancer_col = _find_matching_column(columns, cancer_col_candidates)

        if study_col and cancer_col:
            meta = pd.DataFrame(
                {
                    "study_name": df[study_col].where(df[study_col].notna(), "").astype(str).str.strip(),
                    "cancer_type": df[cancer_col].where(df[cancer_col].notna(), "").astype(str).str.strip(),
                }
            )
            meta = meta[meta["study_name"] != ""].copy()
            meta["study_norm"] = meta["study_name"].map(_normalize_text)
            meta["cancer_norm"] = meta["cancer_type"].map(_normalize_text)
            meta = meta[meta["study_norm"] != ""].copy()

            if not meta.empty:
                return meta, sheet, study_col, cancer_col

    raise ValueError(
        f"Could not find a valid sheet with required columns in {xlsx_path}. "
        f"Tried sheets: {sheet_names}"
    )


def _extract_candidate_tokens(filename: str) -> list[str]:
    """Extract candidate sample alias tokens from filename with priority."""
    name = filename
    if name.lower().endswith(".bed.gz"):
        base_name = name[:-7]
    else:
        base_name = Path(name).stem

    # Remove leading GSM id to get the practical middle part, e.g.:
    # GSM7787767_advPC_13_JS8585_S72 -> advPC_13_JS8585_S72
    body = re.sub(r"(?i)^GSM\d+[_-]?", "", base_name)

    tokens: list[str] = []
    seen_norm: set[str] = set()

    def add_token(token: str) -> None:
        token = str(token).strip("_- ")
        if not token:
            return
        norm = _normalize_text(token)
        if norm and norm not in seen_norm:
            tokens.append(token)
            seen_norm.add(norm)

    # Priority 1: strong pattern tokens
    patterns = [
        re.compile(r"(mPC\d+)", re.IGNORECASE),
        re.compile(r"(advPC[_-]?\d+)", re.IGNORECASE),
        re.compile(r"(BC\d+[A-Za-z0-9]*)", re.IGNORECASE),
    ]
    for idx, pattern in enumerate(patterns):
        for m in pattern.finditer(body):
            token = m.group(1)
            add_token(token)
            if idx == 2:
                # Add shortened BC<digits> fallback, e.g. BC287K27R1 -> BC287
                m2 = re.match(r"(?i)^(BC)(\d+)", token)
                if m2:
                    add_token(f"{m2.group(1)}{m2.group(2)}")

    # Priority 2: body-level token (study_name often appears as filename middle segment)
    add_token(body)

    # Priority 3: derived variants by stripping technical suffixes
    # Examples:
    #   advPC_13_JS8585_S72 -> advPC_13
    #   Ac_mPC15_3R -> Ac_mPC15
    #   BC287K27R1 -> BC287K27
    body_simplified = re.sub(r"(?i)_JS\d+_S\d+$", "", body)
    body_simplified = re.sub(r"(?i)_S\d+$", "", body_simplified)
    body_simplified = re.sub(r"(?i)_all$", "", body_simplified)
    body_simplified = re.sub(r"(?i)([_-]?R\d+)$", "", body_simplified)
    add_token(body_simplified)

    # Also add fragments split by underscores/hyphens to catch partial study aliases.
    parts = [p for p in re.split(r"[_-]+", body_simplified) if p]
    for p in parts:
        if re.match(r"(?i)^(js\d+|s\d+|r\d+)$", p):
            continue
        add_token(p)

    return tokens


def _is_breast_cancer(cancer_type: str) -> bool:
    """Return True only when cancer_type equals 'Breast' (case-insensitive)."""
    return _normalize_text(cancer_type) == "breast"


def _safe_log_value(value: Any) -> str:
    """Convert a value to single-line, tab-safe text."""
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def _unique_destination_path(dst: Path, used_paths_lower: set[str]) -> Path:
    """Find a non-conflicting destination path by appending _dupN if needed."""
    suffix = "".join(dst.suffixes)
    stem = dst.name[: -len(suffix)] if suffix else dst.name

    candidate = dst
    i = 1
    while candidate.exists() or str(candidate).lower() in used_paths_lower:
        candidate = dst.with_name(f"{stem}_dup{i}{suffix}")
        i += 1

    used_paths_lower.add(str(candidate).lower())
    return candidate


def move_files_by_cancer_type_from_study_name(
    root_dir: str,
    metadata_xlsx: str,
    study_col_candidates: list[str] = None,
    cancer_col_candidates: list[str] = None,
    breast_dir_name: str = "breast",
    others_dir_name: str = "others",
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Move root-level files into breast/others based on metadata study_name -> cancer_type matching."""
    if study_col_candidates is None:
        study_col_candidates = ["study_name", "study name", "Study Name", "studyname"]
    if cancer_col_candidates is None:
        cancer_col_candidates = ["cancer_type", "cancer type", "Cancer Type", "cancertype"]

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    breast_dir = root_path / breast_dir_name
    others_dir = root_path / others_dir_name

    metadata_df, used_sheet, used_study_col, used_cancer_col = _load_metadata_sheet(
        metadata_xlsx=metadata_xlsx,
        study_col_candidates=study_col_candidates,
        cancer_col_candidates=cancer_col_candidates,
    )

    files = [
        p
        for p in root_path.iterdir()
        if p.is_file() and re.match(r"(?i)^GSM\d+.*\.bed(?:\.gz)?$", p.name)
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = root_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"move_log_{timestamp}.txt"

    moved_to_breast = 0
    moved_to_others = 0
    unmatched_count = 0
    ambiguous_count = 0
    ambiguous_resolved_count = 0
    error_count = 0

    moved_breast_examples: list[dict] = []
    moved_others_examples: list[dict] = []
    unmatched_examples: list[dict] = []
    ambiguous_examples: list[dict] = []
    error_examples: list[dict] = []

    used_paths_lower: set[str] = set()

    if verbose:
        print(f"[INFO] Root dir: {root_path}")
        print(f"[INFO] Metadata: {metadata_xlsx}")
        print(f"[INFO] Using sheet: {used_sheet} (study='{used_study_col}', cancer='{used_cancer_col}')")
        print(f"[INFO] Files to process (root-level only): {len(files)}")
        print(f"[INFO] dry_run={dry_run}")

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(
            "filename\tGSM\tchosen_token\tmatched_study_name\tcancer_type\taction\tsrc_path\tdst_path\n"
        )

        for i, src_path in enumerate(files, start=1):
            filename = src_path.name
            gsm_match = re.match(r"^(GSM\d+)", filename, flags=re.IGNORECASE)
            gsm_id = gsm_match.group(1) if gsm_match else ""

            chosen_token = ""
            matched_study_name = ""
            cancer_type = ""
            action = ""
            dst_path = None

            try:
                tokens = _extract_candidate_tokens(filename)

                token_hits: list[tuple[str, str, int, pd.DataFrame]] = []
                for t_idx, token in enumerate(tokens):
                    token_norm = _normalize_text(token)
                    if not token_norm:
                        continue
                    # First try exact normalized match to avoid substring collisions (e.g., LJH1 vs LJH10).
                    exact_rows = metadata_df.loc[metadata_df["study_norm"] == token_norm]
                    if not exact_rows.empty:
                        hit_rows = exact_rows
                    else:
                        mask = metadata_df["study_norm"].str.contains(
                            re.escape(token_norm),
                            regex=True,
                            na=False,
                        )
                        hit_rows = metadata_df.loc[mask]
                    if not hit_rows.empty:
                        token_hits.append((token, token_norm, t_idx, hit_rows))

                if not token_hits:
                    unmatched_count += 1
                    action = "unmatched"
                    dst_dir = others_dir
                    chosen_token = tokens[0] if tokens else ""
                    cancer_type = ""
                    matched_study_name = ""
                else:
                    best = max(token_hits, key=lambda x: (len(x[1]), -x[2]))
                    chosen_token, _, _, hit_rows = best

                    matched_studies = hit_rows["study_name"].dropna().astype(str).unique().tolist()
                    matched_study_name = " | ".join(matched_studies[:5])

                    if len(hit_rows) == 1:
                        cancer_type = str(hit_rows["cancer_type"].iloc[0])
                        action = "moved"
                    else:
                        cancer_norms = set(hit_rows["cancer_norm"].fillna("").astype(str).tolist())
                        if len(cancer_norms) == 1:
                            ambiguous_resolved_count += 1
                            cancer_type = str(hit_rows["cancer_type"].iloc[0])
                            action = "moved"
                        else:
                            ambiguous_count += 1
                            action = "ambiguous"
                            cancer_type = "AMBIGUOUS_CONFLICT"

                    dst_dir = breast_dir if _is_breast_cancer(cancer_type) else others_dir

                dst_path_candidate = dst_dir / filename
                dst_path = _unique_destination_path(dst_path_candidate, used_paths_lower)

                if not dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_path), str(dst_path))

                if action == "moved":
                    if dst_dir == breast_dir:
                        moved_to_breast += 1
                        if len(moved_breast_examples) < 20:
                            moved_breast_examples.append(
                                {
                                    "filename": filename,
                                    "chosen_token": chosen_token,
                                    "matched_study_name": matched_study_name,
                                    "cancer_type": cancer_type,
                                    "dst_path": str(dst_path),
                                }
                            )
                    else:
                        moved_to_others += 1
                        if len(moved_others_examples) < 20:
                            moved_others_examples.append(
                                {
                                    "filename": filename,
                                    "chosen_token": chosen_token,
                                    "matched_study_name": matched_study_name,
                                    "cancer_type": cancer_type,
                                    "dst_path": str(dst_path),
                                }
                            )
                elif action == "unmatched":
                    moved_to_others += 1
                    if len(unmatched_examples) < 20:
                        unmatched_examples.append(
                            {
                                "filename": filename,
                                "chosen_token": chosen_token,
                                "dst_path": str(dst_path),
                            }
                        )
                elif action == "ambiguous":
                    moved_to_others += 1
                    if len(ambiguous_examples) < 20:
                        ambiguous_examples.append(
                            {
                                "filename": filename,
                                "chosen_token": chosen_token,
                                "matched_study_name": matched_study_name,
                                "dst_path": str(dst_path),
                            }
                        )

                log_f.write(
                    "\t".join(
                        [
                            _safe_log_value(filename),
                            _safe_log_value(gsm_id),
                            _safe_log_value(chosen_token),
                            _safe_log_value(matched_study_name),
                            _safe_log_value(cancer_type),
                            _safe_log_value(action),
                            _safe_log_value(src_path),
                            _safe_log_value(dst_path if dst_path else ""),
                        ]
                    )
                    + "\n"
                )

                if verbose and (i % 50 == 0 or i == len(files)):
                    print(f"[INFO] Processed {i}/{len(files)}")

            except Exception as e:
                error_count += 1
                action = "error"
                err_msg = f"{type(e).__name__}: {e}"

                if len(error_examples) < 20:
                    error_examples.append(
                        {
                            "filename": filename,
                            "error": err_msg,
                            "src_path": str(src_path),
                        }
                    )

                log_f.write(
                    "\t".join(
                        [
                            _safe_log_value(filename),
                            _safe_log_value(gsm_id),
                            _safe_log_value(chosen_token),
                            _safe_log_value(matched_study_name),
                            _safe_log_value(cancer_type),
                            _safe_log_value(action),
                            _safe_log_value(src_path),
                            _safe_log_value(dst_path if dst_path else ""),
                        ]
                    )
                    + "\n"
                )

                if verbose:
                    print(f"[ERROR] {filename}: {err_msg}")

    summary = {
        "total_files_scanned": len(files),
        "moved_to_breast": moved_to_breast,
        "moved_to_others": moved_to_others,
        "unmatched": unmatched_count,
        "ambiguous": ambiguous_count,
        "errors": error_count,
        "ambiguous_resolved": ambiguous_resolved_count,
        "log_path": str(log_path),
        "metadata_sheet_used": used_sheet,
        "metadata_study_col_used": used_study_col,
        "metadata_cancer_col_used": used_cancer_col,
        "examples_moved_to_breast": moved_breast_examples[:20],
        "examples_moved_to_others": moved_others_examples[:20],
        "examples_unmatched": unmatched_examples[:20],
        "examples_ambiguous": ambiguous_examples[:20],
        "examples_errors": error_examples[:20],
    }

    if verbose:
        print("[DONE] Summary:")
        print(f"  total_files_scanned: {summary['total_files_scanned']}")
        print(f"  moved_to_breast:     {summary['moved_to_breast']}")
        print(f"  moved_to_others:     {summary['moved_to_others']}")
        print(f"  unmatched:           {summary['unmatched']}")
        print(f"  ambiguous:           {summary['ambiguous']}")
        print(f"  errors:              {summary['errors']}")
        print(f"  ambiguous_resolved:  {summary['ambiguous_resolved']}")
        print(f"  log_path:            {summary['log_path']}")

    return summary


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from liquidbiopsy_agent.utils.storage import resolve_data_path

    root_dir = resolve_data_path("GSE243474", path_kind="GSE243474 root", must_exist=True)
    metadata_xlsx = resolve_data_path(
        "GSE243474/meta/41591_2023_2605_MOESM2_ESM.xlsx",
        path_kind="GSE243474 metadata xlsx",
        must_exist=True,
    )

    move_files_by_cancer_type_from_study_name(
        root_dir=root_dir,
        metadata_xlsx=metadata_xlsx,
        dry_run=False,
        verbose=True,
    )
