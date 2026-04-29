"""
Compare "Start page" and "Functional Categories" labels across multiple annotation workbooks.

For the first K dossiers (order from the first workbook), reports (stem, page) keys where
labels disagree between files that annotate that row.
Empty/NaN cells are treated as not-start / "(none)" for functional categories.

Fleiss' κ is computed on (stem, page) rows where **every** listed workbook has that row.

Run:
  python compare_annotation_labels.py
  python compare_annotation_labels.py --first-dossiers 5 --annotations "annotation 1.xlsx" ...
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

from annotation_io import (
    binary_start_label,
    find_start_column,
    read_annotation_sheet,
    stem_order_first_appearance,
)

SCRIPT_DIR = Path(__file__).resolve().parent

FUNC_COL_CANDIDATES = ["Functional Categories", "functional categories", "Functional categories"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fleiss_binary_table(
    key_to_bins: dict[tuple[str, int], dict[str, int]],
    rater_names: list[str],
) -> np.ndarray | None:
    """Rows = subjects with a rating from every rater; columns = [count 0, count 1]."""
    n = len(rater_names)
    rows: list[list[int]] = []
    for bins in key_to_bins.values():
        if len(bins) != n or set(bins.keys()) != set(rater_names):
            continue
        labels = [bins[name] for name in rater_names]
        rows.append([labels.count(0), labels.count(1)])
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def find_func_column(df: pd.DataFrame) -> str | None:
    for candidate in FUNC_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for c in df.columns:
        if str(c).strip().lower() == "functional categories":
            return c
    return None


def normalise_func_label(value) -> str:
    if pd.isna(value):
        return "(none)"
    return str(value).strip()


def fleiss_category_table(
    key_to_cats: dict[tuple[str, int], dict[str, str]],
    rater_names: list[str],
    categories: list[str],
) -> np.ndarray | None:
    """Rows = subjects rated by every rater; columns = one per category (counts)."""
    n = len(rater_names)
    cat_index = {c: i for i, c in enumerate(categories)}
    rows: list[list[int]] = []
    for cats in key_to_cats.values():
        if len(cats) != n or set(cats.keys()) != set(rater_names):
            continue
        row = [0] * len(categories)
        for name in rater_names:
            row[cat_index[cats[name]]] += 1
        rows.append(row)
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare Start page and Functional Categories labels across xlsx files."
    )
    p.add_argument("--first-dossiers", type=int, default=5, help="Number of dossiers to compare.")
    p.add_argument(
        "--annotations",
        type=Path,
        nargs="*",
        default=[
            SCRIPT_DIR / "annotation 1.xlsx",
            SCRIPT_DIR / "annotation 2.xlsx",
            SCRIPT_DIR / "annotation 3.xlsx",
            SCRIPT_DIR / "annotation 4.xlsx",
        ],
        help="Annotation workbooks (same order as training merge tie-break).",
    )
    args = p.parse_args()

    paths = [x.expanduser().resolve() for x in args.annotations]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    dfs = {path.name: read_annotation_sheet(path) for path in paths}
    base_name = paths[0].name
    first_stems = stem_order_first_appearance(dfs[base_name])[: args.first_dossiers]
    first_set = set(first_stems)
    order = {s: i for i, s in enumerate(first_stems)}
    rater_names = [p.name for p in paths]

    print(f"Compared first {args.first_dossiers} dossiers (stem order from {base_name}):")
    print(first_stems)

    # ------------------------------------------------------------------
    # START PAGE comparison
    # ------------------------------------------------------------------
    key_to_bins: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)

    for path in paths:
        df = dfs[path.name]
        start_col = find_start_column(df)
        for _, row in df.iterrows():
            if pd.isna(row.get("image path")):
                continue
            stem = Path(str(row["image path"]).strip()).stem
            if stem not in first_set:
                continue
            page = int(row["page number"])
            key_to_bins[(stem, page)][path.name] = binary_start_label(row[start_col])

    start_conflicts = []
    for key in sorted(key_to_bins.keys(), key=lambda k: (order.get(k[0], 10**9), k[1])):
        bins = key_to_bins[key]
        if len(set(bins.values())) > 1:
            start_conflicts.append((key, bins))

    table_start = fleiss_binary_table(key_to_bins, rater_names)
    print("\n" + "=" * 60)
    print("START PAGE AGREEMENT")
    print("=" * 60)
    if table_start is None:
        print("Fleiss' kappa: skipped (no (stem, page) rows present in all workbooks).")
    elif len(paths) < 2:
        print("Fleiss' kappa: skipped (need at least two annotation files).")
    else:
        k = fleiss_kappa(table_start)
        print(
            f"Fleiss' kappa (binary Start page, {int(table_start[0].sum())} raters, "
            f"{table_start.shape[0]} subjects with complete ratings): {k:.4f}"
        )
    print(f"\nBinary disagreements (yes vs no): {len(start_conflicts)}")
    for (stem, page), bins in start_conflicts[:80]:
        print(f"  {stem}  page {page:>4}: {bins}")
    if len(start_conflicts) > 80:
        print(f"  ... and {len(start_conflicts) - 80} more")

    # ------------------------------------------------------------------
    # FUNCTIONAL CATEGORIES comparison
    # ------------------------------------------------------------------
    key_to_cats: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)

    func_cols_found: dict[str, str | None] = {}
    for path in paths:
        df = dfs[path.name]
        col = find_func_column(df)
        func_cols_found[path.name] = col
        if col is None:
            print(f"\n[WARNING] No 'Functional Categories' column found in {path.name} – skipping.")
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("image path")):
                continue
            stem = Path(str(row["image path"]).strip()).stem
            if stem not in first_set:
                continue
            page = int(row["page number"])
            key_to_cats[(stem, page)][path.name] = normalise_func_label(row[col])

    raters_with_func = [name for name in rater_names if func_cols_found.get(name) is not None]
    all_cats = sorted({cat for cats in key_to_cats.values() for cat in cats.values()})

    func_conflicts = []
    for key in sorted(key_to_cats.keys(), key=lambda k: (order.get(k[0], 10**9), k[1])):
        cats = key_to_cats[key]
        if len(set(cats.values())) > 1:
            func_conflicts.append((key, cats))

    table_func = fleiss_category_table(key_to_cats, raters_with_func, all_cats)

    print("\n" + "=" * 60)
    print("FUNCTIONAL CATEGORIES AGREEMENT")
    print("=" * 60)
    print(f"Categories observed: {all_cats}")
    if table_func is None:
        print("\nFleiss' kappa: skipped (no (stem, page) rows present in all workbooks).")
    elif len(raters_with_func) < 2:
        print("\nFleiss' kappa: skipped (need at least two files with Functional Categories).")
    else:
        k_func = fleiss_kappa(table_func)
        print(
            f"\nFleiss' kappa (Functional Categories, {len(raters_with_func)} raters, "
            f"{table_func.shape[0]} subjects with complete ratings): {k_func:.4f}"
        )

    print(f"\nFunctional category disagreements: {len(func_conflicts)}")
    for (stem, page), cats in func_conflicts[:80]:
        # Show only the raters that contributed to this page
        label_str = "  |  ".join(
            f"{name.replace('annotation ', 'A').replace('.xlsx', '')}: {cats[name]}"
            for name in rater_names
            if name in cats
        )
        print(f"  {stem}  page {page:>4}:  {label_str}")
    if len(func_conflicts) > 80:
        print(f"  ... and {len(func_conflicts) - 80} more")


if __name__ == "__main__":
    main()
