#!/usr/bin/env python
"""Reproduce every table and headline statistic in the write-up
(``StaB-ddG-unsup-writeup/main.tex``) from the data on disk — unambiguously and in one place.

Single source of truth: the IntAct ``feature_type`` -> weak-label classification is imported from
``stabddg.intact.dataset.IntactDataset`` (``feature_type_pos/neg/rate/no_effect/neutral``), so the
paper's tables can never silently drift from the training code.

Tables produced (each maps to a labelled object in main.tex):
  T1  IntAct feature-type taxonomy + counts + weak label   -> tab:intact-taxonomy
  T2  IntAct data funnel + class imbalance                 -> \\paragraph{Scale (data funnel)}
  T3  IntAct (trained) vs SKEMPI 2.0 overlap               -> tab:intact-skempi

Inputs (relative to the repo root):
  data/intact/df_intact_mutations_raw.parquet
  data/intact/df_intact_mutations.parquet
  data/intact/df_intact_mutations_filtered.parquet
  data/intact/df_intact_mutations_filtered_{train,valid,test}.parquet
  data/intact/df_assemblies_filtered.parquet
  data/SKEMPI/skempi_v2.csv
  paper_stats/sifts_pdb_uniprot.json      (SKEMPI PDB->UniProt via SIFTS; auto-fetched if missing)

Outputs (paper_stats/output/):
  stats.json   all numbers, machine-readable
  tables.tex   LaTeX table bodies matching main.tex

Run from the repo root:
  .venv/bin/python paper_stats/make_paper_tables.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pandas as pd

# --- make the repo root importable so `stabddg` resolves regardless of cwd ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stabddg.intact.dataset import IntactDataset as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO_ROOT, "data")
OUT = os.path.join(HERE, "output")
SIFTS_CACHE = os.path.join(HERE, "sifts_pdb_uniprot.json")

# Human-readable MI code per feature type (for the taxonomy table).
MI = {
    "mutation causing(MI:2227)": "MI:2227",
    "mutation increasing(MI:0382)": "MI:0382",
    "mutation increasing strength(MI:1132)": "MI:1132",
    "mutation decreasing(MI:0119)": "MI:0119",
    "mutation decreasing strength(MI:1133)": "MI:1133",
    "mutation disrupting(MI:0573)": "MI:0573",
    "mutation disrupting strength(MI:1128)": "MI:1128",
    "mutation with no effect(MI:2226)": "MI:2226",
    "mutation increasing rate(MI:1131)": "MI:1131",
    "mutation decreasing rate(MI:1130)": "MI:1130",
    "mutation disrupting rate(MI:1129)": "MI:1129",
    "mutation(MI:0118)": "MI:0118",
}


def _label_of(ft: str) -> str:
    """Weak label assigned by the training code (imported, not hard-coded here)."""
    if ft in D.feature_type_pos:
        return "+1"
    if ft in D.feature_type_neg:
        return "-1"
    if ft in D.feature_type_rate:
        return "0 (kinetic)"
    if ft in D.feature_type_no_effect:
        return "0"
    return "excluded"


def _category_of(ft: str) -> str:
    if ft in D.feature_type_pos:
        return "pos"
    if ft in D.feature_type_neg:
        return "neg"
    if ft in D.feature_type_rate:
        return "rate"
    if ft in D.feature_type_no_effect:
        return "no_effect"
    return "generic"


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA, "intact", name))


def _base_ac(ac: str) -> str:
    return str(ac).split("-")[0].upper()


# --------------------------------------------------------------------------------------
# T1 + T2 : IntAct taxonomy, counts, funnel
# --------------------------------------------------------------------------------------
def intact_taxonomy_and_funnel() -> dict:
    raw = len(_read("df_intact_mutations_raw.parquet"))
    parsed = len(_read("df_intact_mutations.parquet"))
    filt = _read("df_intact_mutations_filtered.parquet")

    vc = filt["feature_type"].value_counts()
    taxonomy = []
    for ft, count in vc.items():
        taxonomy.append({
            "feature_type": ft,
            "mi": MI.get(ft, "?"),
            "count": int(count),
            "category": _category_of(ft),
            "label": _label_of(ft),
        })

    def cat_total(cat):
        return int(sum(t["count"] for t in taxonomy if t["category"] == cat))

    filt_pos, filt_neg = cat_total("pos"), cat_total("neg")
    filt_dir = filt_pos + filt_neg

    # trained set = train+valid+test splits (WT-consistent, structure-mapped)
    splits = pd.concat(
        [_read(f"df_intact_mutations_filtered_{s}.parquet") for s in ("train", "valid", "test")],
        ignore_index=True,
    )
    splits["category"] = splits["feature_type"].map(_category_of)
    tr_pos = int((splits["category"] == "pos").sum())
    tr_neg = int((splits["category"] == "neg").sum())
    tr_pairs = pd.Series(
        [frozenset((_base_ac(a), _base_ac(p)))
         for a, p in zip(splits["affected_protein_ac"], splits["participant_protein"])]
    ).nunique()

    assemblies = _read("df_assemblies_filtered.parquet")

    return {
        "taxonomy": taxonomy,
        "funnel": {
            "raw_mutations": raw,
            "parsed_mutations": parsed,
            "filtered": int(len(filt)),
            "filtered_by_category": {c: cat_total(c) for c in
                                     ("pos", "neg", "no_effect", "rate", "generic")},
            "filtered_directional": filt_dir,
            "filtered_pct_positive": round(100 * filt_pos / filt_dir, 1),
            "assemblies_rows": int(len(assemblies)),
            "assemblies_unique": int(assemblies["biological_assembly"].nunique()),
            "assemblies_pdb_entries": int(assemblies["entry_id"].nunique()),
            "trained_rows": int(len(splits)),
            "trained_directional": tr_pos + tr_neg,
            "trained_pos": tr_pos,
            "trained_neg": tr_neg,
            "trained_pct_positive": round(100 * tr_pos / (tr_pos + tr_neg), 1),
            "trained_neutral_no_effect": int((splits["category"] == "no_effect").sum()),
            "trained_rate": int((splits["category"] == "rate").sum()),
            "trained_generic_ignored": int((splits["category"] == "generic").sum()),
            "trained_assemblies": int(splits["biological_assembly"].nunique()),
            "trained_pdb_entries": int(splits["entry_id"].nunique()),
            "trained_interface_pairs": int(tr_pairs),
        },
    }


# --------------------------------------------------------------------------------------
# T3 : SKEMPI overlap (needs SKEMPI PDB -> UniProt via SIFTS)
# --------------------------------------------------------------------------------------
def _load_or_fetch_sifts(pdbs: list[str]) -> dict:
    cache = {}
    if os.path.exists(SIFTS_CACHE):
        cache = json.load(open(SIFTS_CACHE))
    todo = [p for p in pdbs if p not in cache]
    if todo:
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch(pdb):
            url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}"
            try:
                with urllib.request.urlopen(url, timeout=25) as r:
                    d = json.load(r)
                chain2unp = {}
                for ac, info in d.get(pdb, {}).get("UniProt", {}).items():
                    for m in info.get("mappings", []):
                        chain2unp[m["chain_id"]] = ac
                return pdb, chain2unp
            except Exception as e:  # noqa: BLE001
                return pdb, {"__error__": str(e)}

        print(f"[SIFTS] fetching {len(todo)} PDB->UniProt mappings from PDBe ...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for fut in as_completed([ex.submit(fetch, p) for p in todo]):
                pdb, res = fut.result()
                cache[pdb] = res
        json.dump(cache, open(SIFTS_CACHE, "w"))
    return cache


def skempi_overlap() -> dict:
    sk = pd.read_csv(os.path.join(DATA, "SKEMPI", "skempi_v2.csv"), sep=";")
    pdbs = sorted(set(sk["#Pdb"].str.split("_").str[0].str.lower()))
    cache = _load_or_fetch_sifts(pdbs)

    sk_proteins, sk_pairs, sk_pdb = set(), set(), set()
    for pdbid in sk["#Pdb"].unique():
        parts = pdbid.split("_")
        pdb, sides = parts[0].lower(), parts[1:]
        sk_pdb.add(pdb.upper())
        chain2unp = cache.get(pdb, {})
        if not isinstance(chain2unp, dict) or "__error__" in chain2unp:
            continue
        side_unp = []
        for side in sides:
            u = {_base_ac(chain2unp[c]) for c in side if c in chain2unp}
            side_unp.append(u)
            sk_proteins |= u
        if len(side_unp) == 2:
            for a in side_unp[0]:
                for b in side_unp[1]:
                    sk_pairs.add(frozenset((a, b)))

    splits = pd.concat(
        [_read(f"df_intact_mutations_filtered_{s}.parquet") for s in ("train", "valid", "test")],
        ignore_index=True,
    )
    ia_pdb = set(splits["entry_id"].str.upper())
    ia_prot = set(splits["affected_protein_ac"].map(_base_ac)) | set(
        splits["participant_protein"].map(_base_ac))
    ia_pairs = {frozenset((_base_ac(a), _base_ac(p)))
                for a, p in zip(splits["affected_protein_ac"], splits["participant_protein"])}
    row_pairs = [frozenset((_base_ac(a), _base_ac(p)))
                 for a, p in zip(splits["affected_protein_ac"], splits["participant_protein"])]
    rows_on_shared = sum(1 for x in row_pairs if x in sk_pairs)

    def row(level, ia, sk):
        shared = len(ia & sk)
        return {"level": level, "intact": len(ia), "skempi": len(sk),
                "shared": shared, "pct_of_intact": round(100 * shared / len(ia), 1)}

    return {
        "skempi_totals": {"pdb": len(sk_pdb), "proteins": len(sk_proteins),
                          "interface_pairs": len(sk_pairs), "mutations": int(len(sk))},
        "overlap": [
            row("PDB entries", ia_pdb, sk_pdb),
            row("Proteins (UniProt)", ia_prot, sk_proteins),
            row("Interface pairs", ia_pairs, sk_pairs),
        ],
        "rows_on_shared_interface": rows_on_shared,
        "trained_rows": int(len(splits)),
        "rows_on_shared_pct": round(100 * rows_on_shared / len(splits), 1),
    }


# --------------------------------------------------------------------------------------
# formatting / file emission
#   The write-up has exactly TWO numbered tables; the funnel is prose, not a table:
#     table_1_taxonomy       -> main.tex  tab:intact-taxonomy
#     table_2_skempi_overlap -> main.tex  tab:intact-skempi
#     funnel_stats           -> \paragraph{Scale (data funnel)}  (supporting numbers, no table)
#   Each numbered table is emitted as BOTH a .csv (data) and a .tex (paste-ready full table env).
# --------------------------------------------------------------------------------------
def _lint(n: int) -> str:
    """Integer with LaTeX-safe thousands separators, e.g. 77206 -> 77{,}206."""
    return f"{n:,}".replace(",", "{,}")


def _display_name(ft: str) -> str:
    """feature_type string without the trailing '(MI:xxxx)' (the MI code is its own column)."""
    return ft.split("(MI:")[0].strip()


def _latex_label(lbl: str) -> str:
    """Pretty math-mode weak label for LaTeX (CSV keeps the plain form)."""
    return {"+1": "$+1$", "-1": "$-1$", "0": "$0$", "0 (kinetic)": "$0$ (kinetic)"}.get(lbl, lbl)


_AUTOGEN = "% Auto-generated by paper_stats/make_paper_tables.py — regenerate, do not hand-edit.\n"


def write_table_1_taxonomy(tf: dict) -> None:
    tax = tf["taxonomy"]
    with open(os.path.join(OUT, "table_1_taxonomy.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_type", "psi_mi", "count", "weak_label"])
        for t in tax:
            w.writerow([_display_name(t["feature_type"]), t["mi"], t["count"], t["label"]])

    rows = "\n".join(
        f"    {_display_name(t['feature_type'])} & {t['mi']} & {_lint(t['count'])} "
        f"& {_latex_label(t['label'])} \\\\"
        for t in tax
    )
    tex = (
        _AUTOGEN
        + "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{IntAct feature types, PSI-MI codes, counts in the length/WT-filtered corpus "
        f"({_lint(tf['funnel']['filtered'])} rows), and the assigned weak label. ``Strength'' variants "
        "annotate equilibrium affinity; ``rate'' variants annotate kinetics and are treated as neutral; "
        "the generic unspecified type is excluded.}\n"
        "  \\label{tab:intact-taxonomy}\n"
        "  \\begin{tabular}{llrc}\n"
        "    \\toprule\n"
        "    Feature type & PSI-MI & Count & Weak label \\\\\n"
        "    \\midrule\n"
        f"{rows}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    open(os.path.join(OUT, "table_1_taxonomy.tex"), "w").write(tex)


def write_table_2_skempi_overlap(ov: dict) -> None:
    with open(os.path.join(OUT, "table_2_skempi_overlap.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "intact_trained", "skempi", "shared", "pct_of_intact"])
        for o in ov["overlap"]:
            w.writerow([o["level"], o["intact"], o["skempi"], o["shared"], o["pct_of_intact"]])
        w.writerow(["mutation_rows_on_shared_interface", ov["trained_rows"], "",
                    ov["rows_on_shared_interface"], ov["rows_on_shared_pct"]])

    st = ov["skempi_totals"]
    body = "\n".join(
        f"    {o['level']} & {_lint(o['intact'])} & {o['skempi']} & {o['shared']} "
        f"& ${o['pct_of_intact']}\\%$ \\\\"
        for o in ov["overlap"]
    )
    tex = (
        _AUTOGEN
        + "\\begin{table}[h]\n"
        "  \\centering\n"
        "  \\caption{Overlap between the trained IntAct set and SKEMPI~2.0 "
        f"({st['pdb']} PDB entries / {st['interface_pairs']} interface pairs / {st['proteins']} "
        "proteins), after SIFTS PDB$\\rightarrow$UniProt mapping. ``Shared'' is the intersection.}\n"
        "  \\label{tab:intact-skempi}\n"
        "  \\begin{tabular}{lrrrr}\n"
        "    \\toprule\n"
        "    Level & IntAct (trained) & SKEMPI & Shared & \\% of IntAct \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\midrule\n"
        "    Mutation rows on a shared interface & \\multicolumn{4}{r}"
        f"{{${ov['rows_on_shared_interface']} / {_lint(ov['trained_rows'])} "
        f"= {ov['rows_on_shared_pct']}\\%$ (upper bound on repeats)}} \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    open(os.path.join(OUT, "table_2_skempi_overlap.tex"), "w").write(tex)


def write_funnel_stats(tf: dict) -> None:
    with open(os.path.join(OUT, "funnel_stats.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in tf["funnel"].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    w.writerow([f"{k}.{kk}", vv])
            else:
                w.writerow([k, v])


def main():
    os.makedirs(OUT, exist_ok=True)
    tf = intact_taxonomy_and_funnel()
    ov = skempi_overlap()

    json.dump({"intact": tf, "skempi_overlap": ov},
              open(os.path.join(OUT, "stats.json"), "w"), indent=2)
    write_table_1_taxonomy(tf)
    write_table_2_skempi_overlap(ov)
    write_funnel_stats(tf)

    f = tf["funnel"]
    print("=" * 78)
    print(f"TABLE 1  (tab:intact-taxonomy)  IntAct taxonomy, filtered corpus n={f['filtered']:,}")
    print("=" * 78)
    for t in tf["taxonomy"]:
        print(f"  {_display_name(t['feature_type']):<34} {t['mi']:<8} {t['count']:>7,}  -> {t['label']}")
    print("\n" + "=" * 78)
    print("FUNNEL stats (prose in main.tex — NOT a numbered table)")
    print("=" * 78)
    print(f"  raw {f['raw_mutations']:,} -> parsed {f['parsed_mutations']:,} "
          f"-> filtered {f['filtered']:,}")
    print(f"  filtered directional: {f['filtered_directional']:,} "
          f"({f['filtered_pct_positive']}% positive)")
    print(f"  trained rows: {f['trained_rows']:,} | directional anchors: "
          f"{f['trained_directional']:,} ({f['trained_pos']} pos / {f['trained_neg']} neg "
          f"= {f['trained_pct_positive']}% positive)")
    print(f"  trained: {f['trained_assemblies']} assemblies | {f['trained_pdb_entries']} PDB "
          f"| {f['trained_interface_pairs']} interface pairs "
          f"(+{f['trained_neutral_no_effect']} neutral, {f['trained_rate']} rate, "
          f"{f['trained_generic_ignored']} generic ignored)")
    print("\n" + "=" * 78)
    print("TABLE 2  (tab:intact-skempi)  IntAct(trained) vs SKEMPI 2.0 overlap")
    print("=" * 78)
    st = ov["skempi_totals"]
    print(f"  SKEMPI v2: {st['pdb']} PDB | {st['proteins']} proteins | "
          f"{st['interface_pairs']} interface pairs | {st['mutations']:,} mutations")
    for o in ov["overlap"]:
        print(f"  {o['level']:<20} IntAct {o['intact']:>6,} | SKEMPI {o['skempi']:>4} "
              f"| shared {o['shared']:>3} ({o['pct_of_intact']}% of IntAct)")
    print(f"  rows on a shared interface: {ov['rows_on_shared_interface']} / {ov['trained_rows']:,} "
          f"= {ov['rows_on_shared_pct']}% (upper bound on repeats)")

    print("\nwrote to", OUT + "/ :")
    for fn in ("table_1_taxonomy.csv", "table_1_taxonomy.tex",
               "table_2_skempi_overlap.csv", "table_2_skempi_overlap.tex",
               "funnel_stats.csv", "stats.json"):
        print("  -", fn)


if __name__ == "__main__":
    main()
