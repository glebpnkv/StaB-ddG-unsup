# paper_stats — reproducible tables & statistics for the write-up

Everything reported in `StaB-ddG-unsup-writeup/main.tex` is regenerated here from the data on disk,
in one command, so the paper can never silently drift from the code or the data.

```bash
# from the repo root
.venv/bin/python paper_stats/make_paper_tables.py
```

This prints the tables and writes:

- `output/stats.json` — every number, machine-readable
- `output/tables.tex` — LaTeX table bodies matching `main.tex`

## What maps to what

| Script output | Write-up object | Content |
|---|---|---|
| **T1** | `tab:intact-taxonomy` | IntAct feature types, PSI-MI codes, counts in the filtered corpus, weak label |
| **T2** | `\paragraph{Scale (data funnel)}` | raw → parsed → filtered → trained funnel; class imbalance |
| **T3** | `tab:intact-skempi` | IntAct(trained) vs SKEMPI 2.0 overlap at PDB / protein / interface-pair level |

## Single source of truth

The `feature_type → weak-label` mapping (pos / neg / rate / no-effect / excluded) is **imported from
`stabddg.intact.dataset.IntactDataset`**, not re-declared here. If the training-time classification
changes (e.g. the rate-type fix), the taxonomy table changes with it automatically.

## Inputs

- `data/intact/df_intact_mutations_{raw,,_filtered}.parquet`, the `_filtered_{train,valid,test}` splits,
  and `df_assemblies_filtered.parquet`
- `data/SKEMPI/skempi_v2.csv`
- `paper_stats/sifts_pdb_uniprot.json` — SKEMPI PDB→UniProt mappings via the PDBe SIFTS API. Committed
  for deterministic, offline reproduction; **auto-fetched** for any PDB not in the cache (needs network
  only then). Regenerate from scratch by deleting the file and re-running.

## Adding a table

Add a function that returns a dict of numbers, wire it into `main()` and `to_latex()`, and document the
mapping row above. Keep every reported number sourced from a function here — no hand-computed values in
the paper.
