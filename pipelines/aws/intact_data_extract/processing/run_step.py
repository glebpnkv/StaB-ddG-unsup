"""SageMaker Processing entry point for the IntAct data-extraction pipeline.

One script, dispatched by ``--step``, reused across every ProcessingStep (mirrors the GCP
``@dsl.component`` functions). It adapts the existing ``stabddg.jobs.intact_data_extract`` functions
to SageMaker's container I/O convention:

  * ProcessingInputs  are mounted under  /opt/ml/processing/input/<channel>/
  * ProcessingOutputs are read back from /opt/ml/processing/output/<channel>/  (uploaded to S3)

Each channel directory holds the named parquet/json file or the structures sub-tree, so steps chain
purely through S3 with no bespoke staging logic.
"""
import argparse
import json
import os

import pandas as pd

INPUT_ROOT = "/opt/ml/processing/input"
OUTPUT_ROOT = "/opt/ml/processing/output"


def _in(channel: str, name: str) -> str:
    return os.path.join(INPUT_ROOT, channel, name)


def _out_dir(channel: str) -> str:
    path = os.path.join(OUTPUT_ROOT, channel)
    os.makedirs(path, exist_ok=True)
    return path


def _out(channel: str, name: str) -> str:
    return os.path.join(_out_dir(channel), name)


def step_prepare_mutations(args):
    from stabddg.jobs.intact_data_extract import prepare_mutations

    work_dir = "/opt/ml/processing/work"
    os.makedirs(os.path.join(work_dir, "intact"), exist_ok=True)
    df = prepare_mutations(data_dir=work_dir)
    if args.intact_sample_size < 1.0:
        df = df.sample(frac=args.intact_sample_size, random_state=42)
    print(f"prepare_mutations: {len(df)} mutations")
    df.to_parquet(_out("mutations", "df_intact_mutations.parquet"))


def step_fetch_alphafold(args):
    from stabddg.jobs.intact_data_extract import fetch_alphafold_for_uniprots

    df = pd.read_parquet(_in("mutations", "df_intact_mutations.parquet"))
    outcome = fetch_alphafold_for_uniprots(
        df_mutations=df, proteins_dir=_out_dir("proteins"), max_workers=args.max_workers
    )
    print(f"fetch_alphafold: {len(outcome)} UniProts")
    with open(_out("step_outcome", "step_outcome.json"), "w") as f:
        json.dump(outcome, f)


def step_filter_mutations(args):
    from stabddg.jobs.intact_data_extract import filter_mutations_by_available_structures

    df = pd.read_parquet(_in("mutations", "df_intact_mutations.parquet"))
    with open(_in("step_outcome", "step_outcome.json")) as f:
        outcome = json.load(f)
    out = filter_mutations_by_available_structures(df_mutations=df, step_outcome=outcome)
    print(f"filter_mutations: {len(out)} rows kept")
    out.to_parquet(_out("filtered", "df_intact_mutations_filtered.parquet"))


def step_fetch_metadata(args):
    from stabddg.jobs.intact_data_extract import fetch_assemblies_metadata, get_uniprot_pairs

    df = pd.read_parquet(_in("filtered", "df_intact_mutations_filtered.parquet"))
    pairs = get_uniprot_pairs(df)
    raw = fetch_assemblies_metadata(uniprots_pairs=pairs, max_workers=args.max_workers)
    raw = raw.drop(columns=["mutations"], errors="ignore")
    print(f"fetch_metadata: {len(raw)} candidate assemblies")
    raw.to_parquet(_out("assemblies_raw", "df_assemblies_raw.parquet"))


def step_select_assemblies(args):
    from stabddg.jobs.intact_data_extract import select_and_normalize_assemblies

    raw = pd.read_parquet(_in("assemblies_raw", "df_assemblies_raw.parquet"))
    out = select_and_normalize_assemblies(raw)
    print(f"select_assemblies: {len(out)} selected")
    out.to_parquet(_out("assemblies", "df_assemblies.parquet"))


def step_fetch_atoms(args):
    from stabddg.jobs.intact_data_extract import fetch_and_summarize_assemblies_atoms

    df = pd.read_parquet(_in("assemblies", "df_assemblies.parquet"))
    filtered = fetch_and_summarize_assemblies_atoms(
        df_assemblies=df, assemblies_dir=_out_dir("assemblies_atoms"), max_workers=args.max_workers
    )
    print(f"fetch_atoms: {len(filtered)} assemblies with structures")
    filtered.to_parquet(_out("assemblies_filtered", "df_assemblies_filtered.parquet"))


STEPS = {
    "prepare_mutations": step_prepare_mutations,
    "fetch_alphafold": step_fetch_alphafold,
    "filter_mutations": step_filter_mutations,
    "fetch_metadata": step_fetch_metadata,
    "select_assemblies": step_select_assemblies,
    "fetch_atoms": step_fetch_atoms,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=sorted(STEPS))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--intact-sample-size", type=float, default=1.0)
    args = parser.parse_args()

    print(f"=== running step: {args.step} ===")
    STEPS[args.step](args)
    print(f"=== step {args.step} complete ===")


if __name__ == "__main__":
    main()
