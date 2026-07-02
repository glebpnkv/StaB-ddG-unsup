import logging
import operator as op
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce

import gemmi
import numpy as np
import pandas as pd
import requests
import torch
from Bio import Align
from rcsbapi.data import DataQuery
from rcsbapi.model import ModelQuery
from rcsbapi.search import AttributeQuery, NestedAttributeQuery
from requests.adapters import HTTPAdapter
from safetensors.torch import save_file
from urllib3.util.retry import Retry

from stabddg.constants import (
    AA3_TO_1,
    ALPHABET,
    COORDS_ORDER,
    SEQUENCE_DELETION,
    SEQUENCE_UNKNOWN,
)
from stabddg.intact.uniprot import fetch_uniprot_sequences
from stabddg.utils.progress import track

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silence chatty HTTP clients used under the hood by rcsbapi (httpx/urllib3)
# without affecting our own logger.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# Per-thread requests session
_thread_local = threading.local()


def _build_retrying_session() -> requests.Session:
    s = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    s.headers.update(
        {"User-Agent": "stabddg/parallel-fetch (https://alphafold.ebi.ac.uk/)"}
    )
    return s


def _get_thread_session() -> requests.Session:
    ses = getattr(_thread_local, "session", None)
    if ses is None:
        ses = _build_retrying_session()
        _thread_local.session = ses
    return ses


# ---------------------------------------------------------------------------
# Low-level parsers — turn structure text into (N,3) CA coordinates + annotations
# ---------------------------------------------------------------------------


def _read_structure_from_text(text: str, fmt: str) -> gemmi.Structure:
    """Create a gemmi.Structure from in-memory text.
    fmt: one of {"cif", "pdb"} to choose the appropriate reader.
    """
    if fmt == "cif":
        # Parse an mmCIF document from string and convert to Structure
        doc = gemmi.cif.read_string(text)
        return gemmi.make_structure_from_block(doc.sole_block())
    elif fmt == "pdb":
        # Parse legacy PDB text into Structure
        return gemmi.read_pdb_string(text)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _structure_to_atom_dataframe(
    st: gemmi.Structure,
    *,
    chains: set[str] | None = None,
    atoms: set[str] | None = None,
    include_het: bool = False,
    prefer_label_seq: bool = True,
) -> pd.DataFrame:
    """
    Convert a gemmi.Structure into a tidy DataFrame of atoms.

    Parameters
    ----------
    chains : optional set of chain IDs to keep (e.g., {"A", "B"})
    atoms  : optional set of atom names to keep (e.g., {"N","CA","C","O"})
    include_het : include HETATM residues (ligands, waters, etc.)
    prefer_label_seq : if True, use residue.label_seq when available for resnum_label

    Returns
    -------
    pd.DataFrame with columns:
        model (int), chain (str), res_name_3 (str), res_name_1 (str),
        resnum_label (int|None), resnum_auth (int|None), ins_code (str|None),
        atom_name (str), element (str), altloc (str|None),
        occupancy (float), b_factor (float),
        x (float), y (float), z (float), is_het (bool)
    """
    # Make sure label_seq is present (safe no-op if already set)
    st.setup_entities()
    # Fill Residue.label_seq when SEQRES is known
    st.assign_label_seq_id()

    rows = []
    for model_idx, model in enumerate(st):  # gemmi.Model
        for chain in model:  # gemmi.Chain
            chain_id = chain.name
            if chains is not None and chain_id not in chains:
                continue
            for res in chain:  # gemmi.Residue
                is_het = bool(res.het_flag == "H")
                if is_het and not include_het:
                    continue

                aa3 = res.name.upper()
                aa1 = AA3_TO_1.get(aa3, "X")

                # Author residue number (PDB numbering)
                auth_num = res.seqid.num if hasattr(res, "seqid") else None
                ins_code = res.seqid.icode if hasattr(res, "seqid") else ""
                label_seq = int(getattr(res, "label_seq", 0) or 0)
                resnum_label = (
                    int(label_seq) if (prefer_label_seq and label_seq != 0) else None
                )

                for at in (
                    res.first_conformer()
                ):  # gemmi.Atom - picks a single altloc per atom group
                    atom_name = at.name.strip()
                    if atoms is not None and atom_name not in atoms:
                        continue
                    rows.append(
                        {
                            "model": model_idx,
                            "chain": chain_id,
                            "res_name_3": aa3,
                            "res_name_1": aa1,
                            "resnum_label": resnum_label,
                            "resnum_auth": auth_num,
                            "ins_code": ins_code,
                            "atom_name": atom_name,
                            "element": at.element.name,
                            "altloc": (at.altloc if at.has_altloc() else ""),
                            "occupancy": float(at.occ),
                            "b_factor": float(at.b_iso),
                            "x": float(at.pos.x),
                            "y": float(at.pos.y),
                            "z": float(at.pos.z),
                            "is_het": is_het,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1) AlphaFold DB — wild type by UniProt ID -> (N,3) + annotations.
#    Prefers mmCIF when available.
# ---------------------------------------------------------------------------


def _fetch_alphafold_structure(
    uniprot_acc: str,
    session: requests.Session | None = None,
    prefer_format: str = "cif",  # "cif" or "pdb"
) -> tuple[gemmi.Structure, str, str]:
    """
    Fetch AlphaFold model and parse into gemmi.Structure.
    prefer_format selects which URL to choose when both are present.
    Returns (structure, source_url, fmt)
    """
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_acc}"
    ses = session or _get_thread_session()

    r = ses.get(api, timeout=60)
    r.raise_for_status()
    meta = r.json()
    if not meta:
        raise ValueError(f"No AlphaFold entry for {uniprot_acc}")

    m = meta[0]
    cif_url = m.get("cifUrl")
    pdb_url = m.get("pdbUrl")

    url = None
    fmt = None
    if prefer_format.lower() == "pdb" and pdb_url:
        url, fmt = pdb_url, "pdb"
    elif prefer_format.lower() == "cif" and cif_url:
        url, fmt = cif_url, "cif"
    else:
        # Fallback to whichever exists
        url = cif_url or pdb_url
        if not url:
            raise ValueError(f"No structure URL for {uniprot_acc}")
        fmt = "cif" if url.lower().endswith(".cif") else "pdb"

    text = ses.get(url, timeout=120).text
    st = _read_structure_from_text(text, fmt)
    return st, url, fmt


# ---------------------------------------------------------------------------
# Atom dataframe helper (PDB or CIF) for AlphaFold entries
# ---------------------------------------------------------------------------


def fetch_alphafold_atoms_df(
    uniprot_or_pro: str,
    *,
    prefer_format: str = "pdb",  # "pdb" if you want strict compatibility with PDB-centric tooling
    chains: set[str] | None = None,
    atoms: set[str] | None = None,  # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,  # include waters/ligands if True
    prefer_label_seq: bool = True,  # mmCIF label_seq where present
) -> dict:
    """
    Return a dict with:
      - status: "success" or error message
      - df: pandas DataFrame of atoms (if success)
      - source_url: the URL used
      - format: "pdb" or "cif"
    """
    m = re.match(r"^(?P<acc>[A-Z0-9]+(?:-\d+)?)-(?P<pro>PRO_\d+)$", uniprot_or_pro)
    if m:
        return {"status": "unresolved"}  # mirror fetch_alphafold_xyz behavior

    try:
        ses = _get_thread_session()
        st, url, fmt = _fetch_alphafold_structure(
            uniprot_or_pro, session=ses, prefer_format=prefer_format
        )
        df = _structure_to_atom_dataframe(
            st,
            chains=chains,
            atoms=atoms,
            include_het=include_het,
            prefer_label_seq=prefer_label_seq,
        )
        # Getting the representative model if available
        df = df.loc[df["model"] == 0].copy()
        df = df.reset_index(drop=True)

        df["item_id"] = uniprot_or_pro  # Adding uniprot_or_pro as an ID to DataFrame
        df["abs_pos_label"] = df["resnum_label"]

        entry_id = uniprot_or_pro
        chain_map = {"A": uniprot_or_pro}
        metadata = chain_map | {
            "entry_id": entry_id,
        }

        return {
            "status": "success",
            "df": df,
            "source_url": url,
            "format": fmt,
            "metadata": metadata,
        }
    except Exception as e:
        return {"status": str(e)}


def fetch_alphafold_atoms_parallel(
    uniprot_codes: list[str],
    prefer_format: str = "pdb",  # "pdb" if you want strict compatibility with PDB-centric tooling
    chains: set[str] | None = None,
    atoms: set[str] | None = None,  # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,  # include waters/ligands if True
    prefer_label_seq: bool = True,  # mmCIF label_seq where present
    max_workers: int = 8,
    show_progress: bool = True,
    parquet_dir: str | None = None,
    safetensors_dir: str | None = None,
):
    if parquet_dir is not None:
        os.makedirs(parquet_dir, exist_ok=True)
    results: dict[str, dict] = {}

    def _task(code: str) -> tuple[str, dict]:
        try:
            out = fetch_alphafold_atoms_df(
                code,
                prefer_format=prefer_format,
                chains=chains,
                atoms=atoms,
                include_het=include_het,
                prefer_label_seq=prefer_label_seq,
            )

            # Do nothing further if the download did not succeed
            if out["status"] != "success":
                return code, out

            if parquet_dir is not None:
                parquet_path = os.path.join(parquet_dir, f"{code}.parquet")
                out["df"].to_parquet(parquet_path, index=False)
                out["parquet_path"] = parquet_path
            if safetensors_dir is not None:
                st_path = _write_assemblies_safetensors(
                    out["df"], safetensors_dir, code, metadata=out["metadata"]
                )
                out["safetensors_path"] = st_path
            # include df only if not saving to reduce memory
            if parquet_dir is not None and safetensors_dir is not None:
                del out["df"]
            return code, out
        except Exception as e:
            return code, {"status": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, code): code for code in uniprot_codes}
        iterator = as_completed(futs)
        if show_progress:
            iterator = track(iterator, total=len(futs), desc="AlphaFold fetch")
        for fut in iterator:
            code, out = fut.result()
            results[code] = out

    return results


# ---------------------------------------------------------------------------
# 2) Assemblies helpers
# ---------------------------------------------------------------------------


def _search_assemblies_for_uniprots(
    uniprots: list[str], protein_only: bool = True
) -> list[str]:
    """
    Search for assemblies that contain all requested UniProt accessions (ignoring multiplicities here),
    plus assembly-level constraints. Multiplicity & exact-set are enforced later.
    """
    out: list[str] = []
    if not uniprots:
        return out

    want = Counter(uniprots)
    uniq = list(want.keys())

    groups = []
    for up in uniq:
        q_acc = AttributeQuery(
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            operator="exact_match",
            value=up,
        )
        q_db = AttributeQuery(
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            operator="exact_match",
            value="UniProt",
        )
        groups.append(NestedAttributeQuery(q_acc, q_db))

    # Assembly-level filters
    at_least_two_protein_chains = AttributeQuery(
        "rcsb_assembly_info.polymer_entity_instance_count_protein",
        operator="greater_or_equal",
        value=2,
    )

    if len(uniq) == 1:
        # true homomer → exactly one distinct polymer entity
        entity_count_q = AttributeQuery(
            "rcsb_assembly_info.polymer_entity_count", operator="equals", value=1
        )
    else:
        # heteromer → exactly as many distinct polymer entities as requested uniques
        entity_count_q = AttributeQuery(
            "rcsb_assembly_info.polymer_entity_count",
            operator="equals",
            value=len(uniq),
        )

    extra = [at_least_two_protein_chains, entity_count_q]
    if protein_only:
        extra.append(
            AttributeQuery(
                "rcsb_entry_info.selected_polymer_entity_types",
                operator="exact_match",
                value="Protein (only)",
            )
        )

    query = reduce(op.and_, groups + extra)
    try:
        out = list(query(return_type="assembly"))
    except Exception as e:
        logger.error(f"Error searching for assemblies for UniProt accessions: {e}")
    return out


def _entries_meta_and_refs(pdb_ids):
    """
    Fetch, in one shot, for each entry:
      - experimental method
      - resolution (X-ray) or EM reconstruction resolutions
      - set of UniProt accessions present in polymer entities
    Returns: dict[pdb_id] -> {"method": str|None, "best_resolution": float|None, "uniprots": set[str]}
    """
    q = DataQuery(
        input_type="entries",
        input_ids=list(pdb_ids),
        return_data_list=[
            "entries.rcsb_id",
            "exptl.method",
            "rcsb_entry_info.resolution_combined",
            "em_3d_reconstruction.resolution",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        ],
    )
    data = q.exec()  # dict with data.entries [...]
    out = {}
    for e in data["data"]["entries"]:
        pdb_id = e["rcsb_id"]
        # method
        method = None
        if e.get("exptl"):
            method = (e["exptl"][0] or {}).get("method")

        # resolution
        res = None
        rc = (e.get("rcsb_entry_info") or {}).get("resolution_combined")
        if rc:
            res = min([v for v in rc if isinstance(v, (int, float))], default=None)
        if res is None:
            em = e.get("em_3d_reconstruction") or []
            em_res = [
                d.get("resolution") for d in em if d.get("resolution") is not None
            ]
            if em_res:
                res = min(em_res)

        # UniProt set present in entry polymer entities
        ups = set()
        for pe in e.get("polymer_entities") or []:
            cont = pe.get("rcsb_polymer_entity_container_identifiers") or {}
            for ref in cont.get("reference_sequence_identifiers") or []:
                if ref.get("database_name") == "UniProt" and ref.get(
                    "database_accession"
                ):
                    ups.add(ref["database_accession"])
        out[pdb_id] = {"method": method, "best_resolution": res, "uniprots": ups}
    return out


def _fetch_chain_to_uniprot_map_rcsb(pdb_id: str) -> dict[str, str]:
    """
    Map author chain IDs (auth_asym_id) to UniProt accession via RCSB Data API.
    """
    q = DataQuery(
        input_type="entries",
        input_ids=[pdb_id.upper()],
        return_data_list=[
            # per-entity metadata
            "polymer_entities.rcsb_polymer_entity_container_identifiers.entity_id",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.auth_asym_ids",
            # cross-refs – filter to UniProt below
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            "polymer_entities.rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        ],
    )
    resp = q.exec()
    mapping: dict[str, str] = {}

    entries = (resp.get("data", {}) or {}).get("entries") or []
    if not entries:
        return mapping

    for ent in entries[0].get("polymer_entities", []) or []:
        cont = ent.get("rcsb_polymer_entity_container_identifiers") or {}
        refs = cont.get("reference_sequence_identifiers") or []
        # take the first UniProt xref if present
        unp = next(
            (
                r.get("database_accession")
                for r in refs
                if (r.get("database_name") or "").lower() in ("uniprot", "unp")
            ),
            "",
        )
        if not unp:
            continue

        chains = cont.get("auth_asym_ids")
        if isinstance(chains, list):
            chain_list = chains
        elif isinstance(chains, str):
            chain_list = [c.strip() for c in chains.replace(",", " ").split()]
        else:
            chain_list = []

        for ch in chain_list:
            if ch:
                mapping.setdefault(ch, unp)

    return mapping


def _pdbe_mutations(pdb_id):
    """
    PDBe Graph API mutated/modified residues. Empty dict means none reported.
    """
    muts = {}
    for ep in ("mutated_residues", "modified_residues"):
        u = f"https://www.ebi.ac.uk/pdbe/api/v2/pdb/{ep}/{pdb_id}"
        try:
            r = requests.get(u, timeout=30)
            if r.ok:
                muts.update((r.json() or {}).get(pdb_id, {}))
        except Exception:
            pass
    return muts


def _assemblies_uniprot_instance_counts(
    assembly_ids: list[str],
) -> dict[str, dict[str, int]]:
    """
    For each assembly (e.g. '4HHB-1'), count how many polymer_entity_instances
    map to each UniProt accession.
    Returns: { '4HHB-1': {'P69905':2, 'P68871':2}, ... }
    """
    if not assembly_ids:
        return {}

    q = DataQuery(
        input_type="assemblies",
        input_ids=list(assembly_ids),
        return_data_list=[
            "assemblies.rcsb_id",  # compound id 'PDBID-assemblyId'
            "polymer_entity_instances.polymer_entity."
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            "polymer_entity_instances.polymer_entity."
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
        ],
    )
    data = q.exec()["data"]["assemblies"]
    out = {}
    for asm in data:
        asm_id = asm["rcsb_id"]
        counts = defaultdict(int)
        for inst in asm.get("polymer_entity_instances") or []:
            poly = inst.get("polymer_entity") or {}
            cont = poly.get("rcsb_polymer_entity_container_identifiers") or {}
            for ref in cont.get("reference_sequence_identifiers") or []:
                if ref.get("database_name") == "UniProt" and ref.get(
                    "database_accession"
                ):
                    counts[ref["database_accession"]] += 1
                    break  # count each instance once
        out[asm_id] = dict(counts)
    return out


def _filter_assemblies_by_uniprot_counts(
    assembly_ids: list[str], uniprots: list[str]
) -> list[str]:
    """
    Keep assemblies iff:
      (a) the UniProt ID **set** present in the assembly is **exactly** the requested set, and
      (b) each requested UniProt appears with **≥** the requested multiplicity.
    Examples:
      - ["A","B"] → present set must be {A,B} (no extras like peptides).
      - ["P04626","P04626"] → present set {P04626} and count ≥ 2.
    """
    want = Counter(uniprots)
    want_set = set(want.keys())
    counts = _assemblies_uniprot_instance_counts(
        assembly_ids
    )  # {aid: {up: n_instances}}
    keep: list[str] = []
    for aid, c in counts.items():
        present_set = set(c.keys())
        if present_set == want_set and all(
            c.get(up, 0) >= need for up, need in want.items()
        ):
            keep.append(aid)
    return keep


def rank_assemblies(assembly_ids, uniprots):
    """
    Score & sort assemblies:
      + require both UniProts present (heavy penalty if not)
      + prefer no engineered mutations (PDBe), then better resolution,
      + prefer X-ray over EM (small bonus).
    Returns sorted list of dicts.
    """
    uniprots = set(uniprots)

    # batch metadata
    pdb_ids = {aid.split("-")[0] for aid in assembly_ids}
    meta = _entries_meta_and_refs(pdb_ids)  # via rcsbapi.data
    ranked = []
    for aid in assembly_ids:
        pdb_id, asm = aid.split("-")
        m = meta.get(pdb_id, {})
        method = (m.get("method") or "").upper()
        res = m.get("best_resolution")
        has_both = m.get("uniprots", set()).issuperset(uniprots)

        muts = _pdbe_mutations(pdb_id)  # PDBe Graph API
        mut_penalty = 1 if muts else 0
        method_bonus = (
            0 if "X-RAY" in method else (0.25 if "ELECTRON" in method else 0.5)
        )
        res_use = res if isinstance(res, (int, float)) else 9.99

        score = (0 if has_both else 10) + mut_penalty + method_bonus + (res_use / 10.0)
        ranked.append(
            {
                "biological_assembly": aid,
                "score": score,
                "method": m.get("method"),
                "resolution": res,
                "has_both": has_both,
                "mutations": muts,
            }
        )
    return sorted(ranked, key=lambda x: x["score"])


def fetch_assemblies_for_uniprots(
    uniprots: list[str],
) -> list[dict[str, str | float | bool]] | None:
    # Getting assemblies for UniProt accessions
    assembly_ids = _search_assemblies_for_uniprots(uniprots)

    # Filter assemblies by UniProt counts
    assembly_ids = _filter_assemblies_by_uniprot_counts(assembly_ids, uniprots)

    if len(assembly_ids) < 1:
        return None

    ranked = rank_assemblies(assembly_ids, uniprots)
    return ranked


def fetch_assemblies_for_uniprots_parallel(
    uniprots_pairs: list[list[str]],
    max_workers: int = 8,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run fetch_assemblies_for_uniprots in parallel over a list of UniProt pairs.
    Returns a DataFrame with:
      - pair_idx: absolute enumerate index of the input pair
      - uniprot_1, uniprot_2: the pair values as columns
      - plus all fields returned by rank_assemblies for each assembly (or NaNs if None)
    """

    def _task(idx: int, pair: list[str]) -> tuple[int, list[str], list[dict] | None]:
        try:
            out = fetch_assemblies_for_uniprots(pair)
        except Exception as e:
            logger.error(
                f"Error fetching assemblies for UniProt accessions {pair}: {e}"
            )
            out = None
        # Ensure pair length 2 for consistent columns
        a = pair[0] if len(pair) > 0 else None
        b = pair[1] if len(pair) > 1 else None
        return idx, [a, b], out

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, i, pair): i for i, pair in enumerate(uniprots_pairs)}
        iterator = as_completed(futs)
        if show_progress:
            iterator = track(iterator, total=len(futs), desc="Assemblies fetch")
        for fut in iterator:
            idx, (u1, u2), ranked = fut.result()
            if ranked is None or len(ranked) == 0:
                # One row with NaNs for assembly fields
                rows.append(
                    {
                        "pair_idx": idx,
                        "participant_protein": u1,
                        "affected_protein_ac": u2,
                        "assembly_id": None,
                        "score": None,
                        "method": None,
                        "resolution": None,
                        "has_both": None,
                        "mutations": None,
                    }
                )
            else:
                for r in ranked:
                    rows.append(
                        {
                            "pair_idx": idx,
                            "participant_protein": u1,
                            "affected_protein_ac": u2,
                            **r,
                        }
                    )

    # Preparing output DataFrame
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(by=["pair_idx", "score"], ignore_index=True)

    return df_out


def normalize_entry(s: str) -> str:
    # Accept stuff like '8CT8', '8CT8-1', '8CT8_2', '8ct8'
    m = re.match(r"(?i)^([0-9a-z]{4})", s.strip())
    if not m:
        raise ValueError(f"Unrecognized ID: {s}")
    return m.group(1).upper()


def download_assembly_cif(assembly_key: str) -> str:
    """
    Use Model Server via rcsbapi to download the biological assembly mmCIF.
    assembly_id is e.g. '4HHB-1' -> (entry='4HHB', name='1')
    """
    pdb_id, asm = assembly_key.split("-")
    mq = ModelQuery()
    # Model Server assembly endpoint; encoding 'cif' yields mmCIF
    out = mq.get_assembly(
        entry_id=pdb_id,
        name=asm,
        encoding="cif",
    )
    return out


def _build_uniprot_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    # Gaps must be cheap: an observed structure chain is the canonical sequence with residues *missing*
    # (unresolved loops/termini), so skipping a stretch of canonical should cost far less than forcing
    # mismatches. Harsh gap penalties make the aligner shift-and-mismatch instead of gapping, which
    # mis-registers everything downstream of a missing loop.
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.1
    # The observed chain is usually a sub-fragment of the full canonical sequence, so the ends of the
    # canonical (target) and observed (query) sequences may gap freely without penalty.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    return aligner


def assign_uniprot_positions_by_alignment(
    observed_seq: str,
    canonical_seq: str,
    aligner: Align.PairwiseAligner | None = None,
) -> np.ndarray:
    """Assign each residue of ``observed_seq`` its 1-based position in ``canonical_seq`` via global
    sequence alignment.

    Returns an int array of length ``len(observed_seq)``; residues that do not align to any canonical
    position (insertions / engineered residues absent from the reference) are -1.

    This is the robust replacement for SIFTS label-number arithmetic (``add_uniprot_from_sifts``),
    which silently misaligns when applied to biological assemblies: the SIFTS bounds describe the
    deposited entry, but gemmi re-assigns ``label_seq`` on the assembly, so the offset is wrong by a
    different amount per structure. Aligning the observed sequence directly to the canonical UniProt
    sequence sidesteps all numbering systems and also tolerates isoform gaps.
    """
    n = len(observed_seq)
    positions = np.full(n, -1, dtype=np.int64)
    if n == 0 or not canonical_seq:
        return positions

    aligner = aligner or _build_uniprot_aligner()
    # target = canonical (reference), query = observed (structure chain)
    alignment = aligner.align(canonical_seq, observed_seq)[0]
    target_blocks, query_blocks = alignment.aligned  # gap-free aligned blocks
    for (t0, _t1), (q0, q1) in zip(target_blocks, query_blocks):
        length = q1 - q0
        # within a gap-free block, target and query advance together
        positions[q0:q1] = np.arange(t0, t0 + length, dtype=np.int64) + 1  # 1-based UniProt
    return positions


def add_uniprot_by_alignment(
    df_atoms: pd.DataFrame,
    chain_to_acc: dict[str, str],
    canonical_by_acc: dict[str, str | None],
    aligner: Align.PairwiseAligner | None = None,
) -> pd.DataFrame:
    """Assign ``resnum_uniprot`` to every atom by aligning each chain's observed residue sequence to
    the canonical UniProt sequence of the protein mapped to that chain.

    This is the robust replacement for ``add_uniprot_from_sifts``: it relies only on the residue
    *sequence* (not on label/auth numbering that biological assemblies re-write), so it is immune to
    assembly renumbering and naturally absorbs expression tags / cloning artefacts at the termini as
    unaligned residues. Residues with no canonical counterpart get ``NaN``.
    """
    out = df_atoms.copy()
    out["resnum_uniprot"] = np.nan
    if out.empty:
        return out

    aligner = aligner or _build_uniprot_aligner()
    for chain, sub in out.groupby("chain", sort=False):
        acc = chain_to_acc.get(chain)
        canonical = canonical_by_acc.get(acc) if acc else None
        if not canonical:
            continue
        # Order residues N->C. Prefer label numbering; fall back to author numbering.
        order_col = "resnum_label" if sub["resnum_label"].notna().any() else "resnum_auth"
        res = sub.dropna(subset=[order_col]).drop_duplicates(subset=[order_col]).sort_values(order_col)
        observed = "".join(res["res_name_1"].tolist())
        positions = assign_uniprot_positions_by_alignment(observed, canonical, aligner=aligner)
        pos_by_resnum = {
            rn: (int(p) if p > 0 else np.nan)
            for rn, p in zip(res[order_col].tolist(), positions)
        }
        out.loc[sub.index, "resnum_uniprot"] = sub[order_col].map(pos_by_resnum)

    return out


def _write_assemblies_safetensors(
    df_assemblies: pd.DataFrame,
    structures_dir: str,
    file_name: str,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Creates a SafeTensors file for residue and atom data following a specific structure.

    This function processes a given pandas DataFrame containing atomic and residue-level
    information, filters it, organizes data into grouped structures, and saves the
    resulting data into a SafeTensors file. It includes additional residue and chain
    encodings, masking information, and sequence transformations. It ensures that only
    valid amino acids in the provided dataset are converted and uses predefined
    global settings for atom ordering (COORDS_ORDER), sequence alphabets (ALPHABET), and
    unknown residue handling (SEQUENCE_UNKNOWN).

    Parameters:
        df_assemblies (pd.DataFrame): DataFrame containing amino acid residue and atom-level
                                      data to process and encode into tensors.
        structures_dir (str): Directory path where the resulting SafeTensors file will be stored.
        file_name (str): Name of the output file (excluding extension).
        metadata (dict): Additional metadata to be included in the safetensors file.

    Returns:
        str: The full path to the created SafeTensors file.
    """
    # Keep PyTorch single-threaded to avoid CPU oversubscription
    torch.set_num_threads(1)
    os.makedirs(structures_dir, exist_ok=True)

    df_filt = df_assemblies.loc[df_assemblies["atom_name"].isin(COORDS_ORDER)].copy()

    # always define chain_to_encoding
    chain_to_encoding: dict[str, str] = {}

    # Build per-atom tensors
    if df_filt.empty:
        cur_tensor = {
            "X": torch.empty((0, 3), dtype=torch.float32),
            "S": torch.empty((0,), dtype=torch.int64),
            "resnums": torch.empty((0,), dtype=torch.int32),
            "chain_encoding_all": torch.empty((0,), dtype=torch.int32),
            "mask": torch.empty((0,), dtype=torch.float32),
            "residue_idx": torch.empty((0,), dtype=torch.int32),
        }
    else:
        df_filt["abs_pos_label"] = df_filt["abs_pos_label"].fillna(-1).astype(int)
        df_filt_res = df_filt.groupby(
            ["chain", "resnum_label", "res_name_1", "abs_pos_label"], as_index=False
        ).size()
        df_filt_res = df_filt_res.drop(columns=["size"])

        # Atom coordinates grouped by atom_name after filtering to COORDS_ORDER
        x_grouped_dict = (
            df_filt.sort_values(by=["chain", "resnum_label", "atom_name"])
            .groupby("atom_name")[["x", "y", "z"]]
            .apply(lambda x: x.values)
            .to_dict()
        )
        x_grouped = np.stack(
            [x_grouped_dict[a] for a in COORDS_ORDER if a in x_grouped_dict], axis=-2
        )

        # Positions of residues in the chain
        resnums = df_filt_res["abs_pos_label"].values

        # Amino acid sequences expressed as integers
        resnames = df_filt_res["res_name_1"]
        seq_list = [ch if ch in ALPHABET else SEQUENCE_UNKNOWN for ch in resnames]
        seq_list = np.asarray([ALPHABET.index(ch) for ch in seq_list], dtype=np.int32)

        # IDs of chains
        chain_encoding_all = (
            df_filt_res["chain"] != df_filt_res["chain"].shift().bfill()
        ).cumsum()
        chain_encoding_all = chain_encoding_all.values

        # Explicit chain -> encoding map (encoding matches 'chain_encoding_all' tensor values)
        # The first contiguous run of a chain is assigned 1, next chain 2, etc.
        first_occ = df_filt_res["chain"].ne(df_filt_res["chain"].shift()).to_numpy()
        chain_order = df_filt_res["chain"].to_numpy()[first_occ]
        chain_to_encoding = {
            f"chain_encoding:{int(i + 1)}": str(ch) for i, ch in enumerate(chain_order)
        }

        # Mask: having ones everywhere means predictions are required for every amino acid
        mask = torch.ones(len(chain_encoding_all))

        residue_idx = torch.arange(len(chain_encoding_all)) + 100 * torch.from_numpy(
            chain_encoding_all
        )

        cur_tensor = {
            "X": torch.from_numpy(x_grouped).float().contiguous(),
            "resnums": torch.from_numpy(resnums).int().contiguous(),
            "S": torch.from_numpy(seq_list).int().contiguous(),
            "chain_encoding_all": torch.from_numpy(chain_encoding_all + 1)
            .int()
            .contiguous(),
            "mask": mask,
            "residue_idx": residue_idx.int().contiguous(),
        }

    out_path = os.path.join(structures_dir, f"{file_name}.safetensors")

    # Merge chain_to_encoding into metadata so it is saved alongside other fields
    meta_final = {}
    if metadata:
        meta_final.update({k: str(v) for k, v in metadata.items()})
    # Prefix to avoid collisions with other metadata keys if desired; string values required by safetensors
    meta_final = meta_final | chain_to_encoding

    save_file(cur_tensor, out_path, metadata=meta_final)

    return out_path


def fetch_assembly_atoms_df(
    assembly: dict[str, str],
    *,
    chains: set[str] | None = None,
    atoms: set[str] | None = None,
    include_het: bool = False,
    prefer_label_seq: bool = True,
    canonical_by_acc: dict[str, str | None] | None = None,
) -> dict:
    """
    Download assembly mmCIF, parse to atom DataFrame, and build metadata.
    Returns (df_assemblies, metadata).
    """
    assembly_key = assembly["biological_assembly"]
    participant_protein = assembly["participant_protein"]
    affected_protein_ac = assembly["affected_protein_ac"]

    try:
        entry_id = normalize_entry(assembly_key)
        chain_map = _fetch_chain_to_uniprot_map_rcsb(entry_id)
        if chains is None:
            chains = list(
                chain_map.keys()
            )  # Making sure that we are only saving chains that are in the assembly

        cif_string = download_assembly_cif(assembly_key)
        st = _read_structure_from_text(cif_string, fmt="cif")
        block = gemmi.cif.read_string(cif_string).sole_block()
        df = _structure_to_atom_dataframe(
            st,
            chains=chains,
            atoms=atoms,
            include_het=include_het,
            prefer_label_seq=prefer_label_seq,
        )

        # Getting the representative model if available
        rep = block.find_value(
            "_pdbx_nmr_representative.conformer_id"
        )  # string or None
        model_idx = int(rep) - 1 if rep and rep.isdigit() else 0
        df = df.loc[df["model"] == model_idx].copy()
        df = df.reset_index(drop=True)

        # Restrict the chain -> UniProt map to chains that actually appear in df["chain"].
        df_chains = set(df["chain"].unique())
        chain_map = {ch: chain_map[ch] for ch in df_chains if ch in chain_map}

        # Assign UniProt residue positions by aligning each chain's observed sequence to its canonical
        # UniProt sequence. This replaces SIFTS label arithmetic (add_uniprot_from_sifts), which
        # mis-numbers biological assemblies: gemmi re-assigns label_seq on the assembly, so the SIFTS
        # offsets (described against the deposited entry) land residues in the wrong place.
        # canonical_by_acc is normally pre-fetched once by the caller (the same proteins recur across
        # thousands of assemblies); fall back to a per-assembly fetch only when called standalone.
        needed = sorted({a for a in chain_map.values() if a})
        if canonical_by_acc is None:
            canonical_by_acc = fetch_uniprot_sequences(needed)
        elif any(a not in canonical_by_acc for a in needed):
            canonical_by_acc = {
                **fetch_uniprot_sequences([a for a in needed if a not in canonical_by_acc]),
                **canonical_by_acc,
            }
        df = add_uniprot_by_alignment(df, chain_map, canonical_by_acc)
        df["item_id"] = assembly_key  # Adding assembly_key as an ID to DataFrame
        df["abs_pos_label"] = df["resnum_uniprot"]

        # Ensuring that participant_protein and affected_protein_ac are present in chain_map
        if not participant_protein in chain_map.values():
            cur_msg = f"Participant protein {participant_protein} not found in assembly {assembly_key}"
            logger.info(cur_msg)
            raise ValueError(cur_msg)
        if not affected_protein_ac in chain_map.values():
            cur_msg = f"Affected protein {affected_protein_ac} not found in assembly {assembly_key}"
            logger.info(cur_msg)
            raise ValueError(cur_msg)

        metadata = chain_map | {
            "biological_assembly": assembly_key,
            "entry_id": entry_id,
        }
        return {
            "status": "success",
            "df": df,
            "metadata": metadata,
        }
    except Exception as e:
        return {"status": str(e)}


def fetch_assemblies_atoms_parallel(
    assemblies: list[dict[str, str]],
    chains: set[str] | None = None,
    atoms: set[str] | None = None,  # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,  # include waters/ligands if True
    prefer_label_seq: bool = True,  # mmCIF label_seq where present
    max_workers: int = 8,
    show_progress: bool = True,
    parquet_dir: str | None = None,
    safetensors_dir: str | None = None,
    canonical_by_acc: dict[str, str | None] | None = None,
):
    """
    Parallel wrapper that now splits the assembly flow:
      - fetch coordinates (DataFrame + metadata) per assembly
      - optionally save coordinates as Parquet (one file per assembly)
      - optionally write safetensors using the same per-structure writer as proteins
    """
    if parquet_dir is not None:
        os.makedirs(parquet_dir, exist_ok=True)
    results: dict[str, dict] = {}

    def _task(assembly: dict[str, str]) -> tuple[str, dict]:
        assembly_key = assembly["biological_assembly"]

        try:
            out = fetch_assembly_atoms_df(
                assembly,
                chains=chains,
                atoms=atoms,
                include_het=include_het,
                prefer_label_seq=prefer_label_seq,
                canonical_by_acc=canonical_by_acc,
            )

            # Do nothing further if the download did not succeed
            if out["status"] != "success":
                return assembly_key, out

            if parquet_dir is not None:
                parquet_path = os.path.join(parquet_dir, f"{assembly_key}.parquet")
                out["df"].to_parquet(parquet_path, index=False)
                out["parquet_path"] = parquet_path
            if safetensors_dir is not None:
                st_path = _write_assemblies_safetensors(
                    out["df"], safetensors_dir, assembly_key, metadata=out["metadata"]
                )
                out["safetensors_path"] = st_path
            # include df only if not saving to reduce memory
            if parquet_dir is not None and safetensors_dir is not None:
                del out["df"]
            return assembly_key, out
        except Exception as e:
            return assembly_key, {"status": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, assembly): assembly for assembly in assemblies}
        iterator = as_completed(futs)
        if show_progress:
            iterator = track(iterator, total=len(futs), desc="Assemblies fetch")
        for fut in iterator:
            assembly_key, out = fut.result()
            results[assembly_key] = out

    return results


class IntactDataController:
    url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv"

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir
        self.df_raw: pd.DataFrame = pd.DataFrame()  # Raw input data
        self.df: pd.DataFrame = pd.DataFrame()  # Processed data

    def prepare_data(self) -> None:
        self.download_raw_data()
        self.process_raw_data()
        self.add_uniprot_sequences()

    def download_raw_data(self) -> None:
        logger.info("Downloading IntAct Mutations raw data")
        try:
            self.df_raw = pd.read_csv(
                self.url,
                sep="\t",
                on_bad_lines="warn",
                engine="python",
            )

            if self.output_dir is not None:
                self.df_raw.to_parquet(
                    os.path.join(self.output_dir, "df_intact_mutations_raw.parquet"),
                    index=False,
                )
        except Exception as e:
            logger.error(f"Error downloading IntAct Mutations raw data: {e}")

    def process_raw_data(self) -> None:
        if self.df_raw.empty:
            logger.error("No raw data available to process.")
            return

        logger.info("Processing IntAct Mutations raw data")

        # Regular expression to extract UniProtKB accession numbers
        uniprot_regex = r"(\buniprotkb:[^()]+)(?=\()"

        # Columns to drop
        cols_drop = ["PubMedID", "Figure legend", "Interaction AC"]

        # Columns to keep
        cols_features = [
            "Affected protein AC",
            "Feature type",
            "Feature range(s)",
            "Original sequence",
            "Resulting sequence",
            "Interaction participants",
        ]

        cols_features_dict = {
            x: x.lower().replace(" ", "_").replace(r"(", "").replace(r")", "")
            for x in cols_features
        }

        df_raw = self.df_raw.copy()

        # Removing unused columns
        df_raw = df_raw.drop(columns=cols_drop)

        # Splitting 'Interaction participants' column into individual rows
        df = (
            df_raw["Interaction participants"]
            .str.split("|")
            .explode()
            .to_frame("participant")
        )
        df["participant_protein"] = df["participant"].str.extract(uniprot_regex)
        df["unique_count"] = (
            df["participant_protein"].groupby(df.index).transform("nunique")
        )

        # Adding feature columns
        df = df.join(df_raw[cols_features].rename(columns=cols_features_dict))
        df.index.name = "group_id"

        # Removing non-unitprod IDs: intact, chebi, etc.
        df = df.dropna(subset=["participant_protein", "affected_protein_ac"], how="any")
        df = df.loc[~df["participant_protein"].isna(),]
        df = df.loc[df["affected_protein_ac"].str.contains("uniprot")]
        df = df.loc[df["participant_protein"].str.contains("uniprot")]

        df = df.loc[
            ~(
                (df["unique_count"] > 1)
                & (df["participant_protein"] == df["affected_protein_ac"])
            )
        ]

        df = df.drop_duplicates(
            subset=[
                "participant_protein",
                "affected_protein_ac",
                "feature_ranges",
                "original_sequence",
                "resulting_sequence",
            ]
        )

        # Removing "uniprotkb" prefix from protein IDs
        df["participant_protein"] = df["participant_protein"].str.split(":").str[-1]
        df["affected_protein_ac"] = df["affected_protein_ac"].str.split(":").str[-1]

        # Replacing NA values in 'resulting_sequence' with an empty string
        df["resulting_sequence"] = df["resulting_sequence"].fillna("")

        # Extracting start and end positions of mutations
        df["feature_ranges_start"] = (
            df["feature_ranges"].str.split("-").str[0].astype(int)
        )
        df["feature_ranges_end"] = (
            df["feature_ranges"].str.split("-").str[-1].astype(int)
        )
        df["feature_ranges_start"] -= 1  # Index is 1-based

        df = df.reset_index()

        self.df = df

    def add_uniprot_sequences(self) -> None:
        if self.df.empty:
            logger.error("No data available to fetch sequences.")
            return

        logger.info("Adding UniProt sequences to IntAct Mutations data")

        # Collecting all UniProt codes
        uniprot_codes = (
            pd.concat(
                [self.df["participant_protein"], self.df["affected_protein_ac"]],
                ignore_index=True,
            )
            .unique()
            .tolist()
        )

        # TODO fasta_map is not needed since structures are being obtained from _fetch_alphafold_structure
        fasta_map = fetch_uniprot_sequences(uniprot_codes, batch_size=100)

        self.df["participant_protein_seq"] = self.df["participant_protein"].map(
            fasta_map
        )
        self.df["affected_protein_ac_seq"] = self.df["affected_protein_ac"].map(
            fasta_map
        )

        self.df = self.df.dropna(
            subset=["participant_protein_seq", "affected_protein_ac_seq"],
            how="any",
            ignore_index=True,
        )

        # Creating the mutation sequence by replacing the original sequence with the resulting sequence; note that
        # SEQUENCE_DELETION ('.') in the replacement sequence indicates a deletion
        self.df["affected_protein_ac_seq_mut"] = self.df.apply(
            lambda x: (
                x["affected_protein_ac_seq"][: x["feature_ranges_start"]]
                + x["resulting_sequence"]
                + x["affected_protein_ac_seq"][x["feature_ranges_end"] :]
            ).replace(SEQUENCE_DELETION, ""),
            axis=1,
        )

        if self.output_dir is not None:
            self.df.to_parquet(
                os.path.join(self.output_dir, "df_intact_mutations.parquet"),
                index=False,
            )
