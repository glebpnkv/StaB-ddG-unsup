import gc
import logging
import multiprocessing as mp
import operator as op
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import reduce

import gemmi
import numpy as np
import pandas as pd
import requests
import torch
from rcsbapi.data import DataQuery as DataQ
from rcsbapi.model import ModelQuery
from rcsbapi.search import AttributeQuery, NestedAttributeQuery
from requests.adapters import HTTPAdapter
from safetensors.torch import save_file
from tqdm import tqdm
from urllib3.util.retry import Retry

from stabddg.constants import AA3_TO_1, COORDS_ORDER
from stabddg.uniprot import fetch_uniprot_sequences

logging.basicConfig(level=logging.INFO)
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
    s.headers.update({"User-Agent": "stabddg/parallel-fetch (https://alphafold.ebi.ac.uk/)"})
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


def _extract_ca_xyz_and_annotations(st: gemmi.Structure) -> dict:
    """Walk all models/chains/residues and collect CA coordinates.

    Returns
    -------
    xyz : np.ndarray
        Array of shape (N, 3) with float64 coordinates for CA atoms.
    ann : list[tuple[str, int, str, str]]
        Parallel annotations per CA: (chain_id, uniprot_like_index, aa3, aa1).
        The residue index is taken from `label_seq` when available (UniProt-like
        numbering in mmCIF); otherwise fall back to the author seqid.
    """
    ca: list[list[float]] = []
    ann: list[dict[str, int | str]] = []

    for model in st:  # gemmi.Model
        for chain in model:  # gemmi.Chain
            for res in chain:  # gemmi.Residue
                at = res.get_ca()  # returns None for non-standard/hetero residues
                if at is None:
                    continue
                # Coordinates (x, y, z) in Ångström
                ca.append([at.pos.x, at.pos.y, at.pos.z])

                # Prefer label_seq (UniProt-like) if present; otherwise use seqid.num
                unp_idx = res.label_seq if getattr(res, "label_seq", 0) != 0 else res.seqid.num

                aa3 = res.name.upper()
                aa1 = AA3_TO_1.get(aa3, "X")
                # ann.append((chain.name, int(unp_idx), aa3, aa1))
                ann.append({
                    "chain_name": chain.name,
                    "uniprot_like_index": int(unp_idx),
                    "aa3": aa3,
                    "aa1": aa1,
                })

    out = {"ca": np.asarray(ca, dtype=float)} | {"ann": ann}

    return out


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
                # Label sequence index (UniProt-like in mmCIF) if present and non-zero
                label_seq = getattr(res, "label_seq", 0) or 0
                resnum_label = int(label_seq) if (prefer_label_seq and label_seq != 0) else None

                for at in res:  # gemmi.Atom
                    atom_name = at.name.strip()
                    if atoms is not None and atom_name not in atoms:
                        continue

                    rows.append({
                        "model": model_idx,
                        "chain": chain_id,
                        "res_name_3": aa3,
                        "res_name_1": aa1,
                        "resnum_label": resnum_label,
                        "resnum_auth": auth_num,
                        "ins_code": ins_code,
                        "atom_name": atom_name,
                        "element": at.element.name,
                        "altloc": at.altloc if hasattr(at, "altloc") else "",
                        "occupancy": float(at.occ),
                        "b_factor": float(at.b_iso),
                        "x": float(at.pos.x),
                        "y": float(at.pos.y),
                        "z": float(at.pos.z),
                        "is_het": is_het,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1) AlphaFold DB — wild type by UniProt ID -> (N,3) + annotations
#    Follows your template and prefers mmCIF when available.
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


def _fetch_alphafold_xyz(uniprot_acc: str, session: requests.Session | None = None) -> dict:
    """Return (N,3) CA coords and annotations: (chain, uniprot_idx, aa3, aa1)."""
    # Keep existing mmCIF preference for XYZ
    st, url, fmt = _fetch_alphafold_structure(uniprot_acc, session=session, prefer_format="cif")
    out = _extract_ca_xyz_and_annotations(st)
    out["source"] = "alphafold"
    return out


def fetch_alphafold_xyz(uniprot_or_pro: str) -> dict:
    """
    Accepts either a plain UniProt accession/isoform (e.g. 'O92972' or 'O92972-1')
    or a proteoform-like code 'O92972-PRO_0000278753'. Returns (N,3) CA coords + annotations.
    """
    out = {}

    m = re.match(r"^(?P<acc>[A-Z0-9]+(?:-\d+)?)-(?P<pro>PRO_\d+)$", uniprot_or_pro)
    if not m:
        # No PRO_ suffix: just use AlphaFold DB directly
        try:
            out = _fetch_alphafold_xyz(uniprot_or_pro)
            out["status"] = "success"
        except Exception as e:
            status = str(e)
            out["status"] = status

        return out

    out["status"] = "unresolved"
    return out


# ---------------------------------------------------------------------------
# Atom dataframe helper (PDB or CIF) for AlphaFold entries
# ---------------------------------------------------------------------------

def fetch_alphafold_atoms(
    uniprot_or_pro: str,
    *,
    prefer_format: str = "pdb",           # "pdb" if you want strict compatibility with PDB-centric tooling
    chains: set[str] | None = None,
    atoms: set[str] | None = None,        # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,            # include waters/ligands if True
    prefer_label_seq: bool = True,        # mmCIF label_seq where present
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
        st, url, fmt = _fetch_alphafold_structure(uniprot_or_pro, session=ses, prefer_format=prefer_format)
        df = _structure_to_atom_dataframe(
            st,
            chains=chains,
            atoms=atoms,
            include_het=include_het,
            prefer_label_seq=prefer_label_seq,
        )
        return {
            "status": "success",
            "df": df,
            "source_url": url,
            "format": fmt,
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
):
    results: dict[str, dict] = {}

    def _task(code: str) -> tuple[str, dict]:
        out = fetch_alphafold_atoms(
            code,
            prefer_format=prefer_format,
            chains=chains,
            atoms=atoms,
            include_het=include_het,
            prefer_label_seq=prefer_label_seq,
        )
        return code, out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, code): code for code in uniprot_codes}
        iterator = as_completed(futs)
        if show_progress:
            iterator = tqdm(iterator, total=len(futs), desc="AlphaFold fetch")
        for fut in iterator:
            code, out = fut.result()
            results[code] = out

    return results


# ---------------------------------------------------------------------------
# 2) Assemblies helpers
# ---------------------------------------------------------------------------

def _search_assemblies_for_uniprots(
    uniprots: list[str],
    protein_only: bool = True
) -> list[str]:
    """
    Return assembly IDs that (a) contain the supplied UniProt accession(s),
    and (b) have ≥2 protein chains (i.e., are potentially polymers).
    - Heteromer (len==2): require ≥2 protein chains; allow ≥2 entities.
    - Homomer (len==1):  require ≥2 protein chains AND exactly 1 polymer entity.
    """
    out = []

    if not uniprots:
        return out

    groups = []
    for up in uniprots:
        q_acc = AttributeQuery(
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            operator="exact_match",
            value=up
        )
        q_db  = AttributeQuery(
            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
            operator="exact_match",
            value="UniProt"
        )
        groups.append(NestedAttributeQuery(q_acc, q_db))  # Creates the required nested group

    # Assembly-level filters
    at_least_two_protein_chains = AttributeQuery(
        "rcsb_assembly_info.polymer_entity_instance_count_protein",
        operator="greater_or_equal", value=2
    )  # ≥2 protein chains in the biological assembly

    # Distinct polymer entities condition
    if len(uniprots) == 1:   # homomer
        entity_count_q = AttributeQuery(
            "rcsb_assembly_info.polymer_entity_count",
            operator="equals", value=1
        )  # one entity repeated → true homomer
    else:  # heteromer (2+)
        entity_count_q = AttributeQuery(
            "rcsb_assembly_info.polymer_entity_count",
            operator="greater_or_equal", value=2
        )

    extra = [at_least_two_protein_chains, entity_count_q]
    if protein_only:
        extra.append(AttributeQuery(
            "rcsb_entry_info.selected_polymer_entity_types",
            operator="exact_match", value="Protein (only)"
        ))  # optional cleanup

    # Build final AND query
    all_terms = groups + extra
    query = reduce(op.and_, all_terms)

    try:
        # AND the two UniProt groups; request assemblies
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
    q = DataQ(
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
            em_res = [d.get("resolution") for d in em if d.get("resolution") is not None]
            if em_res:
                res = min(em_res)

        # UniProt set present in entry polymer entities
        ups = set()
        for pe in (e.get("polymer_entities") or []):
            cont = (pe.get("rcsb_polymer_entity_container_identifiers") or {})
            for ref in (cont.get("reference_sequence_identifiers") or []):
                if ref.get("database_name") == "UniProt" and ref.get("database_accession"):
                    ups.add(ref["database_accession"])
        out[pdb_id] = {"method": method, "best_resolution": res, "uniprots": ups}
    return out


def _pdbe_mutations(pdb_id):
    """
    PDBe Graph API mutated/modified residues. Empty dict means none reported.
    """
    muts = {}
    for ep in ("mutated_residues", "modified_residues"):
        u = f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/{ep}/{pdb_id}"
        try:
            r = requests.get(u, timeout=30)
            if r.ok:
                muts.update(
                    (r.json() or {}).get(pdb_id, {})
                )
        except Exception:
            pass
    return muts


def _assemblies_uniprot_instance_counts(assembly_ids: list[str]) -> dict[str, dict[str, int]]:
    """
    For each assembly (e.g. '4HHB-1'), count how many polymer_entity_instances
    map to each UniProt accession.
    Returns: { '4HHB-1': {'P69905':2, 'P68871':2}, ... }
    """
    if not assembly_ids:
        return {}

    q = DataQ(
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
        for inst in (asm.get("polymer_entity_instances") or []):
            poly = inst.get("polymer_entity") or {}
            cont = poly.get("rcsb_polymer_entity_container_identifiers") or {}
            for ref in (cont.get("reference_sequence_identifiers") or []):
                if ref.get("database_name") == "UniProt" and ref.get("database_accession"):
                    counts[ref["database_accession"]] += 1
                    break  # count each instance once
        out[asm_id] = dict(counts)
    return out


def _filter_assemblies_by_uniprot_counts(assembly_ids: list[str], uniprots: list[str]) -> list[str]:
    """
    Keep assemblies only if they truly satisfy the stoichiometry:
      - heteromer (2 IDs): each present in ≥1 instance
      - homomer (1 ID): that ID present in ≥2 instances
    """
    want = list(uniprots)
    counts = _assemblies_uniprot_instance_counts(assembly_ids)
    keep = []
    if len(want) == 1:
        up = want[0]
        for aid in assembly_ids:
            if counts.get(aid, {}).get(up, 0) >= 2:
                keep.append(aid)
    else:
        a, b = want[0], want[1]
        for aid in assembly_ids:
            c = counts.get(aid, {})
            if c.get(a, 0) >= 1 and c.get(b, 0) >= 1:
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
        method_bonus = 0 if "X-RAY" in method else (0.25 if "ELECTRON" in method else 0.5)
        res_use = res if isinstance(res, (int, float)) else 9.99

        score = (0 if has_both else 10) + mut_penalty + method_bonus + (res_use / 10.0)
        ranked.append({
            "biological_assembly": aid,
            "score": score,
            "method": m.get("method"),
            "resolution": res,
            "has_both": has_both,
            "mutations": muts
        })
    return sorted(ranked, key=lambda x: x["score"])


def fetch_assemblies_for_uniprots(uniprots: list[str]) -> list[dict[str, str | float | bool]] | None:
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
            logger.error(f"Error fetching assemblies for UniProt accessions {pair}: {e}")
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
            iterator = tqdm(iterator, total=len(futs), desc="Assemblies fetch")
        for fut in iterator:
            idx, (u1, u2), ranked = fut.result()
            if ranked is None or len(ranked) == 0:
                # One row with NaNs for assembly fields
                rows.append({
                    "pair_idx": idx,
                    "participant_protein": u1,
                    "affected_protein_ac": u2,
                    "assembly_id": None,
                    "score": None,
                    "method": None,
                    "resolution": None,
                    "has_both": None,
                    "mutations": None,
                })
            else:
                for r in ranked:
                    rows.append({
                        "pair_idx": idx,
                        "participant_protein": u1,
                        "affected_protein_ac": u2,
                        **r,
                    })

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


def list_assembly_ids(entry_id: str) -> list[str]:
    entry = normalize_entry(entry_id)
    q = AttributeQuery(
        "rcsb_entry_container_identifiers.entry_id",
        operator="exact_match",
        value=entry,
    )
    # Returns ["8CT8-1", "8CT8-2", ...]
    asm_ids = list(q.exec(return_type="assembly"))
    return [aid.split("-")[1] for aid in asm_ids]  # -> ["1","2",...]


def download_all_assemblies(entry_id: str) -> dict[str, str]:
    """
    Download every biological assembly for the entry.
    Returns a dict keyed by '8CT8-1', '8CT8-2', ... with CIF text values.
    """
    entry = normalize_entry(entry_id)
    asm_nums = list_assembly_ids(entry)
    mq = ModelQuery()

    out = {}
    for a in asm_nums:
        # Model Server assembly endpoint; encoding 'cif' yields mmCIF
        out[f"{entry}-{a}"] = mq.get_assembly(
            entry_id=entry,        # NOTE: entry only
            name=a,                # <-- assembly id goes here
            encoding="cif",
            copy_all_categories=True,
        )

    return out


def _write_assemblies_safetensors(
    assemblies_dict: dict[str, pd.DataFrame],
    structures_dir: str
) -> list[tuple[str, str]]:
    """
    Write a safetensors file per entry assembly set.

    - Input assemblies_dict keys look like 'ASM-1', 'ASM-2', ...
    - Output tensor keys will be '1', '2', ... (suffix after the last dash)
    - Each tensor value is a stacked (n_atoms, 3) float32 array of coordinates
      grouped by atom_name after filtering to COORDS_ORDER and sorting by
      ['resnum_auth', 'atom_name'] similarly to _worker_write_safetensors_batch.
    - The resulting file name is '{asm_id}.safetensors' saved into structures_dir.
    """
    # Keep PyTorch single-threaded to avoid CPU oversubscription
    torch.set_num_threads(1)
    os.makedirs(structures_dir, exist_ok=True)

    results: list[tuple[str, str]] = []
    for key, df in assemblies_dict.items():
        df_filt = df.loc[df["atom_name"].isin(COORDS_ORDER)]

        # Build per-atom tensors in the same structure as _worker_write_safetensors_batch
        if df_filt.empty:
            cur_tensor = {a: torch.empty((0, 3), dtype=torch.float32) for a in COORDS_ORDER}
        else:
            grouped = (
                df_filt.sort_values(by=["resnum_auth", "atom_name"])
                      .groupby("atom_name")[["x", "y", "z"]]
                      .apply(lambda x: x.values)
                      .to_dict()
            )
            cur_tensor = {
                a: torch.from_numpy(grouped[a]).float().contiguous() if a in grouped
                   else torch.empty((0, 3), dtype=torch.float32)
                for a in COORDS_ORDER
            }

        out_path = os.path.join(structures_dir, f"{key}.safetensors")
        save_file(cur_tensor, out_path)
        results.append((key, out_path))

    return results


def fetch_assemblies_atoms(
    asm_id: str,
    chains: set[str] | None = None,
    atoms: set[str] | None = None,  # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,  # include waters/ligands if True
    prefer_label_seq: bool = True,  # mmCIF label_seq where present
    structures_dir: str | None = None,
) -> dict:
    try:
        cif_string_dict = download_all_assemblies(asm_id)

        st_dict = {
            k: _read_structure_from_text(v, "cif")
            for k, v in cif_string_dict.items()
        }

        df_assemblies_dict = {
            k: _structure_to_atom_dataframe(
                st,
                chains=chains,
                atoms=atoms,
                include_het=include_het,
                prefer_label_seq=prefer_label_seq,
            )
            for k, st in st_dict.items()
        }

        out = {
            "status": "success",
            "assemblies_dict": df_assemblies_dict
        }

        if structures_dir is not None:
            # Write a single safetensors file with keys as assembly suffixes
            written = _write_assemblies_safetensors(df_assemblies_dict, structures_dir)
            out["safetensors_paths"] = dict(written)

        return out
    except Exception as e:
        return {"status": str(e)}


def fetch_assemblies_atoms_parallel(
    asm_ids: list[str],
    chains: set[str] | None = None,
    atoms: set[str] | None = None,  # e.g., {"N","CA","C","O"} or None for all atoms
    include_het: bool = False,  # include waters/ligands if True
    prefer_label_seq: bool = True,  # mmCIF label_seq where present
    max_workers: int = 8,
    show_progress: bool = True,
):
    results: dict[str, dict] = {}

    def _task(code: str) -> tuple[str, dict]:
        out = fetch_assemblies_atoms(
            code,
            chains=chains,
            atoms=atoms,
            include_het=include_het,
            prefer_label_seq=prefer_label_seq,
        )
        return code, out

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_task, code): code for code in asm_ids}
        iterator = as_completed(futs)
        if show_progress:
            iterator = tqdm(iterator, total=len(futs), desc="Assemblies fetch")
        for fut in iterator:
            code, out = fut.result()
            results[code] = out

    return results


class IntactDataController:
    url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv"

    def __init__(
        self,
        output_dir: str | None = None
    ):
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
                on_bad_lines='warn',
                engine="python",
            )

            if self.output_dir is not None:
                self.df_raw.to_parquet(os.path.join(self.output_dir, "df_intact_mutations_raw.parquet"), index=False)
        except Exception as e:
            logger.error(f"Error downloading IntAct Mutations raw data: {e}")

    def process_raw_data(self) -> None:
        if self.df_raw.empty:
            logger.error("No raw data available to process.")
            return

        logger.info("Processing IntAct Mutations raw data")

        # Regular expression to extract UniProtKB accession numbers
        uniprot_regex = r'(\buniprotkb:[^()]+)(?=\()'

        # Columns to drop
        cols_drop = [
            "PubMedID",
            "Figure legend",
            "Interaction AC"
        ]

        # Columns to keep
        cols_features = [
            "Affected protein AC",
            "Feature type",
            "Feature range(s)",
            "Original sequence",
            "Resulting sequence",
            "Interaction participants"
        ]

        cols_features_dict = {
            x: x.lower().replace(" ", "_").replace(r"(", "").replace(r")", "")
            for x in cols_features
        }

        df_raw = self.df_raw.copy()

        # Removing unused columns
        df_raw = df_raw.drop(
            columns=cols_drop
        )

        # Splitting 'Interaction participants' column into individual rows
        df = df_raw['Interaction participants'].str.split("|").explode().to_frame("participant")
        df["participant_protein"] = (
            df["participant"]
            .str.extract(uniprot_regex)
        )
        df["unique_count"] = df['participant_protein'].groupby(df.index).transform("nunique")

        # Adding feature columns
        df = df.join(
            df_raw[cols_features].rename(
                columns=cols_features_dict
            )
        )
        df.index.name = "group_id"

        # Removing non-unitprod IDs: intact, chebi, etc.
        df = df.dropna(
            subset=["participant_protein", "affected_protein_ac"],
            how="any"
        )
        df = df.loc[
            ~df["participant_protein"].isna(),
        ]
        df = df.loc[
            df["affected_protein_ac"].str.contains("uniprot")
        ]
        df = df.loc[
            df["participant_protein"].str.contains("uniprot")
        ]

        df = df.loc[
            ~(
                (df["unique_count"] > 1) &
                (df["participant_protein"] == df["affected_protein_ac"])
            )
        ]

        df = df.drop_duplicates(
            subset=[
                "participant_protein",
                "affected_protein_ac",
                "feature_ranges",
                "original_sequence",
                "resulting_sequence"
            ]
        )

        # Removing "uniprotkb" prefix from protein IDs
        df["participant_protein"] = df["participant_protein"].str.split(":").str[-1]
        df["affected_protein_ac"] = df["affected_protein_ac"].str.split(":").str[-1]

        # Replacing NA values in 'resulting_sequence' with an empty string
        df["resulting_sequence"] = df["resulting_sequence"].fillna("")

        # Extracting start and end positions of mutations
        df['feature_ranges_start'] = df['feature_ranges'].str.split('-').str[0].astype(int)
        df['feature_ranges_end'] = df['feature_ranges'].str.split('-').str[-1].astype(int)
        df['feature_ranges_start'] -= 1  # Index is 1-based

        df = df.reset_index()

        self.df = df

    def add_uniprot_sequences(self) -> None:
        if self.df.empty:
            logger.error("No data available to fetch sequences.")
            return

        logger.info("Adding UniProt sequences to IntAct Mutations data")

        # Collecting all UniProt codes
        uniprot_codes = pd.concat(
            [
                self.df["participant_protein"],
                self.df["affected_protein_ac"]
            ],
            ignore_index=True
        ).unique().tolist()

        fasta_map = fetch_uniprot_sequences(uniprot_codes, batch_size=100)

        self.df["participant_protein_seq"] = self.df["participant_protein"].map(fasta_map)
        self.df["affected_protein_ac_seq"] = self.df["affected_protein_ac"].map(fasta_map)

        self.df = self.df.dropna(
            subset=["participant_protein_seq", "affected_protein_ac_seq"],
            how="any",
            ignore_index=True
        )
        self.df["affected_protein_ac_seq_mut"] = (
            self.df.apply(
                lambda x: (
                    x["affected_protein_ac_seq"][:x["feature_ranges_start"]] +
                    x["resulting_sequence"] +
                    x["affected_protein_ac_seq"][x["feature_ranges_end"]:]
                ),
                axis=1
            )
        )

        if self.output_dir is not None:
            self.df.to_parquet(os.path.join(self.output_dir, "df_intact_mutations.parquet"), index=False)

def _worker_write_safetensors(cur_uniprot, df_structures, structures_dir):
    # Keep PyTorch single-threaded in workers to avoid CPU oversubscription
    torch.set_num_threads(1)

    df_sample = df_structures.loc[df_structures["uniprot_code"] == cur_uniprot]

    # Creating a dict of arrays per sidechain atom
    cur_tensor = df_sample.sort_values(
        by=["resnum_auth", "atom_name"]
    ).groupby("atom_name")[["x", "y", "z"]].apply(lambda x: x.values).to_dict()

    cur_tensor = {
        k: torch.from_numpy(v).float().contiguous()
        for k, v in cur_tensor.items()
    }

    out_path = os.path.join(structures_dir, f"{cur_uniprot}.safetensors")
    save_file(cur_tensor, out_path)
    return cur_uniprot, out_path

def _worker_write_safetensors_batch(batch_items: list[tuple[str, pd.DataFrame]], structures_dir: str) -> list[tuple[str, str]]:
    """
    Process a batch of (uniprot_code, df_group) pairs in one worker.
    Returns list of (code, out_path).
    """
    # Avoid CPU oversubscription inside each process
    torch.set_num_threads(1)
    out: list[tuple[str, str]] = []
    for cur_uniprot, df_group in batch_items:
        # Same logic as the single-item worker, but on the per-code slice
        cur_tensor = (
            df_group.sort_values(by=["resnum_auth", "atom_name"])
                    .groupby("atom_name")[["x", "y", "z"]]
                    .apply(lambda x: x.values)
                    .to_dict()
        )
        cur_tensor = {k: torch.from_numpy(v).float().contiguous() for k, v in cur_tensor.items()}
        out_path = os.path.join(structures_dir, f"{cur_uniprot}.safetensors")
        save_file(cur_tensor, out_path)
        out.append((cur_uniprot, out_path))
    return out

def _chunked(seq, n):
    """Yield successive chunks of size n from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def write_all_safetensors_parallel(
    df_structures: pd.DataFrame,
    structures_dir: str,
    max_workers: int | None = None,
    batch_size: int = 64,   # number of UniProt codes per worker task
    limit_codes: int | None = None,  # optional for debugging
):
    """
    Faster parallel writer:
      - pre-splits df by uniprot_code once
      - sends batches of groups to each worker to amortize overhead
      - uses 'fork' start method on Unix to reduce serialization overhead
    """
    os.makedirs(structures_dir, exist_ok=True)

    # Pre-split once; each item is (code, df_slice)
    groups = [(code, g.copy(deep=False)) for code, g in df_structures.groupby("uniprot_code", sort=False)]
    if limit_codes is not None:
        groups = groups[:limit_codes]

    total = len(groups)
    if total == 0:
        return [], []

    # Decide workers
    max_workers = max_workers or os.cpu_count() or 1

    # Reasonable default for batch size if user leaves it small/large
    if batch_size <= 0:
        batch_size = max(16, (total // (max_workers * 8)) or 1)

    # Chunk the work
    batches = list(_chunked(groups, batch_size))

    results: list[tuple[str, str]] = []
    errors: list[tuple[str, Exception]] = []

    # Use fork context on Unix to reduce pickling overhead
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        # Non-Unix fallback
        ctx = mp.get_context()

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker_write_safetensors_batch, batch, structures_dir): len(batch) for batch in batches}
        # Progress in number of codes completed, not number of batches
        pbar = tqdm(total=total, desc="Writing safetensors (batched)", smoothing=0.1)
        for fut in as_completed(futs):
            batch_len = futs[fut]
            try:
                out_list = fut.result()
                results.extend(out_list)
            except Exception as e:
                # We don't know which codes failed inside the batch; log the batch size
                errors.append((f"batch_size={batch_len}", e))
            finally:
                pbar.update(batch_len)
        pbar.close()

    # Cleaning up
    del groups, batches
    gc.collect()

    return results, errors
