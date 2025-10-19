import re
from collections import defaultdict

import requests
from tqdm.auto import tqdm

BASE = "https://rest.uniprot.org"
UA = {"User-Agent": "uniprot-bulk-fetch/1.0 (+python-requests)"}

def _batched(xs: list[str], n:int = 50):
    buf = []
    for x in tqdm(xs):
        x = x.strip()
        if not x:
            continue
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def _parse_fasta_to_map(fasta_text):
    """Return {accession_or_isoform: sequence_without_newlines}"""
    m = {}
    acc = None
    seq_lines = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            # Flush previous
            if acc is not None:
                m.setdefault(acc, "".join(seq_lines))
            # Parse accession; UniProt headers are >sp|Q16513-2|...
            parts = line.split("|")
            if len(parts) >= 3 and (parts[0].endswith("sp") or parts[0].endswith("tr")):
                acc = parts[1]
            else:
                # Fallback: grab first ACC-like token (captures isoform too)
                hit = re.search(r"\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?\b", line)
                acc = hit.group(0) if hit else None
            seq_lines = []
        else:
            if acc is not None:
                seq_lines.append(line.strip())
    if acc is not None:
        m.setdefault(acc, "".join(seq_lines))
    return m

def _fetch_stream_map(query_terms, include_isoforms=False, timeout=120):
    """query_terms: list of already-scoped terms like accession:Q..., accession_id:Q...-2"""
    if not query_terms:
        return {}
    q = " OR ".join(f"({t})" for t in query_terms)
    params = {"query": q, "format": "fasta", "compressed": "false"}
    if include_isoforms:
        params["includeIsoform"] = "true"
    r = requests.get(f"{BASE}/uniprotkb/stream", params=params, headers=UA, timeout=timeout)
    try:
        r.raise_for_status()
        out = _parse_fasta_to_map(r.text)
    except Exception as e:
        print(f"Error processing query {query_terms}")
        print(f"{e}")
        out = None
    return out

def _fetch_pro_components(pro_ids, timeout=60):
    """Return {PRO-id: sequence} by slicing parent entry features."""
    out = {}
    by_parent = defaultdict(list)
    for pid in pro_ids:
        acc, _, pro = pid.partition("-PRO_")
        if not pro:
            continue
        by_parent[acc].append(f"PRO_{pro}")

    for acc, want in tqdm(by_parent.items()):
        j = requests.get(f"{BASE}/uniprotkb/{acc}.json", headers=UA, timeout=timeout).json()
        full = j["sequence"]["value"]
        feats = j.get("features", [])
        idx = {f["featureId"]: f for f in feats if f.get("featureId") in want}
        for fid, f in idx.items():
            # Only slice well-defined ranges
            loc = f.get("location", {})
            try:
                start = int(loc["start"]["value"]) - 1
                end = int(loc["end"]["value"])
                out[f"{acc}-{fid}"] = full[start:end]
            except Exception:
                out[f"{acc}-{fid}"] = None
    return out

def fetch_uniprot_sequences(
    ids: list[str],
    batch_size: int = 50
) -> dict[str, str]:
    """
    ids: list like ["Q16513","Q16513-2","O92972-PRO_0000278753"]
    returns: list[str|None] same order, sequences are unwrapped (no \\n)
    """
    # Categorize
    plain, iso, pro = [], [], []
    for x in ids:
        if "-PRO_" in x:
            pro.append(x)
        elif "-" in x:
            iso.append(x)
        else:
            plain.append(x)

    # Fetch plain accessions
    plain_map = {}
    for batch in _batched(plain, n=batch_size):
        terms = [f"accession:{a}" for a in batch]  # no isoforms
        update = _fetch_stream_map(terms, include_isoforms=False)
        if update is not None:
            plain_map.update(update)

    # Fetch isoforms
    iso_map = {}
    for batch in _batched(iso, n=batch_size):
        terms = [f"accession_id:{a}" for a in batch]  # exact isoform ids
        update = _fetch_stream_map(terms, include_isoforms=False)
        if update is not None:
            iso_map.update(update)

    # Fetch PRO components (slice)
    pro_map = _fetch_pro_components(pro)

    # Stitch in input order
    result = {}
    for x in ids:
        if "-PRO_" in x:
            cur_val = pro_map.get(x)
        elif "-" in x:
            cur_val = iso_map.get(x)
        else:
            cur_val = plain_map.get(x)
        # None values are included in the result
        result[x] = cur_val

    return result
