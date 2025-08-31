# AA3_TO_1 is a mapping from the 3-letter amino acid codes (e.g., "ALA") to their
# corresponding 1-letter codes (e.g., "A"). It includes:
# - The 20 canonical amino acids
# - Selenocysteine ("SEC" -> "U") and pyrrolysine ("PYL" -> "O")
# - Ambiguous codes: "ASX" -> "B" (Asn/Asp), "GLX" -> "Z" (Gln/Glu), "XLE" -> "J" (Leu/Ile)
# - "UNK" -> "X" for unknown/unspecified residues
# Keys are uppercase; callers should normalize inputs (e.g., using .upper()) when looking up.
AA3_TO_1: dict[str, str] = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O","ASX":"B","GLX":"Z",
    "XLE":"J","UNK":"X"
}

# ALPHABET is the collection of 1-letter amino acid codes used elsewhere in the codebase.
# It is derived from the values of AA3_TO_1. The ordering matches the insertion order of AA3_TO_1.
ALPHABET: list[str] = list(AA3_TO_1.values())
