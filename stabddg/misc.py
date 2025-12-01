def optional_int(value: str | None) -> int | None:
    """
    argparse type that treats 'null', 'none', and empty string as None.
    Otherwise returns int(value).
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"", "none", "null"}:
        return None
    return int(value)
