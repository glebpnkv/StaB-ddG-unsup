"""Progress reporting that adapts to the runtime.

Interactive terminals and notebooks get a live ``tqdm`` bar. Non-interactive pipeline runs — where
tqdm's carriage-return redraws are invisible or garbled in CloudWatch (it never emits a newline until
the loop ends) — instead get periodic **newline-terminated** log lines, so long fetch loops still
report intermediate progress that a line-based log aggregator can render.

The pipeline path is selected when ``TQDM_DISABLE`` is set (we set it in the SageMaker containers),
so local/notebook behaviour is unchanged. Compared to plain ``TQDM_DISABLE=1`` (which silences tqdm
entirely) this keeps progress visibility without the ``\\r`` spam.
"""
import logging
import os
import time
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)


def track(
    iterable: Iterable,
    total: Optional[int] = None,
    desc: str = "",
    log_every_sec: float = 30.0,
    log_every_n: Optional[int] = None,
) -> Iterator:
    """Yield from ``iterable`` while reporting progress.

    * Interactive (``TQDM_DISABLE`` unset): a ``tqdm.auto`` bar (terminal or notebook widget).
    * Non-interactive (``TQDM_DISABLE`` set): a log line every ``log_every_sec`` seconds (and/or
      every ``log_every_n`` items), plus a final summary line.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    if not os.environ.get("TQDM_DISABLE"):
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, desc=desc)
        return

    start = last = time.monotonic()
    n = 0
    label = desc or "progress"
    logger.info("%s: starting%s", label, f" ({total} items)" if total else "")
    for item in iterable:
        yield item
        n += 1
        now = time.monotonic()
        due = (now - last) >= log_every_sec or (log_every_n is not None and n % log_every_n == 0)
        if due:
            rate = n / max(now - start, 1e-9)
            if total:
                logger.info("%s: %d/%d (%.1f%%) at %.1f/s", label, n, total, 100.0 * n / total, rate)
            else:
                logger.info("%s: %d at %.1f/s", label, n, rate)
            last = now
    dur = time.monotonic() - start
    logger.info("%s: done %d%s in %.0fs", label, n, f"/{total}" if total else "", dur)
